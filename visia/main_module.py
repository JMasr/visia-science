"""
Description
===========

This is a module for running the Visia-Science main Pipeline.
"""

import argparse
import os
import time
from typing import Annotated, Optional, Any

import librosa.feature
import mlflow
import numpy as np
import pandas as pd
from joblib import parallel_backend
from mlflow import MlflowClient
from numpy import ndarray
from pydantic import BaseModel, Field, ConfigDict
from sklearn import impute
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, f1_score, recall_score, precision_score, accuracy_score, \
    precision_recall_fscore_support
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from config.visia_config import BasicConfig
from database import BasicDataBase
from dataset.dataset import AudioStream
from files import read_audio, load_file, save_file
from logger import BasicLogger
from responses.basic_responses import BasicResponse, DataResponse


class BasicExperiment:
    """
    This class defines the experiment object. An experiment needs to have a config file, a model and a dataset.
    The results of the experiment are stored in the results' folder.
    """

    def __init__(self, config_path: str):
        """
        Initialize the class with the specified configuration.

        :param config_path: Path to the configuration file.
        :type config_path: str
        """
        self.exp_config = BasicConfig(path_to_config=config_path)

        # CONFIGURE THE LOGGER
        self.logger = BasicLogger(**self.exp_config.log_config).get_logger()
        self.logger.info(f"Logger created - 200 - Parameters: {self.exp_config.model_dump()}\n")

        # CONFIGURE THE MLFLOW SERVER
        self.mlflow_server = MLFlowServer(mlflow_config=self.exp_config.mlflow_config)
        mlflow_response = self.mlflow_server.is_up()
        if mlflow_response.success:
            self.logger.info(mlflow_response.get_as_str(module="MainModule", action="Check MLFlow server"))
        else:
            self.logger.error(mlflow_response.get_as_str(module="MainModule", action="Check MLFlow server"))

        # SET THE DATASET
        self.dataset = DataSet(**self.exp_config.data_config)
        dset_response = self.dataset.load_dataset(standardization_method_odyssey)
        if dset_response.success:
            self.logger.info(dset_response.get_as_str(module="MainModule", action="Load dataset"))
        else:
            self.logger.error(dset_response.get_as_str(module="MainModule", action="Load dataset"))

        # CONFIGURE THE DATABASE
        if self.exp_config.db_config.get("dset2db", False):
            # Set the database
            self.db = BasicDataBase(**self.exp_config.db_config)
            db_response = self.db.check_connection()
            if db_response.success:
                self.logger.info(db_response.get_as_str(module="MainModule", action="Check connection"))
            else:
                self.logger.error(db_response.get_as_str(module="MainModule", action="Check connection"))

            # Load the data to the database
            db_response = self.data2database()
            if db_response.success:
                self.logger.info(db_response.get_as_str(module="MainModule", action="Load data to database"))
            else:
                self.logger.error(db_response.get_as_str(module="MainModule", action="Load data to database"))

        # TRAIN THE MODEL
        train_config = self.exp_config.train_config
        train_response = self.train_loop(train_config)
        if train_response.success:
            self.logger.info(train_response.get_as_str(module="MainModule", action="Train model"))
        else:
            self.logger.error(train_response.get_as_str(module="MainModule", action="Train model"))

    def data2database(self) -> BasicResponse:
        """
        Load the data from the dataset to the database.

        :return: BasicResponse with the status of the process.
        :rtype: BasicResponse
        """
        self.logger.info(f"Loading {self.dataset.data_name} to {self.db.db_url}")
        if self.check_if_dataset_is_in_database().success:
            return BasicResponse(success=True,
                                 status_code=200,
                                 message="Dataset is already in the database")
        else:
            self.logger.info("Dataset isn't in the database, loading data...")
            try:
                metadata = self.dataset.metadata.copy()
                total_rows = len(metadata)
                # Load the data to the database
                for i, r in tqdm(metadata.iterrows(), total=total_rows, desc="Processing data"):
                    # Load the audio
                    path_to_audio = r["Path"]
                    audio = AudioStream.from_file_path(file_path=path_to_audio)
                    # add audio's information to the
                    metadata.at[i, "Duration"] = audio.duration
                    metadata.at[i, "SampleRate"] = audio.sample_rate

                    # save the audio in the database
                    db_entry = metadata.iloc[i].to_dict()
                    db_entry["raw_data"] = audio.data
                    db_entry["id"] = metadata.iloc[i].FileName.split(".")[0]
                    self.db.collection.insert_one(db_entry)

                    # delete the audio object
                    time.sleep(.005)
                    del audio

                # Check if the data was loaded
                db_response = self.check_if_dataset_is_in_database()
                if db_response.success:
                    # Update the metadata
                    self.dataset.metadata = metadata

                return db_response

            except Exception as e:
                return BasicResponse(success=False,
                                     status_code=500,
                                     message=f"Error loading the data to the database: {e}")

    def check_if_dataset_is_in_database(self) -> BasicResponse:
        """
        Check if the dataset is already in the database.

        :return: BasicResponse object.
        :rtype: BasicResponse
        """
        try:
            # Check if the dataset is in the database
            docs_on_database = self.db.collection.count_documents({})
            if len(self.dataset.metadata) <= docs_on_database:
                return BasicResponse(success=True, status_code=200, message="Dataset is in the database")
            else:
                return BasicResponse(success=False, status_code=404, message="Dataset is not in the database")
        except Exception as e:
            return BasicResponse(success=False, status_code=500,
                                 message=f"Error checking if the dataset is in the database: {e}")

    def train_loop(self, train_config: dict) -> BasicResponse:
        # GET THE SUBSETS
        results_path = train_config.get("experiment_folder", os.path.join(self.exp_config.root_path, "results"))
        if not os.path.exists(results_path):
            os.makedirs(results_path)

        # CONFIGURE THE MODEL AND TRAIN
        model_params: list = train_config.get("models", [])
        if not model_params:
            return BasicResponse(success=False, status_code=404, message="No model parameters found")

        feats_params: list = train_config.get("feats", [])
        if not feats_params:
            return BasicResponse(success=False, status_code=404, message="No features parameters found")

        for feat, model in zip(feats_params, model_params):
            # PROCESS THE DATA
            try:
                self.logger.info("Processing data")
                path_suffix = (f"_{train_config.get('experiment_name', 'default_exp')}"
                               f"_{model.get('model_name', 'default_model')}"
                               f"_{feat.get('feat_name', 'default_feat')}")
                x_train_path = os.path.join(results_path, f"x_train_{path_suffix}.pkl")
                x_dev_path = os.path.join(results_path, f"x_dev_{path_suffix}.pkl")
                y_train_path = os.path.join(results_path, f"y_train_{path_suffix}.pkl")
                y_dev_path = os.path.join(results_path, f"y_dev_{path_suffix}.pkl")

                if (os.path.exists(x_train_path)
                        and os.path.exists(x_dev_path)
                        and os.path.exists(y_train_path)
                        and os.path.exists(y_dev_path)):

                    self.logger.info("Loading data from file")
                    # Load the data
                    x_train = load_file(x_train_path)
                    x_dev = load_file(x_dev_path)
                    y_train = load_file(y_train_path)
                    y_dev = load_file(y_dev_path)
                else:
                    self.logger.info("Processing data")
                    x_train, y_train, x_dev, y_dev = self.dataset.get_subsets()
                    # Save the data
                    save_file(x_train_path, x_train)
                    save_file(x_dev_path, x_dev)
                    save_file(y_train_path, y_train)
                    save_file(y_dev_path, y_dev)
                self.logger.info("Data ready for training!")
            except Exception as e:
                return BasicResponse(success=False, status_code=500, message=f"Error processing the data: {e}")

            # TRAIN THE MODEL
            try:
                self.logger.info(f"Training model: {model.get('model_name')}")
                model["random_state"] = train_config.get("random_state", 42)

                factory = ModelFactory(**model)
                ft_response_with_model = factory.create_model_instance()
                ft_response_with_pipeline = factory.create_pipeline(ft_response_with_model.data.get("model"))
                if not ft_response_with_pipeline.success:
                    return ft_response_with_pipeline

                ft_response_with_pipeline_trained = factory.fit_pipeline(ft_response_with_pipeline.data.get("pipeline"),
                                                                         x_train,
                                                                         y_train)
                pipeline_trained = ft_response_with_pipeline_trained.data.get("pipeline")
                if not ft_response_with_pipeline_trained.success:
                    return ft_response_with_pipeline_trained

                ft_response_with_scores = factory.validation_frame_lv(pipeline_trained,
                                                                      x_dev,
                                                                      y_dev)
                if not ft_response_with_scores.success:
                    return ft_response_with_scores

                scores = ft_response_with_scores.data.get("scores")
                scores_message = "\n".join([f"\t\t\t\t\t\t\t\t\t\t\t\t\t -{k}: {v}-" for k, v in scores.items()])
                self.logger.info(f"Model trained: {model.get('model_name')}")
                self.logger.info(f"*** Scores ***\n{scores_message}")

                # Record the experiment
                if self.mlflow_server.is_up().success:
                    self.logger.info("Recording experiment")
                    # Add model parameters to the ft_response_with_scores
                    ft_response_with_scores.data["model_params"] = model
                    record_response = self.mlflow_server.record_experiment(train_config, ft_response_with_scores)
                    if record_response.success:
                        self.logger.info(record_response.get_as_str(module="MainModule", action="Record experiment"))
                    else:
                        self.logger.error(record_response.get_as_str(module="MainModule", action="Record experiment"))

                return DataResponse(success=True,
                                    status_code=200,
                                    message="Model Trained",
                                    data=dict(scores=scores, model=pipeline_trained))

            except Exception as e:
                return BasicResponse(success=False, status_code=500, message=f"Error training the model: {e}")


class MLFlowServer:
    """
    Class to manage the MLFlow server.
    """

    def __init__(self, mlflow_config=None):
        """
        Initialize the MlFlow server with a specified configuration.

        :param mlflow_config: Path to the configuration file.
        :type mlflow_config: str
        """

        if mlflow_config is None:
            mlflow_config = {}

        # Set the MLFlow server
        self.mlflow_url = mlflow_config.get("mlflow_url", "http://localhost:5000")
        # Set the MLFlow client
        self.mlflow_client = MlflowClient(tracking_uri=self.mlflow_url)

    def is_up(self) -> BasicResponse:
        """
        Check if there is any MLFlow server running.
        """
        experiments = self.mlflow_client.search_experiments()
        if experiments:
            return BasicResponse(success=True, status_code=200, message=f"Actives experiments: {len(experiments)}")
        else:
            return BasicResponse(success=False, status_code=404, message=f"No active server at: {self.mlflow_url}")

    def list_experiments(self) -> BasicResponse:
        """
        List all the experiments in the MLFlow server.

        :return: BasicResponse object.
        """
        try:
            all_experiments = self.mlflow_client.search_experiments()
            return BasicResponse(success=True, status_code=200, message=all_experiments)
        except Exception as e:
            return BasicResponse(success=False, status_code=500, message=f"Error listing experiments: {e}")

    def record_experiment(self, training_config: dict, training_response: BasicResponse) -> BasicResponse:
        try:
            # Define the experiment
            mlflow.set_tracking_uri(self.mlflow_url)
            exp_name = training_config.get("experiment_name", "exp")
            mlflow.set_experiment(exp_name)

            model_name = training_response.data.get("model_params").get("model_name")
            with mlflow.start_run(run_name=f'{model_name}_MFCC_{time.strftime("%Y-%m-%d-%H-%M-%S")}'):
                # Log the parameters
                mlflow.log_params(training_response.data.get("model_params"))
                mlflow.log_param('random_state', training_config.get("random_state", "UNKNOWN"))

                # Log the metrics
                mlflow_metric = self.post_process_metrics(training_response.data.get("scores"))
                mlflow.log_metrics(mlflow_metric)

            return BasicResponse(success=True, status_code=200, message="Experiment recorded")
        except Exception as e:
            return BasicResponse(success=False, status_code=500, message=f"Error recording the experiment: {e}")

    @staticmethod
    def post_process_metrics(all_scores: dict) -> dict:
        """
        Post-process the scores to make them compatible with MLFlow.

        :param all_scores: Scores to post-process.
        :type all_scores: dict
        :return: Post-processed scores.
        :rtype: dict
        """

        mlflow_metric = {}
        for k, v in all_scores.items():
            if isinstance(v, (int, float)):
                mlflow_metric[k] = v
            elif isinstance(v, np.ndarray):
                mlflow_metric[f"{k}_mean"] = v.mean()
                mlflow_metric[f"{k}_std"] = v.std()
                mlflow_metric[f"{k}_min"] = v.min()
                mlflow_metric[f"{k}_max"] = v.max()
                mlflow_metric[f"{k}_median"] = np.median(v)
                for i, val in enumerate(v):
                    mlflow_metric[f"{k}_fold_{i}"] = val

        return mlflow_metric


class DataSet(BaseModel):
    """
    Class to store the information of a dataset.

    :param data_name: Name of the dataset.
    :type data_name: str
    :param data_version: Version of the dataset.
    :type data_version: str
    :param data_path: Path to the dataset.
    :type data_path: str
    :param csv_extra: Path to the extra information of the dataset.
    :type csv_extra: str
    :param csv_metadata: Path to the metadata of the dataset.
    :type csv_metadata: str
    :param data_options: Options for the dataset.
    :type data_options: dict
    :param metadata: Metadata of the dataset.
    :type metadata: pd.DataFrame
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    data_name: Annotated[str, Field(pattern=r"^[a-zA-Z0-9-_]+$")]
    data_version: Annotated[str, Field(pattern=r"^[a-zA-Z0-9.]+$")]
    data_extension: Annotated[str, Field(pattern=r"^[a-zA-Z0-9._-]+$")]

    data_path: Annotated[str, Field(pattern=r"^[a-zA-Z0-9-_./~]+$")]
    csv_extra: Annotated[str, Field(pattern=r"^[a-zA-Z0-9-_./~]+$")]
    csv_metadata: Annotated[str, Field(pattern=r"^[a-zA-Z0-9-_./~]+$")]

    data_options: Optional[dict] = None
    data_description: Optional[dict] = None
    metadata: Optional[pd.DataFrame] = None

    coding_schema: Optional[dict] = None

    def load_dataset(self, standardization_method: Any = None) -> BasicResponse:
        """
        Load the dataset from the metadata.

        :return: BasicResponse object.
        :rtype: BasicResponse
        """
        try:
            # Check if the required path exists
            if not os.path.exists(self.data_path):
                return BasicResponse(success=False,
                                     status_code=404,
                                     message=f"Path not found: {self.data_path}")
            elif not os.path.exists(self.csv_metadata):
                return BasicResponse(success=False,
                                     status_code=404,
                                     message=f"Metadata file not found: {self.csv_metadata}")

            # Load the dataset
            self.metadata = pd.read_csv(self.csv_metadata)
            self.metadata = self.metadata.dropna()
            self.metadata = self.metadata.reset_index(drop=True)

            # Sampling the dataset
            if self.data_options.get("fraction", False):
                # Sample a fraction of the dataset
                fraction = self.data_options.get("fraction", 0.001)
                sample_response = self.sample_fraction(fraction)
                if not sample_response.success:
                    return BasicResponse(success=False,
                                         status_code=500,
                                         message="Error sampling the dataset")

            # Standardize the dataset
            if self.data_options.get("standardize", False):
                # Standardize the dataset
                column_id = self.data_description.get("Id")
                if "Id" not in self.metadata.columns:
                    # Add an id column
                    self.metadata["Id"] = self.metadata[column_id]

                column_path = self.data_description.get("Path")
                if "Path" not in self.metadata.columns:
                    if os.path.exists(os.path.join(self.data_path,
                                                   self.metadata[column_id].values[0])):
                        # Lambda function to create the path
                        self.metadata[column_path] = self.metadata[column_id].apply(
                            lambda x: os.path.join(self.data_path, x))
                    else:
                        # Lambda function to create the path
                        self.metadata[column_path] = self.metadata[column_id].apply(
                            lambda x: os.path.join(self.data_path, x + self.data_extension))

                # Standardize the dataset
                self.metadata, self.coding_schema = standardization_method(self.metadata)

                return BasicResponse(success=True,
                                     status_code=200,
                                     message="Dataset standardization completed")
            else:
                return BasicResponse(success=True, status_code=200, message="Loaded dataset from metadata")

        except Exception as e:
            return BasicResponse(success=False, status_code=500, message=f"Error loading the dataset: {e}")

    def get_subsets(self, frame_level: bool = True) -> tuple:
        if frame_level:
            x_train, y_train = self.subset_frame_level_to_train()
            x_dev, y_dev = self.subset_frame_level_to_dev()
        else:
            x_train, y_train, x_dev, y_dev = [], [], [], []

        return x_train, y_train, x_dev, y_dev

    def subset_frame_level_to_train(self) -> tuple[ndarray, ndarray]:
        """
        Make a subset of the DataFrame with the specified subset.

        :return: Subset of the DataFrame as arrays.
        :rtype: tuple[ndarray, ndarray]
        """
        # Create a copy of the DataFrame
        df_ = self.metadata.copy()
        df_subset = df_[df_["DataSet"] == "train"]
        # Iterate over the rows
        x_subset = np.array([])
        y_subset = np.array([])
        for index, row in tqdm(df_subset.iterrows(), total=len(df_subset), desc="Processing data"):
            # Load the audio
            path_to_audio = row["FilePath"]
            s, sr = read_audio(path_to_audio)
            mfcc = librosa.feature.mfcc(y=s, sr=sr)
            mfcc = mfcc.T

            # Create an array size of the mfcc
            label = np.array([row["Label"] for _ in range(mfcc.shape[0])])

            # Add the mfcc and the label to the X and y arrays
            if x_subset.size == 0:
                x_subset = mfcc
                y_subset = label
            else:
                x_subset = np.vstack((x_subset, mfcc))
                y_subset = np.vstack((y_subset, label))

        return x_subset, y_subset

    def subset_frame_level_to_dev(self) -> tuple[list, list]:
        """
        Make a subset of the DataFrame with the specified subset and return the data as lists.

        :return: Subset of the DataFrame as lists.
        :rtype: tuple[list, list]
        """
        # Create a copy of the DataFrame
        df_ = self.metadata.copy()
        df_subset = df_[df_["DataSet"] == "dev"]
        # Iterate over the rows
        x_subset = []
        y_subset = []
        for index, row in tqdm(df_subset.iterrows(), total=len(df_subset), desc="Processing data"):
            # Load the audio
            path_to_audio = row["FilePath"]
            s, sr = read_audio(path_to_audio)
            mfcc = librosa.feature.mfcc(y=s, sr=sr)
            mfcc = mfcc.T

            # Get the label
            label = row["Label"]

            # Append the mfcc and the label to the X and y arrays
            x_subset.append(mfcc)
            y_subset.append(label)

        return x_subset, y_subset

    def sample_fraction(self, fraction: float = 0.001) -> BasicResponse:
        """
        Sample a fraction of the dataset.

        :param fraction: Fraction to sample.
        :type fraction: float
        :return: BasicResponse object.
        :rtype: BasicResponse
        """
        try:
            self.metadata = self.metadata.sample(frac=fraction)
            return BasicResponse(success=True, status_code=200, message="Dataset sampled")
        except Exception as e:
            return BasicResponse(success=False, status_code=500, message=f"Error sampling the dataset: {e}")


class ModelFactory:
    """
    Class to create a model instance and a pipeline with the specified preprocessors.
    """

    def __init__(self,
                 model_name: str,
                 model_params,
                 k_folds: int = 5,
                 random_state: int = 42,
                 preprocessors: list = None,
                 scoring: dict = None):
        """
        Initialize the class with the specified configuration.
        """
        self.model_name: str = model_name
        self.model_params: dict = model_params
        self.random_state: int = random_state
        self.cross_val_folds = KFold(n_splits=k_folds, random_state=self.random_state, shuffle=True)

        if preprocessors is None:
            self.preprocessors = [impute.SimpleImputer(), StandardScaler()]
        else:
            self.preprocessors = preprocessors

        if scoring is None:
            self.scoring = {'accuracy': 'accuracy',
                            'f1': make_scorer(f1_score, average='weighted', zero_division=1),
                            'recall': make_scorer(recall_score, average='weighted', zero_division=1),
                            'precision': make_scorer(precision_score, average='weighted', zero_division=1),
                            'sensitivity': make_scorer(recall_score, average='weighted', zero_division=1)}
        else:
            self.scoring = scoring

    def create_model_instance(self) -> BasicResponse:
        """
        Create a model instance with the specified parameters.

        :return: Model instance with the specified parameters.
        :rtype: Any
        """
        try:
            if self.model_name == "LogisticRegression":
                model_instance = LogisticRegression(**self.model_params)
            elif self.model_name == "RandomForestClassifier":
                model_instance = RandomForestClassifier(**self.model_params)
            else:
                model_instance = None

            if model_instance is None:
                return BasicResponse(success=False,
                                     status_code=404,
                                     message="Model selected not supported yet.")
            else:
                return DataResponse(success=True,
                                    status_code=200,
                                    message="Model created",
                                    data={"model": model_instance})
        except Exception as e:
            return BasicResponse(success=False,
                                 status_code=500,
                                 message=f"Error creating the model instance: {e}")

    def create_pipeline(self, model_instance: Any) -> BasicResponse:
        """
        Create a pipeline with the specified preprocessors and model instance.

        :param model_instance: Model instance to use.
        :type model_instance: Any
        :return: Pipeline with the specified preprocessors and model instance.
        :rtype: Pipeline
        """
        try:
            steps = [('preprocessor_{}'.format(idx), preprocessor)
                     for idx, preprocessor in enumerate(self.preprocessors)]
            steps.append(('model', model_instance))
            pipeline = Pipeline(steps)

            return DataResponse(success=True, status_code=200, message="Pipeline created", data={"pipeline": pipeline})
        except Exception as e:
            return BasicResponse(success=False, status_code=500, message=f"Error creating the pipeline: {e}")

    @staticmethod
    def validation_frame_lv(pipeline: Pipeline, X: ndarray, y_true: ndarray) -> BasicResponse:
        """
        Perform cross-validation with the specified pipeline and data.

        :param pipeline: Pipeline to use.
        :type pipeline: Pipeline
        :param X: Input data.
        :type X: ndarray
        :param y_true: Target data.
        :type y_true: ndarray
        :return: Scores of the cross-validation process for the specified pipeline.
        :rtype: BasicResponse
        """
        try:
            y_score = np.ndarray([])
            for x_dev in X:
                x_pred = pipeline.predict(x_dev)
                x_pred_voting = np.sum(x_pred, axis=0)

                # Create an array with all 0 but with 1 in the position of the max value
                y_pred = np.zeros(len(x_pred_voting))
                y_pred[np.argmax(x_pred_voting)] = 1
                if y_score.size == 1:
                    y_score = y_pred
                else:
                    y_score = np.vstack((y_score, y_pred))

            #  Convert the y_true list to a np.ndarray
            y_true = np.array(y_true)

            # Convert one-hot encoding to class labels
            y_true_labels = np.argmax(y_true, axis=1)
            y_score_labels = np.argmax(y_score, axis=1)

            # Calculate accuracy
            acc = accuracy_score(y_true_labels, y_score_labels)
            precision, recall, f_beta, support = precision_recall_fscore_support(y_true_labels, y_score_labels,
                                                                                 average='weighted')
            f1_scr = f1_score(y_true_labels, y_score_labels, average='weighted')

            # Create a dictionary of scores
            dict_scores = {'acc_score': float(acc),
                           'f1_scr': float(f1_scr),
                           'f_beta': float(f_beta),
                           'recall': float(recall),
                           'precision': float(precision),
                           }

            return DataResponse(success=True,
                                status_code=200,
                                message="Cross-validation completed",
                                data={"scores": dict_scores})
        except Exception as e:
            return BasicResponse(success=False, status_code=500, message=f"Error performing cross-validation: {e}")

    def perform_cross_validation(self, pipeline: Pipeline, X: ndarray, y: ndarray) -> BasicResponse:
        """
        Perform frame-level validation with the specified pipeline and data.

        :param pipeline: Pipeline to use.
        :type pipeline: Pipeline
        :param X: Input data.
        :type X: ndarray
        :param y: Target data.
        :type y: ndarray
        :return: Scores of the frame-level validation process for the specified pipeline.
        :rtype: BasicResponse
        """
        try:
            scores = cross_validate(pipeline, X, y, cv=self.cross_val_folds, scoring=self.scoring, n_jobs=-1, verbose=1)
            return DataResponse(success=True,
                                status_code=200,
                                message="Frame-level validation completed",
                                data={"scores": scores})
        except Exception as e:
            return BasicResponse(success=False, status_code=500,
                                 message=f"Error performing frame-level validation: {e}")

    @staticmethod
    def fit_pipeline(pipeline, X, y) -> BasicResponse:
        """
        Fit the pipeline with the specified data.

        :param pipeline: Pipeline to fit.
        :type pipeline: Pipeline
        :param X: Input data.
        :type X: ndarray
        :param y: Target data.
        :type y: ndarray
        :return: Fitted pipeline with the specified data.
        :rtype: BasicResponse
        """
        try:
            with parallel_backend('loky', n_jobs=-1):
                pipeline.fit(X, y)
            return DataResponse(success=True, status_code=200, message="Pipeline fitted", data={"pipeline": pipeline})
        except Exception as e:
            return BasicResponse(success=False, status_code=500, message=f"Error fitting the pipeline: {e}")


def standardization_method_odyssey(df_metadata: pd.DataFrame) -> [pd.DataFrame, dict]:
    # Define the emotions
    emotions = ["Angry", "Sad", "Happy", "Surprise", "Fear", "Disgust", "Contempt", "Neutral"]
    emotion_codes = ["A", "S", "H", "U", "F", "D", "C", "N"]

    # Create a dictionary for one-hot encoding
    one_hot_dict = {e: [1.0 if e == ec else 0.0 for ec in emotion_codes] for e in emotion_codes}

    # Filter out rows with undefined EmoClass
    df_metadata = df_metadata[df_metadata['EmoClass'].isin(emotion_codes)]

    # Apply one-hot encoding and create 'Label' column
    for i, e in enumerate(emotion_codes):
        # df_metadata[emotions[i]] = df_metadata['EmoClass'].apply(lambda x: one_hot_dict[x][i])
        df_metadata.loc[df_metadata['EmoClass'] == e, emotions] = one_hot_dict[e]

    # Select relevant columns for the new DataFrame
    df_final = df_metadata[['FileName', 'Split_Set', 'FilePath'] + emotions]
    df_final = df_final.rename(columns={'FileName': 'Id'})
    df_final = df_final.rename(columns={'Split_Set': 'DataSet'})

    # Remove the extension from the file name
    df_final['Id'] = df_final['Id'].apply(lambda x: x.split(".")[0])

    # Remplace the "Train" and "Development" strings with "train" and "dev"
    df_final['DataSet'] = df_final['DataSet'].apply(lambda x: "train" if x == "Train" else "dev")

    # Create 'Label' column with one-hot encoded vectors as lists
    df_final['Label'] = df_final[emotions].values.tolist()

    # Drop individual emotion columns
    df_final.drop(emotions, axis=1, inplace=True)

    return df_final[['Id', 'DataSet', 'Label', 'FilePath']], one_hot_dict


if __name__ == "__main__":
    # Load arguments
    parser = argparse.ArgumentParser()

    # Define arguments
    parser.add_argument('--config', '-c', type=str, default=False)
    # Parse arguments
    args = parser.parse_args()

    # Create the experiment
    experiment = BasicExperiment(config_path=args.config)

    # df_analyzer = DataFrameAnalyzer()
    # df_analyzer.generate_summary_report(df=labels_consensus_processed)
    # df_analyzer.compute_data_quality_measures(df=labels_consensus_processed)
    # df_analyzer.generate_correlation_report(df=labels_consensus_processed, exclude_columns=["FileName"])
