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
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from pydantic import BaseModel, Field, ConfigDict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score, recall_score, precision_score
from sklearn.model_selection import KFold, cross_validate
from tqdm import tqdm

from visia.config.visia_config import BasicConfig
from visia.database import BasicDataBase
from visia.dataset.dataset import AudioStream
from visia.files import read_audio
from visia.logger import BasicLogger
from visia.responses.basic_responses import BasicResponse, DataResponse


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

        # Set the logger
        self.logger = BasicLogger(**self.exp_config.log_config).get_logger()
        self.logger.info(f"Logger created - 200 - Parameters: {self.exp_config.model_dump()}\n")

        # Set the MLFlow server
        self.mlflow_server = MLFlowServer(mlflow_config=self.exp_config.mlflow_config)
        mlflow_response = self.mlflow_server.is_up()
        if mlflow_response.success:
            self.logger.info(mlflow_response.get_as_str(module="MainModule", action="Check MLFlow server"))
        else:
            self.logger.error(mlflow_response.get_as_str(module="MainModule", action="Check MLFlow server"))

        # Set the dataset
        self.dataset = DataSet(**self.exp_config.data_config)
        dset_response = self.dataset.load_dataset(standardization_method_odyssey)
        if dset_response.success:
            self.logger.info(dset_response.get_as_str(module="MainModule", action="Load dataset"))
        else:
            self.logger.error(dset_response.get_as_str(module="MainModule", action="Load dataset"))

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

        # Prepare the cross-validation procedure
        cv = KFold(n_splits=2, random_state=42, shuffle=True)
        # Create model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        # Set the scoring
        scoring = {'accuracy': 'accuracy',
                   'f1': make_scorer(f1_score, average='weighted', zero_division=1),
                   'recall': make_scorer(recall_score, average='weighted', zero_division=1),
                   'precision': make_scorer(precision_score, average='weighted', zero_division=1),
                   'sensitivity': make_scorer(recall_score, average='weighted', zero_division=1)}
        # Train the model
        df_train = self.dataset.metadata.copy()
        self.logger.info("Training model...")
        train_response = self.train_loop(cv=cv,
                                         model=model,
                                         scoring=scoring,
                                         df_to_train=df_train)
        if train_response.success:
            self.logger.info(train_response.get_as_str(module="MainModule", action="Train model"))
            # report performance
            scores = train_response.data.get("scoring")
            print('Accuracy: %.3f (%.3f)' % (scores['test_accuracy'].mean(), scores['test_accuracy'].std()))
            print('F1-score: %.3f (%.3f)' % (scores['test_f1'].mean(), scores['test_f1'].std()))
            print('Recall: %.3f (%.3f)' % (scores['test_recall'].mean(), scores['test_recall'].std()))
            print('Precision: %.3f (%.3f)' % (scores['test_precision'].mean(), scores['test_precision'].std()))
            print('Sensitivity: %.3f (%.3f)' % (scores['test_sensitivity'].mean(), scores['test_sensitivity'].std()))

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

    @staticmethod
    def train_loop(df_to_train: pd.DataFrame, model: Any, cv: Any, scoring: dict) -> BasicResponse:
        """
        Train the model with the dataset.
        """
        try:
            # Copy the dataframe
            df_train = df_to_train.copy()

            x = np.array([])
            y = np.array([])
            # Iterate over the rows
            for i, r in tqdm(df_train.iterrows(), total=len(df_train), desc="Processing data"):
                # Load the audio
                path_to_audio = r["FilePath"]
                s, sr = read_audio(path_to_audio)
                mfcc = librosa.feature.mfcc(y=s, sr=sr)
                mfcc = mfcc.T

                # Create an array size of the mfcc
                label = np.array([r["Label"] for _ in range(mfcc.shape[0])])

                # Add the mfcc and the label to the X and y arrays
                if x.size == 0:
                    x = mfcc
                    y = label
                else:
                    x = np.vstack((x, mfcc))
                    y = np.vstack((y, label))

            # Evaluate model
            scores = cross_validate(model, x, y, scoring=scoring, cv=cv, n_jobs=-1, verbose=1)

            return DataResponse(success=True,
                                status_code=200,
                                message="Model trained successfully",
                                data={"model": model,
                                      "scoring": scores})
        except Exception as e:
            return BasicResponse(success=False,
                                 status_code=500,
                                 message=f"Error training the model: {e}")


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
        self.mlflow_experiment_name = mlflow_config.get("mlflow_experiment_name", f"Experiment_{time.time()}")
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

    data_path: Annotated[str, Field(pattern=r"^[a-zA-Z0-9-_./]+$")]
    csv_extra: Annotated[str, Field(pattern=r"^[a-zA-Z0-9-_./]+$")]
    csv_metadata: Annotated[str, Field(pattern=r"^[a-zA-Z0-9-_./]+$")]

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
            if self.data_options.get("standardize", False):
                # Standardize the dataset
                column_id = self.data_description.get("Id")
                if "Id" not in self.metadata.columns:
                    # Add an id column
                    self.metadata["Id"] = self.metadata[column_id]

                column_path = self.data_description.get("Path")
                if "Path" not in self.metadata.columns:
                    if os.path.exists(os.path.join(self.data_path,
                                                   self.metadata[column_id][0])):
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
