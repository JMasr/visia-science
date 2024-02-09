"""
Description
===========

This is a module for running the Visia-Science main Pipeline.
"""

import argparse
import os
from typing import Annotated, Optional

import pandas as pd
from pydantic import BaseModel, Field, ConfigDict
from tqdm import tqdm

from visia.config.visia_config import BasicConfig
from visia.database import BasicDataBase
from visia.dataset.dataset import AudioStream
from visia.logger import BasicLogger
from visia.responses.basic_responses import BasicResponse


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

        # Set the dataset
        self.dataset = DataSet(**self.exp_config.data_config)
        dset_response = self.dataset.load_dataset()
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

    def data2database(self):
        """
        Load the data from the dataset to the database.
        """

        if self.check_if_dataset_is_in_database():
            self.logger.info("Dataset is already in the database")
        else:
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
                    del audio

                # Check if the data was loaded
                response = self.check_if_dataset_is_in_database()
                if response.success:
                    # Update the metadata
                    self.dataset.metadata = metadata
                    self.logger.info(response.get_as_str(module="MainModule", action="Load data to database"))
                else:
                    self.logger.error(response.get_as_str(module="MainModule", action="Load data to database"))

            except Exception as e:
                self.logger.error(f"Error loading the {self.dataset.data_name} to {self.db.db_url}: {e}")

    def check_if_dataset_is_in_database(self) -> BasicResponse:
        """
        Check if the dataset is already in the database.

        :return: BasicResponse object.
        :rtype: BasicResponse
        """
        try:
            # Check if the dataset is in the database
            if self.db.collection.count_documents({}) == len(self.dataset.metadata):
                return BasicResponse(success=True, status_code=200, message="Dataset is in the database")
            else:
                return BasicResponse(success=False, status_code=404, message="Dataset is not in the database")
        except Exception as e:
            return BasicResponse(success=False, status_code=500,
                                 message=f"Error checking if the dataset is in the database: {e}")


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

    data_path: Annotated[str, Field(pattern=r"^[a-zA-Z0-9-_./]+$")]
    csv_extra: Annotated[str, Field(pattern=r"^[a-zA-Z0-9-_./]+$")]
    csv_metadata: Annotated[str, Field(pattern=r"^[a-zA-Z0-9-_./]+$")]

    data_options: Optional[dict] = None
    metadata: Optional[pd.DataFrame] = None

    def load_dataset(self) -> BasicResponse:
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
            if self.data_options.get("standardize", False):
                # TODO: Add the code to load the dataset without standardization
                return BasicResponse(success=False,
                                     status_code=500,
                                     message="Dataset without standardization are not supported yet.")
            else:
                # Load the dataset
                self.metadata = pd.read_csv(self.csv_metadata)
                self.metadata = self.metadata.dropna()
                self.metadata = self.metadata.reset_index(drop=True)
                return BasicResponse(success=True, status_code=200, message="Loaded dataset from metadata")

        except Exception as e:
            return BasicResponse(success=False, status_code=500, message=f"Error loading the dataset: {e}")

    # def load_data(self, path_to_data: str,
    #               files_format: str = "wav",
    #               data_name: str = "metadata",
    #               data_type: DataTypes = DataTypes.AUDIO) -> BasicResponse:
    #     """
    #     Load the audios from the dataset.
    #
    #     :param path_to_data: Path to the audios.
    #     :type path_to_data: str
    #     :param files_format: Audio format.
    #     :type files_format: str
    #     :param data_name: Name of the data.
    #     :type data_name: str
    #     :param data_type: Data type of the files (txt, json, audio, video, other).
    #     :type data_type: int
    #     :return: BasicResponse object.
    #     :rtype: BasicResponse
    #     """
    #     path = os.path.join(self.root_path, path_to_data)
    #     if os.path.isfile(path):
    #         try:
    #             # Load the data
    #             data = []
    #             if data_type == DataTypes.AUDIO:
    #                 data = [AudioStream.from_file_path(file_path=path)]
    #             elif data_type == DataTypes.VIDEO:
    #                 data = [VideoStream.from_file_path(file_path=path)]
    #             elif data_type == DataTypes.TEXT:
    #                 data = [TextStream.from_file_path(file_path=path)]
    #
    #             # Check if the data was loaded
    #             if not data:
    #                 return BasicResponse(success=False, status_code=500, message=f"Error loading the audios")
    #             else:
    #                 setattr(self, data_name, data)
    #
    #             return BasicResponse(success=True, status_code=200, message=f"Loaded {len(data)} {data_name}")
    #         except Exception as e:
    #             return BasicResponse(success=False, status_code=500, message=f"Error loading the {data_name}: {e}")
    #
    #     elif os.path.isdir(path):
    #         files_response = self.get_files(path_to_files=path,
    #                                         files_format=files_format)
    #         if not files_response.success:
    #             return files_response
    #         else:
    #             data = []
    #             try:
    #                 # Load the data
    #                 if data_type == DataTypes.AUDIO:
    #                     data = [AudioStream.from_file_path(file_path=file) for file in files_response.data]
    #                 elif data_type == DataTypes.VIDEO:
    #                     data = [VideoStream.from_file_path(file_path=file) for file in files_response.data]
    #                 elif data_type == DataTypes.TEXT:
    #                     data = [TextStream.from_file_path(file_path=file) for file in files_response.data]
    #
    #                 # Check if the data was loaded
    #                 if not data:
    #                     return BasicResponse(success=False, status_code=500, message=f"Error loading the audios")
    #                 else:
    #                     setattr(self, data_name, data)
    #
    #                 return BasicResponse(success=True, status_code=200, message=f"Loaded {len(data)} {data_name}")
    #             except Exception as e:
    #                 error_message = f"Error loading the {data_name}-{files_response.data[len(data) + 1]}: {e}"
    #                 return BasicResponse(success=False, status_code=500, message=error_message)
    #
    # @staticmethod
    # def get_files(path_to_files: str, files_format: str = "txt") -> BasicResponse:
    #     """
    #     Get the list of files from a path.
    #
    #     :param path_to_files: Path to the files.
    #     :type path_to_files: str
    #     :param files_format: Files format.
    #     :type files_format: str
    #     :return: List of files.
    #     :rtype: list
    #     """
    #     # Check if the path exists
    #     if not os.path.exists(path_to_files):
    #         return BasicResponse(success=False, status_code=404, message=f"Path not found: {path_to_files}")
    #
    #     # Get the list of files
    #     files = os.listdir(path_to_files)
    #     files = [os.path.join(path_to_files, file) for file in files if file.endswith(files_format)]
    #
    #     # Check if there are files
    #     if not files:
    #         return BasicResponse(success=False, status_code=404, message=f"No files found in: {path_to_files}")
    #
    #     return ListResponse(success=True, status_code=200, message=f"Loaded {len(files)} files", data=files)


if __name__ == "__main__":
    # Load arguments
    parser = argparse.ArgumentParser()

    # Define arguments
    parser.add_argument('--config', '-c', type=str, default=False)
    # Parse arguments
    args = parser.parse_args()

    # Create the experiment
    experiment = BasicExperiment(config_path=args.config)
    experiment.data2database()

    # df_analyzer = DataFrameAnalyzer()
    # df_analyzer.generate_summary_report(df=labels_consensus_processed)
    # df_analyzer.compute_data_quality_measures(df=labels_consensus_processed)
    # df_analyzer.generate_correlation_report(df=labels_consensus_processed, exclude_columns=["FileName"])
