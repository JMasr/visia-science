"""
Description
===========

This is a module for running the Visia-Science main Pipeline.
"""

import argparse
import os

from visia.config.visia_config import BasicConfig
from visia.database import BasicDataBase
from visia.dataset.dataset import TextStream, AudioStream, DataTypes, VideoStream
from visia.logger import BasicLogger
from visia.responses.basic_responses import BasicResponse, ListResponse


class Car:
    """
    Car class.

    :param speed: Speed of the car.
    :type speed: int
    :return: Car object.
    :rtype: Car
    """

    def __init__(self, speed=0):
        self.speed = speed
        self.odometer = 0
        self.time = 0

    def accelerate(self):
        """
        Accelerate the car.

        :return: None
        """
        self.speed += 5

    def brake(self):
        """
        Brake the car.

        :return: None
        """
        self.speed -= 5

    def step(self):
        """
        Update the car's properties.

        :return: None
        """
        self.odometer += self.speed
        self.time += 1

    def average_speed(self) -> float:
        """
        Calculate the average speed.

        :return: Average speed.
        :rtype: float
        """
        return self.odometer / self.time

    def speed_validate(self) -> bool:
        """
        Validate the speed of the car.

        :return: True if the speed is valid, False otherwise.
        :rtype: bool
        """
        return self.speed <= 160


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

        # Set the database
        self.db = BasicDataBase(**self.exp_config.db_config)



class DataSet:
    """

    """

    def __init__(self, version: str = "v.0.0", root_path: str = None):
        """
        Initialize the class with the specified configuration.

        :param version: Version of the dataset.
        :type version: str
        :param root_path: Root path of the dataset.
        :type root_path: str
        """
        self.version = version
        self.root_path = root_path if root_path else os.path.join(os.getcwd())

    def load_data(self, path_to_data: str,
                  files_format: str = "wav",
                  data_name: str = "metadata",
                  data_type: DataTypes = DataTypes.AUDIO) -> BasicResponse:
        """
        Load the audios from the dataset.

        :param path_to_data: Path to the audios.
        :type path_to_data: str
        :param files_format: Audio format.
        :type files_format: str
        :param data_name: Name of the data.
        :type data_name: str
        :param data_type: Data type of the files (txt, json, audio, video, other).
        :type data_type: int
        :return: BasicResponse object.
        :rtype: BasicResponse
        """
        path = os.path.join(self.root_path, path_to_data)
        if os.path.isfile(path):
            try:
                # Load the data
                data = []
                if data_type == DataTypes.AUDIO:
                    data = [AudioStream.from_file_path(file_path=path)]
                elif data_type == DataTypes.VIDEO:
                    data = [VideoStream.from_file_path(file_path=path)]
                elif data_type == DataTypes.TEXT:
                    data = [TextStream.from_file_path(file_path=path)]

                # Check if the data was loaded
                if not data:
                    return BasicResponse(success=False, status_code=500, message=f"Error loading the audios")
                else:
                    setattr(self, data_name, data)

                return BasicResponse(success=True, status_code=200, message=f"Loaded {len(data)} {data_name}")
            except Exception as e:
                return BasicResponse(success=False, status_code=500, message=f"Error loading the {data_name}: {e}")

        elif os.path.isdir(path):
            files_response = self.get_files(path_to_files=path,
                                            files_format=files_format)
            if not files_response.success:
                return files_response
            else:
                data = []
                try:
                    # Load the data
                    if data_type == DataTypes.AUDIO:
                        data = [AudioStream.from_file_path(file_path=file) for file in files_response.data]
                    elif data_type == DataTypes.VIDEO:
                        data = [VideoStream.from_file_path(file_path=file) for file in files_response.data]
                    elif data_type == DataTypes.TEXT:
                        data = [TextStream.from_file_path(file_path=file) for file in files_response.data]

                    # Check if the data was loaded
                    if not data:
                        return BasicResponse(success=False, status_code=500, message=f"Error loading the audios")
                    else:
                        setattr(self, data_name, data)

                    return BasicResponse(success=True, status_code=200, message=f"Loaded {len(data)} {data_name}")
                except Exception as e:
                    error_message = f"Error loading the {data_name}-{files_response.data[len(data) + 1]}: {e}"
                    return BasicResponse(success=False, status_code=500, message=error_message)


    @staticmethod
    def get_files(path_to_files: str, files_format: str = "txt") -> BasicResponse:
        """
        Get the list of files from a path.

        :param path_to_files: Path to the files.
        :type path_to_files: str
        :param files_format: Files format.
        :type files_format: str
        :return: List of files.
        :rtype: list
        """
        # Check if the path exists
        if not os.path.exists(path_to_files):
            return BasicResponse(success=False, status_code=404, message=f"Path not found: {path_to_files}")

        # Get the list of files
        files = os.listdir(path_to_files)
        files = [os.path.join(path_to_files, file) for file in files if file.endswith(files_format)]

        # Check if there are files
        if not files:
            return BasicResponse(success=False, status_code=404, message=f"No files found in: {path_to_files}")

        return ListResponse(success=True, status_code=200, message=f"Loaded {len(files)} files", data=files)


if __name__ == "__main__":
    # Load arguments
    parser = argparse.ArgumentParser()

    # Define arguments
    parser.add_argument('--config', '-c', type=str, default=False)
    # Parse arguments
    args = parser.parse_args()

    # Create the experiment
    experiment = BasicExperiment(config_path=args.config)

    audio = AudioStream.from_file_path(file_path=r"C:\Users\jmram\OneDrive\Documents\GitHub\visia-science\data\MSP-PODCAST_0002_0033.wav")
    experiment.logger.info(f"Audio: {audio}")

    # Create the dataset
    odyssey = DataSet(version="v.1.0", root_path=os.path.join("D:\\", "Odyssey"))

    # Load the audios
    response_odyssey = odyssey.load_data(path_to_data=r"Audios\Audios",
                                         files_format="wav",
                                         data_name="audios",
                                         data_type=DataTypes.AUDIO)
    experiment.logger.info(response_odyssey.get_as_str(module="MainModule", action="Load audios"))
    # Load the transcriptions
    response_odyssey = odyssey.load_data(path_to_data=r"Transcripts\Transcripts",
                                         files_format="txt",
                                         data_name="transcriptions",
                                         data_type=DataTypes.TEXT)
    experiment.logger.info(response_odyssey.get_as_str(module="MainModule", action="Load transcriptions"))

    # Load the labels
    response_odyssey = odyssey.load_data(path_to_data=r"Labels\Labels\labels_consensus.csv",
                                         files_format="txt",
                                         data_name="labels_consensus",
                                         data_type=DataTypes.TEXT)
    experiment.logger.info(response_odyssey.get_as_str(module="MainModule", action="Load labels consensus"))

    response_odyssey = odyssey.load_data(path_to_data=r"Labels\Labels\labels_detailed.csv",
                                         files_format="txt",
                                         data_name="labels_detailed",
                                         data_type=DataTypes.TEXT)
    experiment.logger.info(response_odyssey.get_as_str(module="MainModule", action="Load labels detailed"))
