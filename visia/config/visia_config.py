from __future__ import annotations

import json
import os
from typing import Any


class BasicConfig:
    """
    Main class to store the configuration of the project.

    :param path_to_config: Path to the configuration file.
    :type path_to_config: str
    """

    def __init__(self, path_to_config: str):
        self.path_to_config: str = path_to_config

        self.db_config = None
        self.log_config = None
        self.mlflow_config = None
        self.data_config = None
        self.load_config()

    def load_config(self):
        """
        Set the attributes of the class with the configuration from the JSON file.
        """
        if not os.path.exists(self.path_to_config):
            print("Error loading configuration from file")
            print("Using default configuration")
        else:
            config_json = self.load_config_from_json(self.path_to_config)
            # Create attributes for each key-value pair in the JSON
            for key, value in config_json.items():
                setattr(self, key, value)

    def model_dump(self) -> dict:
        """
        Dump the model data as a dictionary.
        :return: A dict with the model data.
        """
        dump: dict = {}
        for attr in self.__dict__:
            dump[attr] = self.__getattribute__(attr)

        return dump

    @staticmethod
    def load_config_from_json(path: str) -> Any | None:
        """
        Load a json file as a dictionary. Useful to load the configuration of the experiments
        :param path: path to the json file
        :return: dictionary with the configuration
        """
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            else:
                return None
        except Exception or IOError as e:
            print(f"Error loading object: {e}")
            return None
