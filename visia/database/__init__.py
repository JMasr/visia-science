from pymongo import MongoClient, errors
from pymongo.database import Database

from responses.basic_responses import BasicResponse


class BasicDataBase:
    """
    Class to manage the database.
    """

    def __init__(self, **kwargs):
        """
        Initialize the class with the specified configuration.

        :param kwargs: Configuration parameters.
        :type kwargs: dict
        """
        self.client: MongoClient = None
        self.database: Database = None
        self.collection = None

        self.db_name = kwargs.get("db_name", "demo-db")
        self.db_main_collection = kwargs.get("db_main_collection", None)
        self.db_user = kwargs.get("db_user", "demo-user")
        self.db_password = kwargs.get("db_password", "demo-password")

        self.db_type = kwargs.get("db_type", "")
        self.db_engine = kwargs.get("db_engine", "")
        self.db_host = kwargs.get("db_host", "localhost")
        self.db_port = kwargs.get("db_port", "")
        self.db_url = kwargs.get("db_url",
                                 f"{self.db_type}://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/")

        self.init_db()

    def init_db(self):
        """
        Initialize the database.
        """

        if self.db_type == "mongodb" and self.db_engine == "pymongo":
            self.client = MongoClient(self.db_url)
            self.database = self.client[self.db_name]
            self.collection = self.database[self.db_main_collection] if self.db_main_collection else None

    def check_connection(self) -> BasicResponse:
        """
        Check the connection to the database.

        :return: True if the connection is successful, False otherwise.
        :rtype: bool
        """

        if self.db_type == "mongodb" and self.db_engine == "pymongo":
            try:

                if self.client.server_info():
                    return BasicResponse(success=True,
                                         message="Connected to the database.",
                                         status_code=200)
                else:
                    return BasicResponse(success=False,
                                         message="Failed to connect to the database: Server info not available.",
                                         status_code=500)

            except errors.ServerSelectionTimeoutError or Exception as error:
                return BasicResponse(success=False,
                                     message=f"Failed to connect to the database: {error}",
                                     status_code=500)
