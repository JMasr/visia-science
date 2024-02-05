from pymongo import MongoClient


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
        self.db = None
        self.client = None

        self.db_name = kwargs.get("db_name", "demo-db")
        self.db_user = kwargs.get("db_user", "demo-user")
        self.db_password = kwargs.get("db_password", "demo-password")

        self.db_type = kwargs.get("db_type", "")
        self.db_engine = kwargs.get("db_engine", "")
        self.db_host = kwargs.get("db_host", "localhost")
        self.db_port = kwargs.get("db_port", "")
        self.db_url = kwargs.get("db_url", f"{self.db_type}://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}")

        self.init_db()

    def init_db(self):
        """
        Initialize the database.
        """

        if self.db_type == "mongodb" and self.db_engine == "pymongo":
            self.client = MongoClient(self.db_url)
            self.db = self.client[self.db_name]
