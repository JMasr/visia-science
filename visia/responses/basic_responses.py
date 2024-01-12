from pydantic import BaseModel


class BasicResponse(BaseModel):
    """
    This object is used to emulate a response of a request/event/operation on the repository.

    :param success: Boolean that indicates if the operation was successful or not.
    :type success: bool
    :param status_code: Status code of the operation.
    :type status_code: int
    :param message: Message of the operation.
    :type message: str

    """

    success: bool
    status_code: int
    message: str

    def get_as_str(self, module: str, action: str):
        """
        Return the BasicResponse object as a string.

        :param module: Module where the response/event happens.
        :type module: str
        :param action: Action where the response/event comes from.
        :type action: str
        :return: A readable string with all the information of the response/event.
        :rtype: str
        """
        return f"{module} - {action} - Success: {self.success} - Message: {self.message} - Status Code: {self.status_code}"


class ListResponse(BasicResponse):
    """
    This object is an extension of the BasicResponse object.
    It is used to emulate a response of a request/event/operation on the repository that returns a list of objects.

    :param data: List of objects.
    :type data: list

    """

    data: list


class DataResponse(BasicResponse):
    """
    This object is an extension of the BasicResponse object.
    It is used to emulate a response of a request/event/operation on the repository that returns data as a dictionary.

    :param data: Dictionary with the data.
    """

    data: dict
