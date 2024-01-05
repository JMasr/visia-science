import pytest

from src.helper_module import helper_function


@pytest.fixture
def example_initialized_repo():
    """
    Basic example of Fixture Usage.
    In the context of testing, a fixture is a mechanism to set up and provide necessary preconditions for your tests
    :return: a dictionary an initial setup.
    """
    module_fixture = {"success": True}
    return module_fixture


def test_example_aaa_pattern():
    """
    Basic example of the AAA Patter on testing.
    The AAA pattern: Arrange, Act, Assert.
    Clearly separate the setup, the action being tested, and the assertion.
    :return:
    """
    # Arrange
    instance = "do an action"

    # Act
    result = instance + "get a result"

    # Assert
    assert result == "expected_result"


def test_helper_function():
    assert helper_function() == "Helper function result"
