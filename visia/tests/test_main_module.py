"""
Description
===========

Unit tests for the visia main module.
"""

import pytest
from visia.main_module import Car


@pytest.fixture
def my_car():
    """
    Fixture for the Car class.

    :return: a Car object.
    :rtype: Car
    """
    return Car(50)


speed_data = {45, 50, 55, 100}


@pytest.mark.parametrize("speed_brake", speed_data)
def test_car_brake(speed_brake: int):
    """
    Test the brake method of the Car class.

    :param speed_brake: Speed to use for the test.
    :type speed_brake: int
    :return: None
    """
    car = Car(50)
    car.brake()
    assert car.speed == speed_brake


@pytest.mark.parametrize("speed_accelerate", speed_data)
def test_car_accelerate(speed_accelerate: int):
    """
    Test the accelerate method of the Car class.

    :param speed_accelerate:  Speed to use for the test.
    :type speed_accelerate: int
    :return: None
    """
    car = Car(50)
    car.accelerate()
    assert car.speed == speed_accelerate
