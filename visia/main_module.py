"""
Description
===========

This is a module for running the Visia-Science main Pipeline.
"""


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


if __name__ == "__main__":
    my_car = Car()
    print("I'm a car!")
    while True:
        action = input(
            "What should I do? [A]ccelerate, [B]rake, "
            "show [O]dometer, or show average [S]peed?"
        ).upper()
        if action not in "ABOS" or len(action) != 1:
            print("I don't know how to do that")
            continue
        if action == "A":
            my_car.accelerate()
            print("Accelerating...")
        elif action == "B":
            my_car.brake()
            print("Braking...")
        elif action == "O":
            print("The car has driven {} kilometers".format(my_car.odometer))
        elif action == "S":
            print("The car's average speed was {} kph".format(my_car.average_speed()))
        my_car.step()
