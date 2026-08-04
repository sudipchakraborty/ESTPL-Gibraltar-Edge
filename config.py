import json


class Config:

    def __init__(self):

        with open(
            "config/camera.json",
            "r"
        ) as file:

            self.data = json.load(file)


    @property
    def cameras(self):

        return self.data["cameras"]