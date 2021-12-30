import json
from pathlib import Path

import os

JSON_EXT = '.json'

class CreateMLWriter:
    def __init__(self, foldername, filename, imgsize, shapes, outputfile, databasesrc='Unknown', localimgpath=None):
        self.foldername = foldername
        self.filename = filename
        self.databasesrc = databasesrc
        self.imgsize = imgsize
        self.boxlist = []
        self.localimgpath = localimgpath
        self.verified = False
        self.shapes = shapes
        self.outputfile = outputfile

    def write(self):
        if os.path.isfile(self.outputfile):
            with open(self.outputfile, "r") as file:
                input_data = file.read()
                outputdict = json.loads(input_data)
        else:
            outputdict = []

        outputimagedict = {
            "image": self.filename,
            "annotations": []
        }

        for shape in self.shapes:
            points = shape["points"]

            x1 = points[0][0]
            y1 = points[0][1]
            x2 = points[1][0]
            y2 = points[1][1]

            height, width, x, y = self.calculate_coordinates(x1, x2, y1, y2)

            shapedict = {
                "label": shape["label"],
                "coordinates": {
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height
                }
            }
            outputimagedict["annotations"].append(shapedict)

        exists = False
        for i in range(0, len(outputdict)):
            if outputdict[i]["image"] == outputimagedict["image"]:
                exists = True
                outputdict[i] = outputimagedict
                break

        if not exists:
            outputdict.append(outputimagedict)

        Path(self.outputfile).write_text(json.dumps(outputdict), 'utf-8')

    def calculate_coordinates(self, x1, x2, y1, y2):
        if x1 < x2:
            xmin = x1
            xmax = x2
        else:
            xmin = x2
            xmax = x1
        if y1 < y2:
            ymin = y1
            ymax = y2
        else:
            ymin = y2
            ymax = y1
        width = xmax - xmin
        if width < 0:
            width = width * -1
        height = ymax - ymin
        x = xmin + width / 2
        y = ymin + height / 2
        return height, width, x, y


class CreateMLReader:
    def __init__(self, jsonpath, filepath):
        self.jsonpath = jsonpath
        self.shapes = []
        self.verified = False
        self.filename = filepath.split("/")[-1:][0]
        try:
            self.parse_json()
        except ValueError:
            print("JSON decoding failed")

    def parse_json(self):
        with open(self.jsonpath, "r") as file:
            inputdata = file.read()

        outputdict = json.loads(inputdata)
        self.verified = True

        if len(self.shapes) > 0:
            self.shapes = []
        for image in outputdict:
            if image["image"] == self.filename:
                for shape in image["annotations"]:
                    self.add_shape(shape["label"], shape["coordinates"])

    def add_shape(self, label, bndbox):
        xmin = bndbox["x"] - (bndbox["width"] / 2)
        ymin = bndbox["y"] - (bndbox["height"] / 2)

        xmax = bndbox["x"] + (bndbox["width"] / 2)
        ymax = bndbox["y"] + (bndbox["height"] / 2)

        points = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
        self.shapes.append((label, points, None, None, True))

    def get_shapes(self):
        return self.shapes