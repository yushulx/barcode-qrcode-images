""" usage: barcode_annotation_generator.py [-i IMAGEDIR]

Generate barcode annotations for machine learning

optional arguments:
  -i IMAGEDIR, --imageDir IMAGEDIR
                        Path to the folder where the image dataset is stored. If not specified, the CWD will be used.
"""
import os
import re
from shutil import copyfile
import argparse
import math
import random
from dbr import *
import cv2 as cv
from labelFile import *
from pascal_voc_io import *
from yolo_io import *
from create_ml_io import *

license_key = "LICENSE-KEY" # Get license key from https://www.dynamsoft.com/customer/license/trialLicense?product=dbr
reader = BarcodeReader()
reader.init_license(license_key)


def saveLabelsAllFormat(annotationFilePath, filename, shapes, imagePath, classList):
    labelFile = LabelFile()

    try:
        xmlfile = annotationFilePath + XML_EXT
        labelFile.savePascalVocFormat(xmlfile, shapes, imagePath, classList)

        txtfile = annotationFilePath + TXT_EXT
        labelFile.saveYoloFormat(txtfile, shapes, imagePath, classList)

        jsonfile = annotationFilePath + JSON_EXT
        labelFile.saveCreateMLFormat(jsonfile, shapes, imagePath, classList)

        return True
    except LabelFileError as e:
        print(e)
        return False


def scan_folder(source):
    filename = 'classes.txt'
    class_names = []
    if os.path.exists(filename):
        class_names = open(filename).read().strip().split('\n')
        print(class_names)

    files = os.listdir(source)
    for file in files:
        if file.endswith(".jpg") or file.endswith(".png"):
            print(file)
            savedFileName = os.path.splitext(file)[0]
            savedPath = os.path.join(source, savedFileName)
            absolute_path = os.path.join(source, file)

            frame = cv.imread(absolute_path)
            results = reader.decode_buffer(frame)
            # COLOR_RED = (0, 0, 255)
            # thickness = 2
            if results != None:
                annotations = []
                for result in results:
                    # text = result.barcode_text
                    annotation = {}
                    format = result.barcode_format_string
                    points = result.localization_result.localization_points
                    annotation['label'] = format
                    annotation['points'] = []
                    annotation['difficult'] = 0
                    annotations.append(annotation)
                    # cv.line(frame, points[0], points[1], COLOR_RED, thickness)
                    # cv.line(frame, points[1], points[2], COLOR_RED, thickness)
                    # cv.line(frame, points[2], points[3], COLOR_RED, thickness)
                    # cv.line(frame, points[3], points[0], COLOR_RED, thickness)
                    delta = 10
                    xmin, ymin = min(points[0][0], points[1][0], points[2][0], points[3][0]) - delta, min(
                        points[0][1], points[1][1], points[2][1], points[3][1]) - delta
                    xmax, ymax = max(points[0][0], points[1][0], points[2][0], points[3][0]) + delta, max(
                        points[0][1], points[1][1], points[2][1], points[3][1]) + delta

                    if xmin < 0:
                        xmin = 0
                    if ymin < 0:
                        ymin = 0
                    if xmax > frame.shape[1]:
                        xmax = frame.shape[1]
                    if ymax > frame.shape[0]:
                        ymax = frame.shape[0]

                    # cv.rectangle(frame, (xmin, ymin),
                    #              (xmax, ymax), COLOR_RED, 2)
                    annotation['points'].append((xmin, ymin))
                    annotation['points'].append((xmax, ymax))
            saveLabelsAllFormat(savedPath, file, annotations, absolute_path, class_names)
            # cv.imshow('DBR Detection', frame)
            # cv.waitKey(0)

def main():

    parser = argparse.ArgumentParser(description="Generate barcode annotations for machine learning",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-i', '--imageDir',
        help='Path to the folder where the image dataset is stored.',
        type=str,
        default=os.getcwd()
    )
    args = parser.parse_args()

    scan_folder(args.imageDir)


if __name__ == '__main__':
    main()
