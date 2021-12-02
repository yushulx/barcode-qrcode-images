import cv2 as cv
import numpy as np
import time
from dbr import *
import os

# Initialize Dynamsoft Barcode Reader
reader = BarcodeReader()
# Apply for a trial license: https://www.dynamsoft.com/customer/license/trialLicense
license_key = "LICENSE-KEY"
reader.init_license(license_key)

# Load YOLOv4-tiny model
class_names = open('../../data/obj.names').read().strip().split('\n')
net = cv.dnn.readNetFromDarknet('../../yolov4-tiny-custom.cfg', '../../backup/yolov4-tiny-custom_last.weights')
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)

model = cv.dnn_DetectionModel(net)
width = 640
height = 640

CONFIDENCE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
COLOR_RED = (0,0,255)
COLOR_BLUE = (255,0,0)

def decode(filename, template_name):
    frame = cv.imread(filename)

    if frame.shape[1] > 1024 or frame.shape[0] > 1024:
        width = 1024
        height = 1024
    model.setInputParams(size=(width, height), scale=1/255, swapRB=True)

    template_path = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + template_name
    settings = reader.reset_runtime_settings() 
    error = reader.init_runtime_settings_with_file(template_path, EnumConflictMode.CM_OVERWRITE)

    # YOLO detection
    yolo_start = time.time()
    classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    yolo_end = time.time()
    print("YOLO detection time: %.2f s" % (yolo_end - yolo_start))

    settings = reader.get_runtime_settings()
    settings.expected_barcodes_count = len(classes)
    settings.barcode_format_ids = EnumBarcodeFormat.BF_QR_CODE
    reader.update_runtime_settings(settings)

    before = time.time()
    results = reader.decode_buffer(frame)
    after = time.time()
    total_dbr_time = after - before

    if results != None:
        found = len(results)
        for result in results:
            text = result.barcode_text 
            points = result.localization_result.localization_points
            data = np.array([[points[0][0], points[0][1]], [points[1][0], points[1][1]], [points[2][0], points[2][1]], [points[3][0], points[3][1]]])
            cv.drawContours(image=frame, contours=[data], contourIdx=-1, color=(0, 0, 255), thickness=2, lineType=cv.LINE_AA)
            cv.putText(frame, text, points[0], cv.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_RED)
    else:
        found = 0
    print("With YOLOv4. Decoding Time: %.2f s, total found: %d" % (after - before, found))

    dbr_found = found
    for (classid, score, box) in zip(classes, scores, boxes):
        label = "%s : %f" % (class_names[classid], score) 
        cv.rectangle(frame, box, COLOR_BLUE, 2)
        cv.putText(frame, label, (box[0], box[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_BLUE, 2)


    cv.putText(frame, 'DBR+YOLO %.2f s, DBR found: %d, YOLO found: %d' % (yolo_end - yolo_start + total_dbr_time, dbr_found, len(classes)), (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_RED)
    cv.imshow(template_name, frame)

decode("test.jpg", "l1.json")
decode("test.jpg", "l2.json")
decode("test.jpg", "l3.json")
decode("test.jpg", "l4.json")
decode("test.jpg", "l5.json")
cv.waitKey(0)