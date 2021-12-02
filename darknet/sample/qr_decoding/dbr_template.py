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

def decode(filename, template_name):
    frame = cv.imread(filename)

    template_path = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + template_name
    settings = reader.reset_runtime_settings() 
    error = reader.init_runtime_settings_with_file(template_path, EnumConflictMode.CM_OVERWRITE)

    before = time.time()
    results = reader.decode_buffer(frame)
    after = time.time()

    COLOR_RED = (0,0,255)
    thickness = 2
    if results != None:
        found = len(results)
        for result in results:
            text = result.barcode_text 
            points = result.localization_result.localization_points
            data = np.array([[points[0][0], points[0][1]], [points[1][0], points[1][1]], [points[2][0], points[2][1]], [points[3][0], points[3][1]]])
            cv.drawContours(image=frame, contours=[data], contourIdx=-1, color=COLOR_RED, thickness=thickness, lineType=cv.LINE_AA)
            cv.putText(frame, result.barcode_text, points[0], cv.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_RED)

        cv.putText(frame, '%.2f s, Qr found: %d' % (after - before, found), (20, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_RED)
    else:
        cv.putText(frame, '%.2f s, Qr found: %d' % (after - before, 0), (20, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_RED)

    cv.imshow(template_name, frame)

decode("test.jpg", "l1.json")
decode("test.jpg", "l2.json")
decode("test.jpg", "l3.json")
decode("test.jpg", "l4.json")
decode("test.jpg", "l5.json")
cv.waitKey(0)