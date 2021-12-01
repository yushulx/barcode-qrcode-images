import cv2 as cv
import numpy as np
import time

def detect(filename, frame):
    width = 640
    height = 640
    if frame.shape[1] > 1024 or frame.shape[0] > 1024:
        width = 1024
        height = 1024
    model.setInputParams(size=(width, height), scale=1/255, swapRB=True)

    # Inferencing
    CONFIDENCE_THRESHOLD = 0.2
    NMS_THRESHOLD = 0.4
    COLOR_RED = (0,0,255)
    COLOR_BLUE = (255,0,0)
    start_time = time.time()
    classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    elapsed_ms = time.time() - start_time

    cv.putText(frame, '%.2f s, Qr found: %d' % (elapsed_ms, len(classes)), (40, 40), cv.FONT_HERSHEY_SIMPLEX, 1, COLOR_RED, 2)
    for (classid, score, box) in zip(classes, scores, boxes):
        label = "%s : %f" % (class_names[classid], score) 
        cv.rectangle(frame, box, COLOR_BLUE, 2)
        cv.putText(frame, label, (box[0], box[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_BLUE, 2)

    cv.imshow(filename, frame)

# Load class names
class_names = open('../data/obj.names').read().strip().split('\n')

# Load YOLOv4-tiny model
net = cv.dnn.readNetFromDarknet('../yolov4-tiny-custom.cfg', '../backup/yolov4-tiny-custom_last.weights')
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
# net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU) # DNN_TARGET_OPENCL DNN_TARGET_CPU DNN_TARGET_CUDA
# net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)
model = cv.dnn_DetectionModel(net)

frame = cv.imread('test01.png')
detect('test01.png', frame)

frame = cv.imread('test02.png')
detect('test02.png', frame)

cv.waitKey()