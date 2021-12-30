## Installation
- [Darknet](https://github.com/AlexeyAB/darknet.git)

## Steps to Train the Qr code Detector with YOLOv4
1. Partition the image set:  
    
    ```bash
    python partition_dataset.py -i ../images -r 0.1
    ```
2. Train the weights:
    
    ```bash
    darknet detector train data/obj.data yolov4-tiny-custom-640.cfg yolov4-tiny.conv.29
    ```

3. Test the Qr detector:
    
    ```bash
    darknet detector test data/obj.data yolov4-tiny-custom-640.cfg backup/yolov4-tiny-custom-640_last.weights sample/test01.png
    ```

## Sample Usage

```bash
pip install opencv-python
cd sample
python qr_detector.py
```