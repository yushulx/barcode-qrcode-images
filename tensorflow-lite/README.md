## Steps to Train the Qr code Detector with TensorFlow Lite Model Maker
1. Install TensorFlow Lite Model Maker:
    
    ```bash
    pip install tflite-model-maker
    ```
2. Partition the image set:
    
    ```bash
    python partition_dataset.py -x -i ../images -r 0.1 -o ./
    ```
3. Train the model:
    
    ```bash
    python train.py
    ```
4. Test the model:
    
    ```bash
    python test.py
    ```

## Android Demo: QR Code Detection with TensorFlow Lite

![QR Code detection with TensorFlow Lite](https://www.dynamsoft.com/codepool/img/2022/01/tensorflow-lite-qr-code-detection.jpg)

## References
- [TensorFlow Lite Model Maker](https://www.tensorflow.org/lite/guide/model_maker)
- [DataLoader](https://www.tensorflow.org/lite/api_docs/python/tflite_model_maker/object_detector/DataLoader)
- [Integrate object detectors](https://www.tensorflow.org/lite/inference_with_metadata/task_library/object_detector)
- [TensorFlow examples](https://github.com/tensorflow/examples.git)

