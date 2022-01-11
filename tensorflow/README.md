## Installation
1. [protobuf](https://github.com/protocolbuffers/protobuf)
2. Object Detection API with TensorFlow 2
    
    ```bash
    git clone https://github.com/tensorflow/models.git
    cd models/research
    # Compile protos.
    protoc object_detection/protos/*.proto --python_out=.
    # Install TensorFlow Object Detection API.
    cp object_detection/packages/tf2/setup.py .
    python -m pip install --use-feature=2020-resolver .
    # Test the installation.
    python object_detection/builders/model_builder_tf2_test.py
    ```

## Steps to Train the Qr code Detector with Tensorflow
1. Partition the image set    
    
    ```bash
    cd tensorflow
    python partition_dataset.py -x -i ../images -r 0.1 -o ./
    ```
2. Create `label_map.pbtxt` 
    
    ```pbtxt
    item {
        id: 1
        name: 'QR_CODE'
    }
    ```
3. Convert image and XML files to Tensorflow records:
    
    ```bash
    python generate_tfrecord.py -x train -l annotations/label_map.pbtxt -o annotations/train.record
    python generate_tfrecord.py -x test -l annotations/label_map.pbtxt -o annotations/test.record
    ```
3. Download and etract the pre-trained [SSD ResNet50 V1 FPN 640x640](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz) model.
4. Edit `ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/pipeline.config`:

    ```json
    num_classes: 1 # We only have one Qr code class
    batch_size: 32 # Increase/Decrease this value depending on the available memory (Higher values require more memory and vice-versa)
    fine_tune_checkpoint: "ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0" # Path to checkpoint of pre-trained model
    fine_tune_checkpoint_type: "detection" # Set this to "detection" since we want to be training the full detection model
    use_bfloat16: false # Set this to false if you are not training on a TPU
    
    train_input_reader {
      label_map_path: "annotations/label_map.pbtxt" # Path to label map file
      tf_record_input_reader {
        input_path: "annotations/train.record" # Path to training TFRecord file
      }
    }

    eval_input_reader {
      label_map_path: "annotations/label_map.pbtxt" # Path to label map file
      shuffle: false
      num_epochs: 1
      tf_record_input_reader {
        input_path: "annotations/test.record" # Path to testing TFRecord
      }
    }
    ```
5. Train the model.
    
    ```bash
    python model_main_tf2.py --model_dir=models/my_ssd_resnet50_v1_fpn --pipeline_config_path=models/my_ssd_resnet50_v1_fpn/pipeline.config
    ```
6. Monitor the training progress:
    
    ```bash
    tensorboard --logdir=models/my_ssd_resnet50_v1_fpn
    ```

7. Evaluate the trained model.
    
    ```bash
    python model_main_tf2.py --model_dir=models/my_ssd_resnet50_v1_fpn --pipeline_config_path=models/my_ssd_resnet50_v1_fpn/pipeline.config  --checkpoint_dir=models/my_ssd_resnet50_v1_fpn   
    ```
8. Export the model.
    
    ```bash
    python .\exporter_main_v2.py --input_type image_tensor --pipeline_config_path .\models\my_ssd_resnet50_v1_fpn\pipeline.config --trained_checkpoint_dir .\models\my_ssd_resnet50_v1_fpn\ --output_directory .\exported-models\my_ssd_resnet50_v1_fpn
    ```

## References
- [https://github.com/tensorflow/models](https://github.com/tensorflow/models)
- [https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html)
- [https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)
- [https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_training_and_evaluation.md](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_training_and_evaluation.md)