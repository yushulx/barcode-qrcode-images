# https://www.tensorflow.org/lite/tutorials/model_maker_object_detection
from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.config import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

import tensorflow as tf
assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)

spec = model_spec.get('efficientdet_lite0')
# https://www.tensorflow.org/lite/api_docs/python/tflite_model_maker/object_detector/DataLoader#from_pascal_voc
train_data = object_detector.DataLoader.from_pascal_voc(images_dir="train", annotations_dir="train", label_map={1: "QR_CODE"} )
validation_data = object_detector.DataLoader.from_pascal_voc(images_dir="test", annotations_dir="test", label_map={1: "QR_CODE"} )
model = object_detector.create(train_data, model_spec=spec, batch_size=8, train_whole_model=True, validation_data=validation_data)
model.evaluate(validation_data)
model.export(export_dir='.')
model.evaluate_tflite('model.tflite', validation_data)