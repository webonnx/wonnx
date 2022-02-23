# Wonnx command-line interface (`nnx`)

Command-line interface for inference using wonnx

ONNX defines a standardized format to exchange machine learning models. However, up to this point there is no easy way to
perform one-off inference using such a model without resorting to Python. Installation of Python and the required libraries
(e.g. TensorFlow and underlying GPU setup) can be cumbersome. Additionally specific code is always needed to transfer
inputs (images, text, etc.) in and out of the formats required by the model (i.e. image classification models want their
images as fixed-size tensors with the pixel values normalized to specific values, et cetera).

This project provides a very simple all-in-one binary command line tool that can be used to perform inference using ONNX
models on the GPU. Thanks to wonnx,  inference is performed on the GPU. 

NNX tries to make educated guesses about how to transform input and output for a model. These guesses are a default - i.e.
it should always be possible to override them. The goal is to reduce the amount of configuration required to be able to
run a model. Currently the following heuristics are applied:

- The first input and first output specified in the ONNX file are used by default.
- Models taking inputs of shape (1,3,w,h) and (3,w,h) will be fed images resized to w\*h with pixel values normalized to
  0...1 (currently we also apply the SqueezeNet normalization)
- Similarly, models taking inputs of shape (1,1,w,h) and (1,w,h) will be fed black-and-white images with pixel values
  normalized to 0...1.
- When a label file is supplied, an output vector of shape (n,) will be interpreted as providing the probabilities for each
  class. The label for each class is taken from the n'th line in the label file.

## Usage

```sh
$ nnx infer ./data/models/opt-squeeze.onnx -i data=./data/images/pelican.jpeg --labels ./data/models/squeeze-labels.txt --probabilities
n01608432 kite: 21.820244
n02051845 pelican: 21.112095
n02018795 bustard: 20.359694
n01622779 great grey owl, great gray owl, Strix nebulosa: 20.176003
n04417672 thatch, thatched roof: 19.638676
n02028035 redshank, Tringa totanus: 19.606218
n02011460 bittern: 18.90648
n02033041 dowitcher: 18.708323
n01829413 hornbill: 18.595457
n01616318 vulture: 17.508785

$ nnx infer ./data/models/opt-mnist.onnx -i Input3=./data/images/7.jpg
[-1.2942507, 0.5192305, 8.655695, 9.474595, -13.768464, -5.8907413, -23.467274, 28.252314, -6.7598896, 3.9513395]

$ nnx infer ./data/models/opt-mnist.onnx -i Input3=./data/images/7.jpg --labels ./data/models/mnist-labels.txt --top=1
Seven

$ nnx info ./data/models/opt-mnist.onnx
+------------------+---------------------------------------------------+
| Model version    | 1                                                 |
+------------------+---------------------------------------------------+
| IR version       | 3                                                 |
+------------------+---------------------------------------------------+
| Producer name    | CNTK                                              |
+------------------+---------------------------------------------------+
| Producer version | 2.5.1                                             |
+------------------+---------------------------------------------------+
| Opsets           | 8                                                 |
+------------------+---------------------------------------------------+
| Inputs           | +--------+-------------+-----------+------+       |
|                  | | Name   | Description | Shape     | Type |       |
|                  | +--------+-------------+-----------+------+       |
|                  | | Input3 |             | 1x1x28x28 | f32  |       |
|                  | +--------+-------------+-----------+------+       |
+------------------+---------------------------------------------------+
| Outputs          | +------------------+-------------+-------+------+ |
|                  | | Name             | Description | Shape | Type | |
|                  | +------------------+-------------+-------+------+ |
|                  | | Plus214_Output_0 |             | 1x10  | f32  | |
|                  | +------------------+-------------+-------+------+ |
+------------------+---------------------------------------------------+
| Ops used         | +---------+---------------------+                 |
|                  | | Op      | Attributes          |                 |
|                  | +---------+---------------------+                 |
|                  | | Reshape |                     |                 |
|                  | +---------+---------------------+                 |
|                  | | Gemm    | transA=0            |                 |
|                  | |         | transB=0            |                 |
|                  | |         | beta=1              |                 |
|                  | |         | alpha=1             |                 |
|                  | +---------+---------------------+                 |
|                  | | Relu    |                     |                 |
|                  | +---------+---------------------+                 |
|                  | | Conv    | auto_pad=SAME_UPPER |                 |
|                  | |         | group=1             |                 |
|                  | |         | strides=<INTS>      |                 |
|                  | |         | dilations=<INTS>    |                 |
|                  | |         | kernel_shape=<INTS> |                 |
|                  | +---------+---------------------+                 |
|                  | | MaxPool | strides=<INTS>      |                 |
|                  | |         | kernel_shape=<INTS> |                 |
|                  | |         | auto_pad=NOTSET     |                 |
|                  | |         | pads=<INTS>         |                 |
|                  | +---------+---------------------+                 |
+------------------+---------------------------------------------------+
```

- Replace `nnx` with `cargo run --release --` to run development version
- Prepend `RUST_LOG=wonnx-cli=info` to see useful logging from the CLI tool, `RUST_LOG=wonnx=info` to see logging from WONNX.

## CPU inference using `tract`

The nnx utility can use [tract](https://github.com/sonos/tract) as CPU-based backend for ONNX inference. In order to use
this, nnx needs to be compiled with the `cpu` feature enabled. You can then specify one of the following arguments:
*  `--backend cpu` to select the cpu backend
* `--fallback` to select the cpu backend when the gpu backend cannot be used (e.g. because of an unsupported operation type)
* `--compare` to run inference on both CPU and GPU backends and compare the output
* `--benchmark` to run the specified inference a hundred times, then report the performance
* `--compare --benchmark` to run inference on both CPU and GPU a hundred times each, and compare the performance

A benchmarking example (the below result was obtained on an Apple M1 Max system):

```sh
# Run from workspace root
$ cargo run --release --features=cpu -- infer ./data/models/opt-squeeze.onnx -i data=./data/images/pelican.jpeg --compare --benchmark
OK (gpu=572ms, cpu=1384ms, 2.42x)
```

## End-to-end example with Keras

1. `pip install tensorflow onnx tf2onnx`

2. Create a very simple model for the MNIST digits:

```python
from tensorflow.keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# train_images will be (60000,28,28) i.e. 60k black-and-white images of 28x28 pixels (which are ints between 0..255)
# train_labels will be (60000,) i.e. 60k integers ranging 0...9
# test_images/test_labels are similar but only have 10k items

# Build model
from tensorflow import keras
from tensorflow.keras import layers

# Convert images to have pixel values as floats between 0...1
train_images_input = train_images.astype("float32") / 255

model = keras.Sequential([
    layers.Reshape((28*28,), input_shape=(28,28)),
    layers.Dense(512, activation = 'relu'),
    layers.Dropout(rate=0.01),
    layers.Dense(10,  activation = 'softmax')
])

model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(train_images_input, train_labels, epochs=20, batch_size=1024)
```

3. Save Keras model to ONNX with inferred dimensions:

```python
import tf2onnx
import tensorflow as tf
import onnx
input_signature = [tf.TensorSpec([1,28,28], tf.float32, name='input')]
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=13)

from onnx import helper, shape_inference
inferred_model = shape_inference.infer_shapes(onnx_model)

onnx.save(onnx_model, "tymnist.onnx")
onnx.save(inferred_model, "tymnist-inferred.onnx")
```

4. Infer with NNX:

```sh
nnx  ./tymnist-inferred.onnx infer -i input=./data/mnist-7.png --labels ./data/models/mnist-labels.txt
```

5. compare inference result with what Keras would generate (`pip install numpy pillow matplotlib`):

```python
import PIL
import numpy
import matplotlib.pyplot as plt
m5 = PIL.Image.open("data/mnist-7.png").resize((28,28), PIL.Image.ANTIALIAS)
nm5 = numpy.array(m5).reshape((1,28,28))
model.predict(nm5)
```