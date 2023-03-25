import onnx
import wonnx
from onnx import helper
from onnx import TensorProto
from torchvision import transforms
import numpy as np
import cv2

import os
basedir = os.path.dirname(os.path.realpath(__file__))

def test_parse_model():
    node_def = helper.make_node(
        "Relu",  # node name
        ["x"],  # inputs
        ["y"],  # outputs
    )
    X = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])
    Y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        [node_def],
        "test-model",
        [X],
        [Y],
    )

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name="onnx-example")
    model = onnx.shape_inference.infer_shapes(model_def)

    session = wonnx.Session.from_bytes(onnx._serialize(model))
    inputs = {"x": [-1.0, 2.0]}
    assert session.run(inputs) == {"y": [0.0, 2.0]}, "Single Relu does not work"


def test_from_path():
    # Create the model (ModelProto)
    session = wonnx.Session.from_path(
        os.path.join(basedir, "../../data/models/single_relu.onnx")
    )
    inputs = {"x": [-1.0, 2.0]}
    assert session.run(inputs) == {"y": [0.0, 2.0]}, "Single Relu does not work"


def test_mnist():
    image = cv2.imread(os.path.join(basedir, "../../data/images/7.jpg"))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (28, 28)).astype(np.float32) / 255
    input = np.reshape(gray, (1, 1, 28, 28))

    # Create the model (ModelProto)
    session = wonnx.Session.from_path(
        os.path.join(basedir, "../../data/models/opt-mnist.onnx")
    )
    inputs = {"Input3": input.flatten().tolist()}
    assert (
        np.argmax(session.run(inputs)["Plus214_Output_0"]) == 7
    ), "MNIST does not work"


def test_squeezenet():
    image = cv2.imread(os.path.join(basedir, "../../data/images/pelican.jpeg"))
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # apply transforms to the input image
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    input_tensor = transform(rgb_image)

    # Create the model (ModelProto)
    session = wonnx.Session.from_path(
        os.path.join(basedir, "../../data/models/opt-squeeze.onnx")
    )
    inputs = {"data": input_tensor.flatten().tolist()}
    result = session.run(inputs)["squeezenet0_flatten0_reshape0"]

    print(f"result 144={result[144]} argmax={np.argmax(result)} score_max={np.max(result)}")

    assert (
        np.argmax(result) == 144
    ), "Squeezenet does not work"
