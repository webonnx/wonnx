import onnx
import wonnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto

import numpy as np
import cv2


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

    session = wonnx.PySession.from_bytes(onnx._serialize(model))
    inputs = {"x": [-1.0, 2.0]}
    assert session.run(inputs) == {"y": [0.0, 2.0]}, "Single Relu does not work"


def test_from_path():

    # Create the model (ModelProto)
    session = wonnx.PySession.from_path(
        "../examples/data/models/single_relu.onnx"
    )
    inputs = {"x": [-1.0, 2.0]}
    assert session.run(inputs) == {"y": [0.0, 2.0]}, "Single Relu does not work"


def test_mnist():

    image = cv2.imread("../examples/data/images/7.jpg")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (28, 28)).astype(np.float32) / 255
    input = np.reshape(gray, (1, 1, 28, 28))
    # Create the model (ModelProto)

    session = wonnx.PySession.from_path(
        "../examples/data/models/opt-mnist.onnx"
    )
    inputs = {"Input3": input.flatten().tolist()}
    assert (
        np.argmax(session.run(inputs)["Plus214_Output_0"]) == 7
    ), "MNIST does not work"


def test_squeezenet():

    image = cv2.imread("../examples/data/images/7.jpg")

    # Create the model (ModelProto)

    session = wonnx.PySession.from_path(
        "../examples/data/models/opt-squeeze.onnx"
    )
    inputs = {"Input3": input.flatten().tolist()}
    assert (
        np.argmax(session.run(inputs)["Plus214_Output_0"]) == 7
    ), "MNIST does not work"
