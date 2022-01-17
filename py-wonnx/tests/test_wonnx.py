import onnx
import wonnx


def test_parse_model():
    model = onnx.load_model(
        "/home/peter/Documents/BLOG/wonnx/examples/data/models/single_relu.onnx"
    )
    session = wonnx.wonnx.PySession.from_bytes(onnx._serialize(model))
    inputs = {"x": [-1.0, 2.0]}
    return session.run(inputs)


test_parse_model()
