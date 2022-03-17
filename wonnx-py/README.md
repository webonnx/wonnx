# WONNX Python module

This crate allows using WONNX from Python.

## Building

To build the Python module for development:

````bash
cd wonnx-py
python3 -m venv .env
source .env/bin/activate
pip install maturin
maturin develop
````

## Testing

For testing, additional dependencies are required. First, ensure you have the protobuf compiler (`protoc`) installed and
in your PATH (on macOS, you can use `brew install protobuf` and possibly `brew link protobuf` to install it). Nextl, install
Python dependencies:

````bash
pip install -r requirements.txt
````

To test a specific operator, you can use the following command:

```bash
OP_TESTED=reduce pytest tests/test_specific_op.py
```

To test the current set of fully tested op:

```bash
python tests/test_onnx_backend.py
```