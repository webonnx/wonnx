# WONNX Python module

This crate allows using WONNX from Python.

## Building

To build the Python module for development:

````bash
cd wonnx-py
python3 -m venv .venv
source .venv/bin/activate
pip install maturin
maturin develop
````

You can also use `make python` from the project root. If you want to specify a specific version of python, use:
`make python PYTHON=python3.10` (e.g. if you used `brew install python@3.10` to install Python 3.10 on macOS). 

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