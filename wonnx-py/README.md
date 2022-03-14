# WONNX Python module

This crate allows using WONNX from Python.

## Building

To build the Python module for development:

````sh
cd wonnx-py
python3 -m venv .env
source .env/bin/activate
pip install maturin
maturin develop
pip install -r requirements.txt
````

## Testing

To test a specific operator, you can use the following command:
```bash
OP_TESTED=reduce pytest tests/test_specific_op.py
```

To test the current set of fully tested op:
```bash
python tests/test_onnx_backend.py
```