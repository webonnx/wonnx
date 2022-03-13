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
````