cd wonnx-py
rustup override set nightly-2022-01-01
export RUSTFLAGS='-C target-feature=+fxsr,+sse,+sse2,+sse3,+ssse3,+sse4.1,+popcnt'
python3 -m venv .env
source .env/bin/activate
pip install maturin
pip install -r requirements.txt
maturin develop
pytest tests/test_onnx_backend.py