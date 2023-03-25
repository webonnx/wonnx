.PHONY = all clean wasm-test python-test python-test-op python-test-backend test wonnx-test
.DEFAULT_GOAL := wonnx
PYTHON = python3
OP_TESTED = reduce_sum

all: wonnx wasm python

clean:
	rm -rf target
	rm -rf wonnx-py/.venv

test: wonnx-test python-test

wasm:
	RUSTFLAGS=--cfg=web_sys_unstable_apis wasm-pack build --target web -d `pwd`/target/pkg --out-name wonnx --scope webonnx ./wonnx-wasm

wasm-debug:
	RUSTFLAGS=--cfg=web_sys_unstable_apis wasm-pack build --target web -d `pwd`/target/pkg --out-name wonnx --scope webonnx ./wonnx-wasm --dev

wasm-test:
	@echo "Open http://localhost:8080/wonnx-wasm/ in your browser"
	$(PYTHON) -m http.server 8080

target/release/nnx: wonnx/src/*.rs
	cargo build --release

target/debug/nnx: wonnx/src/*.rs
	cargo build

wonnx: target/release/nnx

wonnx-debug: target/debug/nnx

wonnx-test:
	cargo test
	
venv = wonnx-py/.venv
	
$(venv):
	cd wonnx-py; $(PYTHON) -m venv .venv; source ./.venv/bin/activate; pip install -r requirements.txt

python: $(venv)
	cd wonnx-py; source ./.venv/bin/activate; maturin build

python-develop: $(venv) wonnx/src/*.rs wonnx-py/src/*.rs
	cd wonnx-py; source ./.venv/bin/activate; maturin develop

python-test-backend: python-develop
	cd wonnx-py; source ./.venv/bin/activate; pytest ./tests/test_onnx_backend.py

python-test-op: python-develop
	cd wonnx-py; source ./.venv/bin/activate; pytest ./tests/test_specific_op.py

python-test-general: python-develop
	cd wonnx-py; source ./.venv/bin/activate; pytest ./tests/test_wonnx.py

python-test: python-test-backend python-test-general