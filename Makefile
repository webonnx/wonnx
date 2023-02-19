.PHONY = all clean
.DEFAULT_GOAL := wonnx

wasm:
	RUSTFLAGS=--cfg=web_sys_unstable_apis wasm-pack build --target web -d `pwd`/target/pkg --out-name wonnx --scope webonnx ./wonnx-wasm

wasm-debug:
	RUSTFLAGS=--cfg=web_sys_unstable_apis wasm-pack build --target web -d `pwd`/target/pkg --out-name wonnx --scope webonnx ./wonnx-wasm --dev

wasm-test:
	@echo "Open http://localhost:8080/wonnx-wasm/ in your browser"
	python3 -m http.server 8080

wonnx:
	cargo build --release

wonnx-debug:
	cargo build

all: wonnx wasm

clean:
	rm -rf target