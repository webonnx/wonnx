# WONNX WebAssembly package

This crate allows using WONNX on the Web through WebAssembly (WASM). 

## How it works

The wonnx create is compiled to a WASM module which a browser can run at near-native speed. The module also comes with a thin binding layer in JavaScript, which allows JS code to call public functions in the WASM module, and allows the WASM module to call certain functions in the JS code. To perform inference, the JS code calls into the WASM module to load an ONNX model and inference inputs into memory. WONNX will then compile the model into shader code, just as it would when runing natively. However, in this case WONNX will use the WebGPU API in the browser (which it has access to through the binding layer) to perform the heavy lifting. 

## Usage

The wonnx-wasm package provides a simple interface using which you can perform inference using ONNX models, mostly resembling the API of the native version. The API looks like this:

````js
import init, { Session, Input } from "/target/pkg/wonnx.js";

async funtion run() {
	await init();

	try {
		const session = await Session.fromBytes(modelBytes /* Uint8Array containing the ONNX file */);
		const input = new Input();
		input.insert("x", [13.0, -37.0]);
		const result = await session.run(input); // This will be an object where the keys are the names of the model outputs and the values are arrays of numbers.
		session.free();
		input.free();
	}
	catch(e) {
		console.error(e.toString()); // The error will be of type SessionError
	}
}

run();
````

For a basic example, see [index.html](./index.html), or see [squeeze.html](./squeeze.html) for an example of how to use images.

## Building

From the root of the repository, run the following commands:

````bash
# Install the wasm-pack build tool
cargo install wasm-pack

# Build release version. The RUSTFLAGS are needed because WebGPU APIs are still unstable in web_sys
RUSTFLAGS=--cfg=web_sys_unstable_apis wasm-pack build --target web -d `pwd`/target/pkg --out-name wonnx ./wonnx-wasm

# Add --dev if you want a debug build
RUSTFLAGS=--cfg=web_sys_unstable_apis wasm-pack build --target web -d `pwd`/target/pkg --out-name wonnx ./wonnx-wasm --dev
````

To test the freshly built package, run `python3 -m http.server 8080` and open http://localhost:8080/wonnx-wasm/ in a web browser.

Note that you will have to use a browser that supports WebGPU (and has the feature enabled). For Chrome, install Chrome Canary
and enable the `Unsafe WebGPU` flag by navigating to `chrome://flags/#enable-unsafe-webgpu`. For Firefox, enable the
`dom.webgpu.enabled` setting by navigating to `about:config`.