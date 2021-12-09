# wonnx


![GitHub Workflow Status](https://img.shields.io/github/workflow/status/haixuantao/wonnx/CI)
![Crates.io (latest)](https://img.shields.io/crates/dv/wonnx)
![Crates.io](https://img.shields.io/crates/l/wonnx)


Wonnx aims for running blazing Fast AI on any device.

## Supported Platforms (enabled by `wgpu`)

   API   |    Windows                    |  Linux & Android   |    macOS & iOS     |
  -----  | ----------------------------- | ------------------ | ------------------ |
  Vulkan | :white_check_mark:            | :white_check_mark: |                    |
  Metal  |                               |                    | :white_check_mark: |
  DX12   | :white_check_mark: (W10 only) |                    |                    |
  DX11   | :construction:                |                    |                    |
  GLES3  |                               | :ok:               |                    |

:white_check_mark: = First Class Support — :ok: = Best Effort Support — :construction: = Unsupported, but support in progress

## Getting Started

- Install Rust
- Install Vulkan, Metal, or DX12 for the GPU API.
- git clone this repo.

```bash
git clone https://github.com/haixuanTao/wonnx.git
```

- Then with git lfs installed: 

```bash
cargo run --example squeeze --release
```

## To run a model from scratch

- To run an onnx model, first simplify it with [onnx-simplifier](https://github.com/daquexian/onnx-simplifier), with the command:

```bash
# pip install -U pip && pip install onnx-simplifier
python -m onnxsim mnist-8.onnx  opt-mnist.onnx
```

- Then you can run it following the example in the examples folder:

```bash
cargo run --example mnist --release
```

## To use

```rust
async fn execute_gpu() -> Vec<f32> {
    // USER INPUT

    let n: usize = 512 * 512 * 128;
    let mut input_data = HashMap::new();
    let data = vec![-1.0f32; n];
    input_data.insert("x", data.as_slice());

    let mut session = wonnx::Session::from_path("examples/data/models/single_relu.onnx")
        .await
        .unwrap();

    wonnx::run(&mut session, input_data).await.unwrap()
}
```

> Examples are available in the `examples` folder
 
## Test 

```bash
cargo test
```

## Test WASM (not yet implemented)
```bash
export RUSTFLAGS=--cfg=web_sys_unstable_apis
wasm-pack test --node
```

## Language interface

Aiming to be widely usable through:

- a Python binding using PyO3
- a JS binding using WASM



