# wonnx

Wonnx aimed at being an ONNX Runtime for every GPU on all platforms written in 100% Rust.

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
- Then: 
```bash
cargo run --example custom_graph
```

## To use

```rust
async fn execute_gpu() -> Option<Vec<f32>> {
    // USER INPUT

    let n: usize = 512 * 512 * 128;
    let mut input_data = HashMap::new();
    let data = vec![-1.0f32; n];
    let dims = vec![n as i64];
    input_data.insert("x", (data.as_slice(), dims.as_slice()));

    let mut session = wonnx::Session::from_path("tests/single_relu.onnx")
        .await
        .unwrap();

    session.run(input_data).await
}
```

## Test 

```bash
cargo test
```

## Test WASM (not yet implemented)
```bash
export RUSTFLAGS=--cfg=web_sys_unstable_apis
wasm-pack test --node
```

## Examples are available in the `tests` folder


