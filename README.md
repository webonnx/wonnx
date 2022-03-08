<center><img src="logo.svg" alt="WONNX" width="700"/></center>

![GitHub Workflow Status](https://img.shields.io/github/workflow/status/webonnx/wonnx/CI)
![Crates.io (latest)](https://img.shields.io/crates/dv/wonnx)
![Crates.io](https://img.shields.io/crates/l/wonnx)


Wonnx is a GPU-accelerated ONNX inference run-time written 100% in Rust, ready for the web.

## Supported Platforms (enabled by `wgpu`)

   API   |    Windows                    |  Linux & Android   |    macOS & iOS     |
  -----  | ----------------------------- | ------------------ | ------------------ |
  Vulkan | ✅                            | ✅                 |                    |
  Metal  |                               |                    | ✅                 |
  DX12   | ✅                 (W10 only) |                    |                    |
  DX11   | :construction:                |                    |                    |
  GLES3  |                               | :ok:               |                    |

:white_check_mark: = First Class Support — :ok: = Best Effort Support — :construction: = Unsupported, but support in progress

## Getting started

### From the command line

Ensure your system supports either Vulkan, Metal or DX12 for access to the GPU. Then either download a binary release,
or install Rust and run `cargo install --git https://github.com/webonnx/wonnx.git wonnx-cli` to install the CLI.

The CLI tool (`nnx`) provides a convenient interface for tinkering with models (see the [README](./wonnx-cli/README.md) for more information):

````bash
nnx info ./data/models/opt-squeeze.onnx
nnx infer ./data/models/opt-squeeze.onnx -i data=./data/images/pelican.jpeg --labels ./data/models/squeeze-labels.txt --top 3
````

### From Rust

Add the `wonnx` crate as dependency (`cargo add wonnx` if you have cargo-add). Then, see the [examples](./wonnx/examples)
for usage examples, or [browse the API docs](https://docs.rs/wonnx).

### From Python

```bash
pip install wonnx
```

And then, to use:

```python
from wonnx import PySession
session = PySession.from_path(
    "../data/models/single_relu.onnx"
)
inputs = {"x": [-1.0, 2.0]}
assert session.run(inputs) == {"y": [0.0, 2.0]}
```

Then run `python3` with the above Python code!

For more details on the Python package including build instructions, see [wonnx-py](./wonnx-py/README.md).

### In the browser, using WebGPU + WebAssembly

````bash
npm install @webonnx/wonnx-wasm
````

And then, on the client side:

````js
import init, { Session, Input } from "@webonnx/wonnx-wasm";

// Check for WebGPU availability first: if(navigator.gpu) { .. }
await init();
const session = await Session.fromBytes(modelBytes /* Uint8Array containing the ONNX file */);
const input = new Input();
input.insert("x", [13.0, -37.0]);
const result = await session.run(input); // This will be an object where the keys are the names of the model outputs and the values are arrays of numbers.
session.free();
input.free();
````

The package [@webonnx/wonnx-wasm](https://www.npmjs.com/package/@webonnx/wonnx-wasm) provides an interface to WONNX, 
which is included as WebAssembly module and will use the browser's WebGPU implementation. See [wonnx-wasm-example](https://github.com/webonnx/wonnx-wasm-example)
for a more complete usage example involving a bundler.

For more details on the JS/WASM package including build instructions, see [wonnx-wasm](./wonnx-wasm/README.md).

### For development

To work on wonnx itself, follow the following steps:

- Install Rust
- Install Vulkan, Metal, or DX12 for the GPU API.
- Ensure Git LFS is installed
- git clone this repo.

```bash
git clone https://github.com/webonnx/wonnx.git
git lfs install
```

Ensure Git LFS is initialized and has downloaded the model files (in `wonnx/examples/data/models`). Then, you're all set!

You can run one of the included examples through cargo:

```bash
cargo run --example squeeze --release
```

## Running other models

- To run an onnx model, first simplify it with [onnx-simplifier](https://github.com/daquexian/onnx-simplifier), with the command:

```bash
# pip install -U pip && pip install onnx-simplifier
python -m onnxsim mnist-8.onnx opt-mnist.onnx
```

- Then you can run it following the example in the examples folder:

```bash
cargo run --example mnist --release
```

Examples are available in the [examples folder](./wonnx/examples/).

## Tested models

- Squeezenet
- MNIST

## GPU selection

Except when running in WebAssembly, you may set the following environment variables to influence GPU selection by WGPU:

* `WGPU_ADAPTER_NAME` with a substring of the name of the adapter you want to use (e.g. `1080` will match `NVIDIA GeForce 1080ti`).
* `WGPU_BACKEND` with a comma separated list of the backends you want to use (`vulkan`, `metal`, `dx12`, `dx11`, or `gl`).
* `WGPU_POWER_PREFERENCE` with the power preference to choose when a specific adapter name isn't specified (`high` or `low`)

## Contribution: On implementing a new Operator

Contribution are very much welcomed even without large experience in DL, WGSL, or Rust. I hope that, this project can be a sandbox for all of us to learn more about those technologies beyond this project initial scope.

To implement an operator all you have to do is:
1. Add a new matching pattern in `compiler.rs`
2. Retrieve its attributes values using the `get_attribute` function:
```Rust
    let alpha = get_attribute("alpha", Some(1.0), node);
    // or without default value
    let alpha = get_attribute::<f32>("alpha", None, node);
```
3. Add any variable you want to use in the WGSL shader using `context`.
4. Write a new WGSL template in the `templates` folder.
> Available types are in `structs.wgsl` but you can also generate new ones within your templates.
5. Respect the binding layout that each entry is incremented by 1 starting from 0, with input first and output last. If the number of binding is above 4. Increment the binding group. You can change the input within `sequencer.rs`
6. Write the logic.

There is default variables in the context: 
- `{{ i_lens[0] }}`: the length of the input 0. This also work for output: `{{ o_lens[0] }}` and other input `{{ i_lens[1] }}`
- `{{ i_shape[0] }}`: the array of dimensions of input 0. To get the first dimension of the array, just use: `{{ i_shape[0][0] }}` 
- `{{ i_chunks[0] }}`: the size of the chunks of each dimensions of input 0. By default, each variable is represented as a long array of values where to get to specific values you have to move by chunks. Those chunks are represented within this variable. To get the size of the chunks of the first dimensions use: `{{ i_chunks[0][0] }}`.
- `{{ op_type }}` the op type as some op_type like activation are using the same template.

7. Test it using the utils function and place it in the tests folder. The test can look as follows:
```Rust
#[test]
fn test_matmul_square_matrix() {
    // USER INPUT

    let n = 16;
    let mut input_data = HashMap::new();

    let data_a = ndarray::Array2::eye(n);
    let mut data_b = ndarray::Array2::<f32>::zeros((n, n));
    data_b[[0, 0]] = 0.2;
    data_b[[0, 1]] = 0.5;

    let sum = data_a.dot(&data_b);

    input_data.insert("A".to_string(), data_a.as_slice().unwrap());
    input_data.insert("B".to_string(), data_b.as_slice().unwrap());

    let n = n as i64;
    let model = model(graph(
        vec![tensor("A", &[n, n]), tensor("B", &[n, n])],
        vec![tensor("C", &[n, n])],
        vec![],
        vec![],
        vec![node(vec!["A", "B"], vec!["C"], "MatMul", "MatMul", vec![])],
    ));

    let session =
        pollster::block_on(wonnx::Session::from_model(model)).expect("Session did not create");

    let result = pollster::block_on(session.run(input_data)).unwrap();

    assert_eq!(result["C"].as_slice(), sum.as_slice().unwrap());
}
```
> Check out tera documentation for other templating operation: https://tera.netlify.app/docs/

8. If at any point you want to do optimisation of several node you can do it within `sequencer.rs`.

## Supported Operators (ref [ONNX IR](https://github.com/onnx/onnx/blob/master/docs/Operators.md?plain=1)) 

|**Operator**|**Since version**|**Implemented**|
|-|-|-|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Abs">Abs</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Abs-13">13</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Abs-6">6</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Abs-1">1</a>|✅|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Acos">Acos</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Acos-7">7</a>|✅|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Acosh">Acosh</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Acosh-9">9</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Add">Add</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Add-14">14</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Add-13">13</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Add-7">7</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Add-6">6</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Add-1">1</a>|✅|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#And">And</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#And-7">7</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#And-1">1</a>|✅|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#ArgMax">ArgMax</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#ArgMax-13">13</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#ArgMax-12">12</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#ArgMax-11">11</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#ArgMax-1">1</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#ArgMin">ArgMin</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#ArgMin-13">13</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#ArgMin-12">12</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#ArgMin-11">11</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#ArgMin-1">1</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Asin">Asin</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Asin-7">7</a>|✅|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Asinh">Asinh</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Asinh-9">9</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Atan">Atan</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Atan-7">7</a>|✅|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Atanh">Atanh</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Atanh-9">9</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#AveragePool">AveragePool</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#AveragePool-11">11</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#AveragePool-10">10</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#AveragePool-7">7</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#AveragePool-1">1</a>|✅|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#BatchNormalization">BatchNormalization</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#BatchNormalization-15">15</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#BatchNormalization-14">14</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#BatchNormalization-9">9</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#BatchNormalization-7">7</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#BatchNormalization-6">6</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#BatchNormalization-1">1</a>|✅|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#BitShift">BitShift</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#BitShift-11">11</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Cast">Cast</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Cast-13">13</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Cast-9">9</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Cast-6">6</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Cast-1">1</a>|✅|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Ceil">Ceil</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Ceil-13">13</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Ceil-6">6</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Ceil-1">1</a>|✅|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Clip">Clip</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Clip-13">13</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Clip-12">12</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Clip-11">11</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Clip-6">6</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Clip-1">1</a>|✅|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Compress">Compress</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Compress-11">11</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Compress-9">9</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Concat">Concat</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Concat-13">13</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Concat-11">11</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Concat-4">4</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Concat-1">1</a>|✅|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#ConcatFromSequence">ConcatFromSequence</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#ConcatFromSequence-11">11</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Constant">Constant</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Constant-13">13</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Constant-12">12</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Constant-11">11</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Constant-9">9</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Constant-1">1</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#ConstantOfShape">ConstantOfShape</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#ConstantOfShape-9">9</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Conv">Conv</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Conv-11">11</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Conv-1">1</a>|✅|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#ConvInteger">ConvInteger</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#ConvInteger-10">10</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#ConvTranspose">ConvTranspose</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#ConvTranspose-11">11</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#ConvTranspose-1">1</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Cos">Cos</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Cos-7">7</a>|✅|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Cosh">Cosh</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Cosh-9">9</a>|✅|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#CumSum">CumSum</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#CumSum-14">14</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#CumSum-11">11</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#DepthToSpace">DepthToSpace</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#DepthToSpace-13">13</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#DepthToSpace-11">11</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#DepthToSpace-1">1</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#DequantizeLinear">DequantizeLinear</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#DequantizeLinear-13">13</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#DequantizeLinear-10">10</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Det">Det</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Det-11">11</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Div">Div</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Div-14">14</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Div-13">13</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Div-7">7</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Div-6">6</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Div-1">1</a>|✅|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Dropout">Dropout</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Dropout-13">13</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Dropout-12">12</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Dropout-10">10</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Dropout-7">7</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Dropout-6">6</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Dropout-1">1</a>|✅|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Einsum">Einsum</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Einsum-12">12</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Elu">Elu</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Elu-6">6</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Elu-1">1</a>|✅|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Equal">Equal</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Equal-13">13</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Equal-11">11</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Equal-7">7</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Equal-1">1</a>|✅|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Erf">Erf</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Erf-13">13</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Erf-9">9</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Exp">Exp</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Exp-13">13</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Exp-6">6</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Exp-1">1</a>|✅|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Expand">Expand</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Expand-13">13</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Expand-8">8</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#EyeLike">EyeLike</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#EyeLike-9">9</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Flatten">Flatten</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Flatten-13">13</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Flatten-11">11</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Flatten-9">9</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Flatten-1">1</a>|✅|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Floor">Floor</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Floor-13">13</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Floor-6">6</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Floor-1">1</a>|✅|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#GRU">GRU</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#GRU-14">14</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#GRU-7">7</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#GRU-3">3</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#GRU-1">1</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Gather">Gather</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Gather-13">13</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Gather-11">11</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Gather-1">1</a>|✅ (axis=0)|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#GatherElements">GatherElements</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#GatherElements-13">13</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#GatherElements-11">11</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#GatherND">GatherND</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#GatherND-13">13</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#GatherND-12">12</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#GatherND-11">11</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Gemm">Gemm</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Gemm-13">13</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Gemm-11">11</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Gemm-9">9</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Gemm-7">7</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Gemm-6">6</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Gemm-1">1</a>|✅|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#GlobalAveragePool">GlobalAveragePool</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#GlobalAveragePool-1">1</a>|✅|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#GlobalLpPool">GlobalLpPool</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#GlobalLpPool-2">2</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#GlobalLpPool-1">1</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#GlobalMaxPool">GlobalMaxPool</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#GlobalMaxPool-1">1</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Greater">Greater</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Greater-13">13</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Greater-9">9</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Greater-7">7</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Greater-1">1</a>|✅|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#GridSample">GridSample</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#GridSample-16">16</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#HardSigmoid">HardSigmoid</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#HardSigmoid-6">6</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#HardSigmoid-1">1</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Hardmax">Hardmax</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Hardmax-13">13</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Hardmax-11">11</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Hardmax-1">1</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Identity">Identity</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Identity-16">16</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Identity-14">14</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Identity-13">13</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Identity-1">1</a>|✅|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#If">If</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#If-16">16</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#If-13">13</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#If-11">11</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#If-1">1</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#InstanceNormalization">InstanceNormalization</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#InstanceNormalization-6">6</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#InstanceNormalization-1">1</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#IsInf">IsInf</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#IsInf-10">10</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#IsNaN">IsNaN</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#IsNaN-13">13</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#IsNaN-9">9</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#LRN">LRN</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#LRN-13">13</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#LRN-1">1</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#LSTM">LSTM</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#LSTM-14">14</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#LSTM-7">7</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#LSTM-1">1</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#LeakyRelu">LeakyRelu</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#LeakyRelu-6">6</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#LeakyRelu-1">1</a>|✅|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Less">Less</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Less-13">13</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Less-9">9</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Less-7">7</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Less-1">1</a>|✅|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Log">Log</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Log-13">13</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Log-6">6</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Log-1">1</a>|✅|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Loop">Loop</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Loop-16">16</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Loop-13">13</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Loop-11">11</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Loop-1">1</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#LpNormalization">LpNormalization</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#LpNormalization-1">1</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#LpPool">LpPool</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#LpPool-11">11</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#LpPool-2">2</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#LpPool-1">1</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#MatMul">MatMul</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#MatMul-13">13</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#MatMul-9">9</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#MatMul-1">1</a>|✅|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#MatMulInteger">MatMulInteger</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#MatMulInteger-10">10</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Max">Max</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Max-13">13</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Max-12">12</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Max-8">8</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Max-6">6</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Max-1">1</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#MaxPool">MaxPool</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#MaxPool-12">12</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#MaxPool-11">11</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#MaxPool-10">10</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#MaxPool-8">8</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#MaxPool-1">1</a>|✅|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#MaxRoiPool">MaxRoiPool</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#MaxRoiPool-1">1</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#MaxUnpool">MaxUnpool</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#MaxUnpool-11">11</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#MaxUnpool-9">9</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Mean">Mean</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Mean-13">13</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Mean-8">8</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Mean-6">6</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Mean-1">1</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Min">Min</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Min-13">13</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Min-12">12</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Min-8">8</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Min-6">6</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Min-1">1</a>|✅|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Mod">Mod</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Mod-13">13</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Mod-10">10</a>|✅|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Mul">Mul</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Mul-14">14</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Mul-13">13</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Mul-7">7</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Mul-6">6</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Mul-1">1</a>|✅|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Multinomial">Multinomial</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Multinomial-7">7</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Neg">Neg</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Neg-13">13</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Neg-6">6</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Neg-1">1</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#NonMaxSuppression">NonMaxSuppression</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#NonMaxSuppression-11">11</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#NonMaxSuppression-10">10</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#NonZero">NonZero</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#NonZero-13">13</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#NonZero-9">9</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Not">Not</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Not-1">1</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#OneHot">OneHot</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#OneHot-11">11</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#OneHot-9">9</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Optional">Optional</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Optional-15">15</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#OptionalGetElement">OptionalGetElement</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#OptionalGetElement-15">15</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#OptionalHasElement">OptionalHasElement</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#OptionalHasElement-15">15</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Or">Or</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Or-7">7</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Or-1">1</a>|✅|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#PRelu">PRelu</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#PRelu-9">9</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#PRelu-7">7</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#PRelu-6">6</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#PRelu-1">1</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Pad">Pad</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Pad-13">13</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Pad-11">11</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Pad-2">2</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Pad-1">1</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Pow">Pow</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Pow-15">15</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Pow-13">13</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Pow-12">12</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Pow-7">7</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Pow-1">1</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#QLinearConv">QLinearConv</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#QLinearConv-10">10</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#QLinearMatMul">QLinearMatMul</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#QLinearMatMul-10">10</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#QuantizeLinear">QuantizeLinear</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#QuantizeLinear-13">13</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#QuantizeLinear-10">10</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#RNN">RNN</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#RNN-14">14</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#RNN-7">7</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#RNN-1">1</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#RandomNormal">RandomNormal</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#RandomNormal-1">1</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#RandomNormalLike">RandomNormalLike</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#RandomNormalLike-1">1</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#RandomUniform">RandomUniform</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#RandomUniform-1">1</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#RandomUniformLike">RandomUniformLike</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#RandomUniformLike-1">1</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Reciprocal">Reciprocal</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Reciprocal-13">13</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Reciprocal-6">6</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Reciprocal-1">1</a>|✅|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceL1">ReduceL1</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#ReduceL1-13">13</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#ReduceL1-11">11</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#ReduceL1-1">1</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceL2">ReduceL2</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#ReduceL2-13">13</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#ReduceL2-11">11</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#ReduceL2-1">1</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceLogSum">ReduceLogSum</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#ReduceLogSum-13">13</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#ReduceLogSum-11">11</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#ReduceLogSum-1">1</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceLogSumExp">ReduceLogSumExp</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#ReduceLogSumExp-13">13</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#ReduceLogSumExp-11">11</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#ReduceLogSumExp-1">1</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceMax">ReduceMax</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#ReduceMax-13">13</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#ReduceMax-12">12</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#ReduceMax-11">11</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#ReduceMax-1">1</a>|✅|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceMean">ReduceMean</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#ReduceMean-13">13</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#ReduceMean-11">11</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#ReduceMean-1">1</a>|✅|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceMin">ReduceMin</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#ReduceMin-13">13</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#ReduceMin-12">12</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#ReduceMin-11">11</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#ReduceMin-1">1</a>|✅|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceProd">ReduceProd</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#ReduceProd-13">13</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#ReduceProd-11">11</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#ReduceProd-1">1</a>|✅|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceSum">ReduceSum</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#ReduceSum-13">13</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#ReduceSum-11">11</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#ReduceSum-1">1</a>|✅|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceSumSquare">ReduceSumSquare</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#ReduceSumSquare-13">13</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#ReduceSumSquare-11">11</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#ReduceSumSquare-1">1</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Relu">Relu</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Relu-14">14</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Relu-13">13</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Relu-6">6</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Relu-1">1</a>|✅|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Reshape">Reshape</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Reshape-14">14</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Reshape-13">13</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Reshape-5">5</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Reshape-1">1</a>|✅|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Resize">Resize</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Resize-13">13</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Resize-11">11</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Resize-10">10</a>|✅|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReverseSequence">ReverseSequence</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#ReverseSequence-10">10</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#RoiAlign">RoiAlign</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#RoiAlign-16">16</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#RoiAlign-10">10</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Round">Round</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Round-11">11</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Scan">Scan</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Scan-11">11</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Scan-9">9</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Scan-8">8</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Scatter">Scatter</a> (deprecated)|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Scatter-11">11</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Scatter-9">9</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#ScatterElements">ScatterElements</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#ScatterElements-16">16</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#ScatterElements-13">13</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#ScatterElements-11">11</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#ScatterND">ScatterND</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#ScatterND-16">16</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#ScatterND-13">13</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#ScatterND-11">11</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Selu">Selu</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Selu-6">6</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Selu-1">1</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#SequenceAt">SequenceAt</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#SequenceAt-11">11</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#SequenceConstruct">SequenceConstruct</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#SequenceConstruct-11">11</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#SequenceEmpty">SequenceEmpty</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#SequenceEmpty-11">11</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#SequenceErase">SequenceErase</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#SequenceErase-11">11</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#SequenceInsert">SequenceInsert</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#SequenceInsert-11">11</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#SequenceLength">SequenceLength</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#SequenceLength-11">11</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Shape">Shape</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Shape-15">15</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Shape-13">13</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Shape-1">1</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Shrink">Shrink</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Shrink-9">9</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Sigmoid">Sigmoid</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Sigmoid-13">13</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Sigmoid-6">6</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Sigmoid-1">1</a>|✅|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Sign">Sign</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Sign-13">13</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Sign-9">9</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Sin">Sin</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Sin-7">7</a>|✅|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Sinh">Sinh</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Sinh-9">9</a>|✅|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Size">Size</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Size-13">13</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Size-1">1</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Slice">Slice</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Slice-13">13</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Slice-11">11</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Slice-10">10</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Slice-1">1</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Softplus">Softplus</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Softplus-1">1</a>|✅|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Softsign">Softsign</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Softsign-1">1</a>|✅|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#SpaceToDepth">SpaceToDepth</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#SpaceToDepth-13">13</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#SpaceToDepth-1">1</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Split">Split</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Split-13">13</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Split-11">11</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Split-2">2</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Split-1">1</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#SplitToSequence">SplitToSequence</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#SplitToSequence-11">11</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Sqrt">Sqrt</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Sqrt-13">13</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Sqrt-6">6</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Sqrt-1">1</a>|✅|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Squeeze">Squeeze</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Squeeze-13">13</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Squeeze-11">11</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Squeeze-1">1</a>|✅|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#StringNormalizer">StringNormalizer</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#StringNormalizer-10">10</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Sub">Sub</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Sub-14">14</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Sub-13">13</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Sub-7">7</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Sub-6">6</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Sub-1">1</a>|✅|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Sum">Sum</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Sum-13">13</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Sum-8">8</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Sum-6">6</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Sum-1">1</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Tan">Tan</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Tan-7">7</a>|✅|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Tanh">Tanh</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Tanh-13">13</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Tanh-6">6</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Tanh-1">1</a>|✅|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#TfIdfVectorizer">TfIdfVectorizer</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#TfIdfVectorizer-9">9</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#ThresholdedRelu">ThresholdedRelu</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#ThresholdedRelu-10">10</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Tile">Tile</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Tile-13">13</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Tile-6">6</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Tile-1">1</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#TopK">TopK</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#TopK-11">11</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#TopK-10">10</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#TopK-1">1</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Transpose">Transpose</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Transpose-13">13</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Transpose-1">1</a>|✅|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Trilu">Trilu</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Trilu-14">14</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Unique">Unique</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Unique-11">11</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Unsqueeze">Unsqueeze</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Unsqueeze-13">13</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Unsqueeze-11">11</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Unsqueeze-1">1</a>|✅|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Upsample">Upsample</a> (deprecated)|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Upsample-10">10</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Upsample-9">9</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Upsample-7">7</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Where">Where</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Where-16">16</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Where-9">9</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Xor">Xor</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Xor-7">7</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Xor-1">1</a>|
|**Function**|**Since version**|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Bernoulli">Bernoulli</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Bernoulli-15">15</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#CastLike">CastLike</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#CastLike-15">15</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Celu">Celu</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Celu-12">12</a>|✅|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#DynamicQuantizeLinear">DynamicQuantizeLinear</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#DynamicQuantizeLinear-11">11</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#GreaterOrEqual">GreaterOrEqual</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#GreaterOrEqual-12">12</a>|✅|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#HardSwish">HardSwish</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#HardSwish-14">14</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#LessOrEqual">LessOrEqual</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#LessOrEqual-12">12</a>|✅|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#LogSoftmax">LogSoftmax</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#LogSoftmax-13">13</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#LogSoftmax-11">11</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#LogSoftmax-1">1</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#MeanVarianceNormalization">MeanVarianceNormalization</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#MeanVarianceNormalization-13">13</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#MeanVarianceNormalization-9">9</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#NegativeLogLikelihoodLoss">NegativeLogLikelihoodLoss</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#NegativeLogLikelihoodLoss-13">13</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#NegativeLogLikelihoodLoss-12">12</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Range">Range</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Range-11">11</a>|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#Softmax">Softmax</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Softmax-13">13</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Softmax-11">11</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Softmax-1">1</a>|✅ (axis=1)|
|<a href="https://github.com/onnx/onnx/blob/main/docs/Operators.md#SoftmaxCrossEntropyLoss">SoftmaxCrossEntropyLoss</a>|<a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#SoftmaxCrossEntropyLoss-13">13</a>, <a href="https://github.com/onnx/onnx/blob/main/docs/Changelog.md#SoftmaxCrossEntropyLoss-12">12</a>|
