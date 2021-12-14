<center><img src="header.png" alt="WONNX" width="700"/></center>

![GitHub Workflow Status](https://img.shields.io/github/workflow/status/haixuantao/wonnx/CI)
![Crates.io (latest)](https://img.shields.io/crates/dv/wonnx)
![Crates.io](https://img.shields.io/crates/l/wonnx)


Wonnx aims for running blazing Fast AI on any device.

## Supported Platforms (enabled by `wgpu`)

   API   |    Windows                    |  Linux & Android   |    macOS & iOS     |
  -----  | ----------------------------- | ------------------ | ------------------ |
  Vulkan | ✅                            | ✅                 |                    |
  Metal  |                               |                    | ✅                 |
  DX12   | ✅                 (W10 only) |                    |                    |
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

## Running a model from scratch

- To run an onnx model, first simplify it with [onnx-simplifier](https://github.com/daquexian/onnx-simplifier), with the command:

```bash
# pip install -U pip && pip install onnx-simplifier
python -m onnxsim mnist-8.onnx opt-mnist.onnx
```

- Then you can run it following the example in the examples folder:

```bash
cargo run --example mnist --release
```

## Usage in rust

```rust
fn main() -> HashMap<String, Vec<f32>> {
    let mut input_data = HashMap::new();
    let image = load_squeezenet_image(); // Load image
    input_data.insert("data".to_string(), image.as_slice().unwrap());

    let session = pollster::block_on(wonnx::Session::from_path(
        "examples/data/models/opt-squeeze.onnx",
    ))
    .expect("session did not create");
    let result = pollster::block_on(session.run(input_data)).unwrap();
    let result = result["squeezenet0_flatten0_reshape0"];
    let mut probabilities = result.iter().enumerate().collect::<Vec<_>>();

    probabilities.sort_unstable_by(|a, b| b.1.partial_cmp(a.1).unwrap());

    assert_eq!(probabilities[0].0, 22);
}
```

> Examples are available in the `examples` folder


## Tested Model

- Squeezenet
- MNIST


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
5. Respect the binding layout that each entry is incremented by 1 starting from 0, with input first and output last. If the number of binding is above 4. Increment the binding group.
6. Write the logic.

There is default variables in the context: 
- `{{ i_len_0 }}` the length of the input 0. This also work for output: `{{ o_len_0 }}` and other input `{{ i_len_1 }}`
- `{{ i_dims_0 }}` an array of the dimensions of input 0.

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


## Supported Operators (ref [ONNX IR](https://github.com/onnx/onnx/blob/master/docs/Operators.md?plain=1)) 

|**Operator**|**Since version**|**Implemented**|
|-|-|-|
|<a href="#Abs">Abs</a>|<a href="Changelog.md#Abs-13">13</a>, <a href="Changelog.md#Abs-6">6</a>, <a href="Changelog.md#Abs-1">1</a>|✅|
|<a href="#Acos">Acos</a>|<a href="Changelog.md#Acos-7">7</a>|✅|
|<a href="#Acosh">Acosh</a>|<a href="Changelog.md#Acosh-9">9</a>|
|<a href="#Add">Add</a>|<a href="Changelog.md#Add-14">14</a>, <a href="Changelog.md#Add-13">13</a>, <a href="Changelog.md#Add-7">7</a>, <a href="Changelog.md#Add-6">6</a>, <a href="Changelog.md#Add-1">1</a>|✅|
|<a href="#And">And</a>|<a href="Changelog.md#And-7">7</a>, <a href="Changelog.md#And-1">1</a>|✅|
|<a href="#ArgMax">ArgMax</a>|<a href="Changelog.md#ArgMax-13">13</a>, <a href="Changelog.md#ArgMax-12">12</a>, <a href="Changelog.md#ArgMax-11">11</a>, <a href="Changelog.md#ArgMax-1">1</a>|
|<a href="#ArgMin">ArgMin</a>|<a href="Changelog.md#ArgMin-13">13</a>, <a href="Changelog.md#ArgMin-12">12</a>, <a href="Changelog.md#ArgMin-11">11</a>, <a href="Changelog.md#ArgMin-1">1</a>|
|<a href="#Asin">Asin</a>|<a href="Changelog.md#Asin-7">7</a>|✅|
|<a href="#Asinh">Asinh</a>|<a href="Changelog.md#Asinh-9">9</a>|
|<a href="#Atan">Atan</a>|<a href="Changelog.md#Atan-7">7</a>|✅|
|<a href="#Atanh">Atanh</a>|<a href="Changelog.md#Atanh-9">9</a>|
|<a href="#AveragePool">AveragePool</a>|<a href="Changelog.md#AveragePool-11">11</a>, <a href="Changelog.md#AveragePool-10">10</a>, <a href="Changelog.md#AveragePool-7">7</a>, <a href="Changelog.md#AveragePool-1">1</a>|✅|
|<a href="#BatchNormalization">BatchNormalization</a>|<a href="Changelog.md#BatchNormalization-15">15</a>, <a href="Changelog.md#BatchNormalization-14">14</a>, <a href="Changelog.md#BatchNormalization-9">9</a>, <a href="Changelog.md#BatchNormalization-7">7</a>, <a href="Changelog.md#BatchNormalization-6">6</a>, <a href="Changelog.md#BatchNormalization-1">1</a>|
|<a href="#BitShift">BitShift</a>|<a href="Changelog.md#BitShift-11">11</a>|
|<a href="#Cast">Cast</a>|<a href="Changelog.md#Cast-13">13</a>, <a href="Changelog.md#Cast-9">9</a>, <a href="Changelog.md#Cast-6">6</a>, <a href="Changelog.md#Cast-1">1</a>|
|<a href="#Ceil">Ceil</a>|<a href="Changelog.md#Ceil-13">13</a>, <a href="Changelog.md#Ceil-6">6</a>, <a href="Changelog.md#Ceil-1">1</a>|✅|
|<a href="#Clip">Clip</a>|<a href="Changelog.md#Clip-13">13</a>, <a href="Changelog.md#Clip-12">12</a>, <a href="Changelog.md#Clip-11">11</a>, <a href="Changelog.md#Clip-6">6</a>, <a href="Changelog.md#Clip-1">1</a>|✅|
|<a href="#Compress">Compress</a>|<a href="Changelog.md#Compress-11">11</a>, <a href="Changelog.md#Compress-9">9</a>|
|<a href="#Concat">Concat</a>|<a href="Changelog.md#Concat-13">13</a>, <a href="Changelog.md#Concat-11">11</a>, <a href="Changelog.md#Concat-4">4</a>, <a href="Changelog.md#Concat-1">1</a>|✅|
|<a href="#ConcatFromSequence">ConcatFromSequence</a>|<a href="Changelog.md#ConcatFromSequence-11">11</a>|
|<a href="#Constant">Constant</a>|<a href="Changelog.md#Constant-13">13</a>, <a href="Changelog.md#Constant-12">12</a>, <a href="Changelog.md#Constant-11">11</a>, <a href="Changelog.md#Constant-9">9</a>, <a href="Changelog.md#Constant-1">1</a>|
|<a href="#ConstantOfShape">ConstantOfShape</a>|<a href="Changelog.md#ConstantOfShape-9">9</a>|
|<a href="#Conv">Conv</a>|<a href="Changelog.md#Conv-11">11</a>, <a href="Changelog.md#Conv-1">1</a>|✅|
|<a href="#ConvInteger">ConvInteger</a>|<a href="Changelog.md#ConvInteger-10">10</a>|
|<a href="#ConvTranspose">ConvTranspose</a>|<a href="Changelog.md#ConvTranspose-11">11</a>, <a href="Changelog.md#ConvTranspose-1">1</a>|
|<a href="#Cos">Cos</a>|<a href="Changelog.md#Cos-7">7</a>|✅|
|<a href="#Cosh">Cosh</a>|<a href="Changelog.md#Cosh-9">9</a>|✅|
|<a href="#CumSum">CumSum</a>|<a href="Changelog.md#CumSum-14">14</a>, <a href="Changelog.md#CumSum-11">11</a>|
|<a href="#DepthToSpace">DepthToSpace</a>|<a href="Changelog.md#DepthToSpace-13">13</a>, <a href="Changelog.md#DepthToSpace-11">11</a>, <a href="Changelog.md#DepthToSpace-1">1</a>|
|<a href="#DequantizeLinear">DequantizeLinear</a>|<a href="Changelog.md#DequantizeLinear-13">13</a>, <a href="Changelog.md#DequantizeLinear-10">10</a>|
|<a href="#Det">Det</a>|<a href="Changelog.md#Det-11">11</a>|
|<a href="#Div">Div</a>|<a href="Changelog.md#Div-14">14</a>, <a href="Changelog.md#Div-13">13</a>, <a href="Changelog.md#Div-7">7</a>, <a href="Changelog.md#Div-6">6</a>, <a href="Changelog.md#Div-1">1</a>|✅|
|<a href="#Dropout">Dropout</a>|<a href="Changelog.md#Dropout-13">13</a>, <a href="Changelog.md#Dropout-12">12</a>, <a href="Changelog.md#Dropout-10">10</a>, <a href="Changelog.md#Dropout-7">7</a>, <a href="Changelog.md#Dropout-6">6</a>, <a href="Changelog.md#Dropout-1">1</a>|✅|
|<a href="#Einsum">Einsum</a>|<a href="Changelog.md#Einsum-12">12</a>|
|<a href="#Elu">Elu</a>|<a href="Changelog.md#Elu-6">6</a>, <a href="Changelog.md#Elu-1">1</a>|✅|
|<a href="#Equal">Equal</a>|<a href="Changelog.md#Equal-13">13</a>, <a href="Changelog.md#Equal-11">11</a>, <a href="Changelog.md#Equal-7">7</a>, <a href="Changelog.md#Equal-1">1</a>|✅|
|<a href="#Erf">Erf</a>|<a href="Changelog.md#Erf-13">13</a>, <a href="Changelog.md#Erf-9">9</a>|
|<a href="#Exp">Exp</a>|<a href="Changelog.md#Exp-13">13</a>, <a href="Changelog.md#Exp-6">6</a>, <a href="Changelog.md#Exp-1">1</a>|✅|
|<a href="#Expand">Expand</a>|<a href="Changelog.md#Expand-13">13</a>, <a href="Changelog.md#Expand-8">8</a>|
|<a href="#EyeLike">EyeLike</a>|<a href="Changelog.md#EyeLike-9">9</a>|
|<a href="#Flatten">Flatten</a>|<a href="Changelog.md#Flatten-13">13</a>, <a href="Changelog.md#Flatten-11">11</a>, <a href="Changelog.md#Flatten-9">9</a>, <a href="Changelog.md#Flatten-1">1</a>|✅|
|<a href="#Floor">Floor</a>|<a href="Changelog.md#Floor-13">13</a>, <a href="Changelog.md#Floor-6">6</a>, <a href="Changelog.md#Floor-1">1</a>|✅|
|<a href="#GRU">GRU</a>|<a href="Changelog.md#GRU-14">14</a>, <a href="Changelog.md#GRU-7">7</a>, <a href="Changelog.md#GRU-3">3</a>, <a href="Changelog.md#GRU-1">1</a>|
|<a href="#Gather">Gather</a>|<a href="Changelog.md#Gather-13">13</a>, <a href="Changelog.md#Gather-11">11</a>, <a href="Changelog.md#Gather-1">1</a>|
|<a href="#GatherElements">GatherElements</a>|<a href="Changelog.md#GatherElements-13">13</a>, <a href="Changelog.md#GatherElements-11">11</a>|
|<a href="#GatherND">GatherND</a>|<a href="Changelog.md#GatherND-13">13</a>, <a href="Changelog.md#GatherND-12">12</a>, <a href="Changelog.md#GatherND-11">11</a>|
|<a href="#Gemm">Gemm</a>|<a href="Changelog.md#Gemm-13">13</a>, <a href="Changelog.md#Gemm-11">11</a>, <a href="Changelog.md#Gemm-9">9</a>, <a href="Changelog.md#Gemm-7">7</a>, <a href="Changelog.md#Gemm-6">6</a>, <a href="Changelog.md#Gemm-1">1</a>|✅|
|<a href="#GlobalAveragePool">GlobalAveragePool</a>|<a href="Changelog.md#GlobalAveragePool-1">1</a>|
|<a href="#GlobalLpPool">GlobalLpPool</a>|<a href="Changelog.md#GlobalLpPool-2">2</a>, <a href="Changelog.md#GlobalLpPool-1">1</a>|
|<a href="#GlobalMaxPool">GlobalMaxPool</a>|<a href="Changelog.md#GlobalMaxPool-1">1</a>|
|<a href="#Greater">Greater</a>|<a href="Changelog.md#Greater-13">13</a>, <a href="Changelog.md#Greater-9">9</a>, <a href="Changelog.md#Greater-7">7</a>, <a href="Changelog.md#Greater-1">1</a>|✅|
|<a href="#GridSample">GridSample</a>|<a href="Changelog.md#GridSample-16">16</a>|
|<a href="#HardSigmoid">HardSigmoid</a>|<a href="Changelog.md#HardSigmoid-6">6</a>, <a href="Changelog.md#HardSigmoid-1">1</a>|
|<a href="#Hardmax">Hardmax</a>|<a href="Changelog.md#Hardmax-13">13</a>, <a href="Changelog.md#Hardmax-11">11</a>, <a href="Changelog.md#Hardmax-1">1</a>|
|<a href="#Identity">Identity</a>|<a href="Changelog.md#Identity-16">16</a>, <a href="Changelog.md#Identity-14">14</a>, <a href="Changelog.md#Identity-13">13</a>, <a href="Changelog.md#Identity-1">1</a>|
|<a href="#If">If</a>|<a href="Changelog.md#If-16">16</a>, <a href="Changelog.md#If-13">13</a>, <a href="Changelog.md#If-11">11</a>, <a href="Changelog.md#If-1">1</a>|
|<a href="#InstanceNormalization">InstanceNormalization</a>|<a href="Changelog.md#InstanceNormalization-6">6</a>, <a href="Changelog.md#InstanceNormalization-1">1</a>|
|<a href="#IsInf">IsInf</a>|<a href="Changelog.md#IsInf-10">10</a>|
|<a href="#IsNaN">IsNaN</a>|<a href="Changelog.md#IsNaN-13">13</a>, <a href="Changelog.md#IsNaN-9">9</a>|
|<a href="#LRN">LRN</a>|<a href="Changelog.md#LRN-13">13</a>, <a href="Changelog.md#LRN-1">1</a>|
|<a href="#LSTM">LSTM</a>|<a href="Changelog.md#LSTM-14">14</a>, <a href="Changelog.md#LSTM-7">7</a>, <a href="Changelog.md#LSTM-1">1</a>|
|<a href="#LeakyRelu">LeakyRelu</a>|<a href="Changelog.md#LeakyRelu-6">6</a>, <a href="Changelog.md#LeakyRelu-1">1</a>|
|<a href="#Less">Less</a>|<a href="Changelog.md#Less-13">13</a>, <a href="Changelog.md#Less-9">9</a>, <a href="Changelog.md#Less-7">7</a>, <a href="Changelog.md#Less-1">1</a>|✅|
|<a href="#Log">Log</a>|<a href="Changelog.md#Log-13">13</a>, <a href="Changelog.md#Log-6">6</a>, <a href="Changelog.md#Log-1">1</a>|✅|
|<a href="#Loop">Loop</a>|<a href="Changelog.md#Loop-16">16</a>, <a href="Changelog.md#Loop-13">13</a>, <a href="Changelog.md#Loop-11">11</a>, <a href="Changelog.md#Loop-1">1</a>|
|<a href="#LpNormalization">LpNormalization</a>|<a href="Changelog.md#LpNormalization-1">1</a>|
|<a href="#LpPool">LpPool</a>|<a href="Changelog.md#LpPool-11">11</a>, <a href="Changelog.md#LpPool-2">2</a>, <a href="Changelog.md#LpPool-1">1</a>|
|<a href="#MatMul">MatMul</a>|<a href="Changelog.md#MatMul-13">13</a>, <a href="Changelog.md#MatMul-9">9</a>, <a href="Changelog.md#MatMul-1">1</a>|✅|
|<a href="#MatMulInteger">MatMulInteger</a>|<a href="Changelog.md#MatMulInteger-10">10</a>|
|<a href="#Max">Max</a>|<a href="Changelog.md#Max-13">13</a>, <a href="Changelog.md#Max-12">12</a>, <a href="Changelog.md#Max-8">8</a>, <a href="Changelog.md#Max-6">6</a>, <a href="Changelog.md#Max-1">1</a>|
|<a href="#MaxPool">MaxPool</a>|<a href="Changelog.md#MaxPool-12">12</a>, <a href="Changelog.md#MaxPool-11">11</a>, <a href="Changelog.md#MaxPool-10">10</a>, <a href="Changelog.md#MaxPool-8">8</a>, <a href="Changelog.md#MaxPool-1">1</a>|✅|
|<a href="#MaxRoiPool">MaxRoiPool</a>|<a href="Changelog.md#MaxRoiPool-1">1</a>|
|<a href="#MaxUnpool">MaxUnpool</a>|<a href="Changelog.md#MaxUnpool-11">11</a>, <a href="Changelog.md#MaxUnpool-9">9</a>|
|<a href="#Mean">Mean</a>|<a href="Changelog.md#Mean-13">13</a>, <a href="Changelog.md#Mean-8">8</a>, <a href="Changelog.md#Mean-6">6</a>, <a href="Changelog.md#Mean-1">1</a>|
|<a href="#Min">Min</a>|<a href="Changelog.md#Min-13">13</a>, <a href="Changelog.md#Min-12">12</a>, <a href="Changelog.md#Min-8">8</a>, <a href="Changelog.md#Min-6">6</a>, <a href="Changelog.md#Min-1">1</a>|✅|
|<a href="#Mod">Mod</a>|<a href="Changelog.md#Mod-13">13</a>, <a href="Changelog.md#Mod-10">10</a>|✅|
|<a href="#Mul">Mul</a>|<a href="Changelog.md#Mul-14">14</a>, <a href="Changelog.md#Mul-13">13</a>, <a href="Changelog.md#Mul-7">7</a>, <a href="Changelog.md#Mul-6">6</a>, <a href="Changelog.md#Mul-1">1</a>|✅|
|<a href="#Multinomial">Multinomial</a>|<a href="Changelog.md#Multinomial-7">7</a>|
|<a href="#Neg">Neg</a>|<a href="Changelog.md#Neg-13">13</a>, <a href="Changelog.md#Neg-6">6</a>, <a href="Changelog.md#Neg-1">1</a>|
|<a href="#NonMaxSuppression">NonMaxSuppression</a>|<a href="Changelog.md#NonMaxSuppression-11">11</a>, <a href="Changelog.md#NonMaxSuppression-10">10</a>|
|<a href="#NonZero">NonZero</a>|<a href="Changelog.md#NonZero-13">13</a>, <a href="Changelog.md#NonZero-9">9</a>|
|<a href="#Not">Not</a>|<a href="Changelog.md#Not-1">1</a>|
|<a href="#OneHot">OneHot</a>|<a href="Changelog.md#OneHot-11">11</a>, <a href="Changelog.md#OneHot-9">9</a>|
|<a href="#Optional">Optional</a>|<a href="Changelog.md#Optional-15">15</a>|
|<a href="#OptionalGetElement">OptionalGetElement</a>|<a href="Changelog.md#OptionalGetElement-15">15</a>|
|<a href="#OptionalHasElement">OptionalHasElement</a>|<a href="Changelog.md#OptionalHasElement-15">15</a>|
|<a href="#Or">Or</a>|<a href="Changelog.md#Or-7">7</a>, <a href="Changelog.md#Or-1">1</a>|✅|
|<a href="#PRelu">PRelu</a>|<a href="Changelog.md#PRelu-9">9</a>, <a href="Changelog.md#PRelu-7">7</a>, <a href="Changelog.md#PRelu-6">6</a>, <a href="Changelog.md#PRelu-1">1</a>|
|<a href="#Pad">Pad</a>|<a href="Changelog.md#Pad-13">13</a>, <a href="Changelog.md#Pad-11">11</a>, <a href="Changelog.md#Pad-2">2</a>, <a href="Changelog.md#Pad-1">1</a>|
|<a href="#Pow">Pow</a>|<a href="Changelog.md#Pow-15">15</a>, <a href="Changelog.md#Pow-13">13</a>, <a href="Changelog.md#Pow-12">12</a>, <a href="Changelog.md#Pow-7">7</a>, <a href="Changelog.md#Pow-1">1</a>|
|<a href="#QLinearConv">QLinearConv</a>|<a href="Changelog.md#QLinearConv-10">10</a>|
|<a href="#QLinearMatMul">QLinearMatMul</a>|<a href="Changelog.md#QLinearMatMul-10">10</a>|
|<a href="#QuantizeLinear">QuantizeLinear</a>|<a href="Changelog.md#QuantizeLinear-13">13</a>, <a href="Changelog.md#QuantizeLinear-10">10</a>|
|<a href="#RNN">RNN</a>|<a href="Changelog.md#RNN-14">14</a>, <a href="Changelog.md#RNN-7">7</a>, <a href="Changelog.md#RNN-1">1</a>|
|<a href="#RandomNormal">RandomNormal</a>|<a href="Changelog.md#RandomNormal-1">1</a>|
|<a href="#RandomNormalLike">RandomNormalLike</a>|<a href="Changelog.md#RandomNormalLike-1">1</a>|
|<a href="#RandomUniform">RandomUniform</a>|<a href="Changelog.md#RandomUniform-1">1</a>|
|<a href="#RandomUniformLike">RandomUniformLike</a>|<a href="Changelog.md#RandomUniformLike-1">1</a>|
|<a href="#Reciprocal">Reciprocal</a>|<a href="Changelog.md#Reciprocal-13">13</a>, <a href="Changelog.md#Reciprocal-6">6</a>, <a href="Changelog.md#Reciprocal-1">1</a>|
|<a href="#ReduceL1">ReduceL1</a>|<a href="Changelog.md#ReduceL1-13">13</a>, <a href="Changelog.md#ReduceL1-11">11</a>, <a href="Changelog.md#ReduceL1-1">1</a>|
|<a href="#ReduceL2">ReduceL2</a>|<a href="Changelog.md#ReduceL2-13">13</a>, <a href="Changelog.md#ReduceL2-11">11</a>, <a href="Changelog.md#ReduceL2-1">1</a>|
|<a href="#ReduceLogSum">ReduceLogSum</a>|<a href="Changelog.md#ReduceLogSum-13">13</a>, <a href="Changelog.md#ReduceLogSum-11">11</a>, <a href="Changelog.md#ReduceLogSum-1">1</a>|
|<a href="#ReduceLogSumExp">ReduceLogSumExp</a>|<a href="Changelog.md#ReduceLogSumExp-13">13</a>, <a href="Changelog.md#ReduceLogSumExp-11">11</a>, <a href="Changelog.md#ReduceLogSumExp-1">1</a>|
|<a href="#ReduceMax">ReduceMax</a>|<a href="Changelog.md#ReduceMax-13">13</a>, <a href="Changelog.md#ReduceMax-12">12</a>, <a href="Changelog.md#ReduceMax-11">11</a>, <a href="Changelog.md#ReduceMax-1">1</a>|
|<a href="#ReduceMean">ReduceMean</a>|<a href="Changelog.md#ReduceMean-13">13</a>, <a href="Changelog.md#ReduceMean-11">11</a>, <a href="Changelog.md#ReduceMean-1">1</a>|
|<a href="#ReduceMin">ReduceMin</a>|<a href="Changelog.md#ReduceMin-13">13</a>, <a href="Changelog.md#ReduceMin-12">12</a>, <a href="Changelog.md#ReduceMin-11">11</a>, <a href="Changelog.md#ReduceMin-1">1</a>|
|<a href="#ReduceProd">ReduceProd</a>|<a href="Changelog.md#ReduceProd-13">13</a>, <a href="Changelog.md#ReduceProd-11">11</a>, <a href="Changelog.md#ReduceProd-1">1</a>|
|<a href="#ReduceSum">ReduceSum</a>|<a href="Changelog.md#ReduceSum-13">13</a>, <a href="Changelog.md#ReduceSum-11">11</a>, <a href="Changelog.md#ReduceSum-1">1</a>|
|<a href="#ReduceSumSquare">ReduceSumSquare</a>|<a href="Changelog.md#ReduceSumSquare-13">13</a>, <a href="Changelog.md#ReduceSumSquare-11">11</a>, <a href="Changelog.md#ReduceSumSquare-1">1</a>|
|<a href="#Relu">Relu</a>|<a href="Changelog.md#Relu-14">14</a>, <a href="Changelog.md#Relu-13">13</a>, <a href="Changelog.md#Relu-6">6</a>, <a href="Changelog.md#Relu-1">1</a>|✅|
|<a href="#Reshape">Reshape</a>|<a href="Changelog.md#Reshape-14">14</a>, <a href="Changelog.md#Reshape-13">13</a>, <a href="Changelog.md#Reshape-5">5</a>, <a href="Changelog.md#Reshape-1">1</a>|✅|
|<a href="#Resize">Resize</a>|<a href="Changelog.md#Resize-13">13</a>, <a href="Changelog.md#Resize-11">11</a>, <a href="Changelog.md#Resize-10">10</a>|✅|
|<a href="#ReverseSequence">ReverseSequence</a>|<a href="Changelog.md#ReverseSequence-10">10</a>|
|<a href="#RoiAlign">RoiAlign</a>|<a href="Changelog.md#RoiAlign-16">16</a>, <a href="Changelog.md#RoiAlign-10">10</a>|
|<a href="#Round">Round</a>|<a href="Changelog.md#Round-11">11</a>|
|<a href="#Scan">Scan</a>|<a href="Changelog.md#Scan-11">11</a>, <a href="Changelog.md#Scan-9">9</a>, <a href="Changelog.md#Scan-8">8</a>|
|<a href="#Scatter">Scatter</a> (deprecated)|<a href="Changelog.md#Scatter-11">11</a>, <a href="Changelog.md#Scatter-9">9</a>|
|<a href="#ScatterElements">ScatterElements</a>|<a href="Changelog.md#ScatterElements-16">16</a>, <a href="Changelog.md#ScatterElements-13">13</a>, <a href="Changelog.md#ScatterElements-11">11</a>|
|<a href="#ScatterND">ScatterND</a>|<a href="Changelog.md#ScatterND-16">16</a>, <a href="Changelog.md#ScatterND-13">13</a>, <a href="Changelog.md#ScatterND-11">11</a>|
|<a href="#Selu">Selu</a>|<a href="Changelog.md#Selu-6">6</a>, <a href="Changelog.md#Selu-1">1</a>|
|<a href="#SequenceAt">SequenceAt</a>|<a href="Changelog.md#SequenceAt-11">11</a>|
|<a href="#SequenceConstruct">SequenceConstruct</a>|<a href="Changelog.md#SequenceConstruct-11">11</a>|
|<a href="#SequenceEmpty">SequenceEmpty</a>|<a href="Changelog.md#SequenceEmpty-11">11</a>|
|<a href="#SequenceErase">SequenceErase</a>|<a href="Changelog.md#SequenceErase-11">11</a>|
|<a href="#SequenceInsert">SequenceInsert</a>|<a href="Changelog.md#SequenceInsert-11">11</a>|
|<a href="#SequenceLength">SequenceLength</a>|<a href="Changelog.md#SequenceLength-11">11</a>|
|<a href="#Shape">Shape</a>|<a href="Changelog.md#Shape-15">15</a>, <a href="Changelog.md#Shape-13">13</a>, <a href="Changelog.md#Shape-1">1</a>|
|<a href="#Shrink">Shrink</a>|<a href="Changelog.md#Shrink-9">9</a>|
|<a href="#Sigmoid">Sigmoid</a>|<a href="Changelog.md#Sigmoid-13">13</a>, <a href="Changelog.md#Sigmoid-6">6</a>, <a href="Changelog.md#Sigmoid-1">1</a>|✅|
|<a href="#Sign">Sign</a>|<a href="Changelog.md#Sign-13">13</a>, <a href="Changelog.md#Sign-9">9</a>|
|<a href="#Sin">Sin</a>|<a href="Changelog.md#Sin-7">7</a>|✅|
|<a href="#Sinh">Sinh</a>|<a href="Changelog.md#Sinh-9">9</a>|✅|
|<a href="#Size">Size</a>|<a href="Changelog.md#Size-13">13</a>, <a href="Changelog.md#Size-1">1</a>|
|<a href="#Slice">Slice</a>|<a href="Changelog.md#Slice-13">13</a>, <a href="Changelog.md#Slice-11">11</a>, <a href="Changelog.md#Slice-10">10</a>, <a href="Changelog.md#Slice-1">1</a>|
|<a href="#Softplus">Softplus</a>|<a href="Changelog.md#Softplus-1">1</a>|✅|
|<a href="#Softsign">Softsign</a>|<a href="Changelog.md#Softsign-1">1</a>|✅|
|<a href="#SpaceToDepth">SpaceToDepth</a>|<a href="Changelog.md#SpaceToDepth-13">13</a>, <a href="Changelog.md#SpaceToDepth-1">1</a>|
|<a href="#Split">Split</a>|<a href="Changelog.md#Split-13">13</a>, <a href="Changelog.md#Split-11">11</a>, <a href="Changelog.md#Split-2">2</a>, <a href="Changelog.md#Split-1">1</a>|
|<a href="#SplitToSequence">SplitToSequence</a>|<a href="Changelog.md#SplitToSequence-11">11</a>|
|<a href="#Sqrt">Sqrt</a>|<a href="Changelog.md#Sqrt-13">13</a>, <a href="Changelog.md#Sqrt-6">6</a>, <a href="Changelog.md#Sqrt-1">1</a>|✅|
|<a href="#Squeeze">Squeeze</a>|<a href="Changelog.md#Squeeze-13">13</a>, <a href="Changelog.md#Squeeze-11">11</a>, <a href="Changelog.md#Squeeze-1">1</a>|✅|
|<a href="#StringNormalizer">StringNormalizer</a>|<a href="Changelog.md#StringNormalizer-10">10</a>|
|<a href="#Sub">Sub</a>|<a href="Changelog.md#Sub-14">14</a>, <a href="Changelog.md#Sub-13">13</a>, <a href="Changelog.md#Sub-7">7</a>, <a href="Changelog.md#Sub-6">6</a>, <a href="Changelog.md#Sub-1">1</a>|✅|
|<a href="#Sum">Sum</a>|<a href="Changelog.md#Sum-13">13</a>, <a href="Changelog.md#Sum-8">8</a>, <a href="Changelog.md#Sum-6">6</a>, <a href="Changelog.md#Sum-1">1</a>|
|<a href="#Tan">Tan</a>|<a href="Changelog.md#Tan-7">7</a>|✅|
|<a href="#Tanh">Tanh</a>|<a href="Changelog.md#Tanh-13">13</a>, <a href="Changelog.md#Tanh-6">6</a>, <a href="Changelog.md#Tanh-1">1</a>|✅|
|<a href="#TfIdfVectorizer">TfIdfVectorizer</a>|<a href="Changelog.md#TfIdfVectorizer-9">9</a>|
|<a href="#ThresholdedRelu">ThresholdedRelu</a>|<a href="Changelog.md#ThresholdedRelu-10">10</a>|
|<a href="#Tile">Tile</a>|<a href="Changelog.md#Tile-13">13</a>, <a href="Changelog.md#Tile-6">6</a>, <a href="Changelog.md#Tile-1">1</a>|
|<a href="#TopK">TopK</a>|<a href="Changelog.md#TopK-11">11</a>, <a href="Changelog.md#TopK-10">10</a>, <a href="Changelog.md#TopK-1">1</a>|
|<a href="#Transpose">Transpose</a>|<a href="Changelog.md#Transpose-13">13</a>, <a href="Changelog.md#Transpose-1">1</a>|✅|
|<a href="#Trilu">Trilu</a>|<a href="Changelog.md#Trilu-14">14</a>|
|<a href="#Unique">Unique</a>|<a href="Changelog.md#Unique-11">11</a>|
|<a href="#Unsqueeze">Unsqueeze</a>|<a href="Changelog.md#Unsqueeze-13">13</a>, <a href="Changelog.md#Unsqueeze-11">11</a>, <a href="Changelog.md#Unsqueeze-1">1</a>|
|<a href="#Upsample">Upsample</a> (deprecated)|<a href="Changelog.md#Upsample-10">10</a>, <a href="Changelog.md#Upsample-9">9</a>, <a href="Changelog.md#Upsample-7">7</a>|
|<a href="#Where">Where</a>|<a href="Changelog.md#Where-16">16</a>, <a href="Changelog.md#Where-9">9</a>|
|<a href="#Xor">Xor</a>|<a href="Changelog.md#Xor-7">7</a>, <a href="Changelog.md#Xor-1">1</a>|
|**Function**|**Since version**|
|<a href="#Bernoulli">Bernoulli</a>|<a href="Changelog.md#Bernoulli-15">15</a>|
|<a href="#CastLike">CastLike</a>|<a href="Changelog.md#CastLike-15">15</a>|
|<a href="#Celu">Celu</a>|<a href="Changelog.md#Celu-12">12</a>|✅|
|<a href="#DynamicQuantizeLinear">DynamicQuantizeLinear</a>|<a href="Changelog.md#DynamicQuantizeLinear-11">11</a>|
|<a href="#GreaterOrEqual">GreaterOrEqual</a>|<a href="Changelog.md#GreaterOrEqual-12">12</a>|✅|
|<a href="#HardSwish">HardSwish</a>|<a href="Changelog.md#HardSwish-14">14</a>|
|<a href="#LessOrEqual">LessOrEqual</a>|<a href="Changelog.md#LessOrEqual-12">12</a>|✅|
|<a href="#LogSoftmax">LogSoftmax</a>|<a href="Changelog.md#LogSoftmax-13">13</a>, <a href="Changelog.md#LogSoftmax-11">11</a>, <a href="Changelog.md#LogSoftmax-1">1</a>|
|<a href="#MeanVarianceNormalization">MeanVarianceNormalization</a>|<a href="Changelog.md#MeanVarianceNormalization-13">13</a>, <a href="Changelog.md#MeanVarianceNormalization-9">9</a>|
|<a href="#NegativeLogLikelihoodLoss">NegativeLogLikelihoodLoss</a>|<a href="Changelog.md#NegativeLogLikelihoodLoss-13">13</a>, <a href="Changelog.md#NegativeLogLikelihoodLoss-12">12</a>|
|<a href="#Range">Range</a>|<a href="Changelog.md#Range-11">11</a>|
|<a href="#Softmax">Softmax</a>|<a href="Changelog.md#Softmax-13">13</a>, <a href="Changelog.md#Softmax-11">11</a>, <a href="Changelog.md#Softmax-1">1</a>|✅|
|<a href="#SoftmaxCrossEntropyLoss">SoftmaxCrossEntropyLoss</a>|<a href="Changelog.md#SoftmaxCrossEntropyLoss-13">13</a>, <a href="Changelog.md#SoftmaxCrossEntropyLoss-12">12</a>|