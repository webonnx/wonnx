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

## Implementated Operator(from [ONNX IR](https://github.com/onnx/onnx/blob/master/docs/Operators.md?plain=1)) 

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
|<a href="#RandomUniform">Rand