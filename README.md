# wonnx

Wonnx (Web ONNX) is an ONNX runtime based on `wgpu` aimed at being a universal GPU runtime, written in Rust.

## Why ONNX?

The problem I faced, was that I was using ONNX models and could not run it on my work computer with GPU.
Microsoft DirectML, and Intel OpenVino provider did not really match my expectations, and so I needed something different.

The idea then grew to build a Rust based ONNX runtime with `wgpu` for fun and being able to run everywhere. I think that a lot of gpu acceleration are 
closed sourced largely targeting C/C++, making it difficult to build cross-platform applications without requring CUDA, ROCm, which are not fun to install.

Wonnx aimed at being:
- Cross-platform (including Web) and Hardware agnostic.
- 100% Open Source Rust ( No CUDA / ... )
- Non-Profit

## Command Encoder should be as full as possible.

To make it as complete as possible

- Use vec4 and mat4x4 as much as possible.
- Put as much instruction within one command encoder as possible.

compiling. parallelazable task.

such as multiplication + activation
therefore, some thing should be executed.
some thing should be added.

