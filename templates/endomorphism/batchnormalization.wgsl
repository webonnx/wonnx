
{%- include "structs.wgsl" -%}

struct Block {
    data: [[stride({{ elem_stride }})]] array<{{ elem_type }}>;
}; 

// X (input)
[[group(0), binding(0)]]
var<storage, read> {{ inputs[0] }}: Block;

// Scale
[[group(0), binding(1)]]
var<storage, read> {{ inputs[1] }}: Array;

// B (bias)
[[group(0), binding(2)]]
var<storage, read> {{ inputs[2] }}: Array;

// Input mean
[[group(0), binding(3)]]
var<storage, read> {{ inputs[3] }}: Array;

// Input variance
[[group(1), binding(0)]]
var<storage, read> {{ inputs[4] }}: Array;

// Y (Output)
[[group(1), binding(1)]]
var<storage, write> {{ outputs[0] }}: Block;

[[stage(compute), workgroup_size(1)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
    let channel = global_id.y;
    let batch = global_id.z;
    let index = global_id.x + batch * {{ batch_size }}u + channel * {{ channel_size }}u;

    // Y = (X - input_mean) / sqrt(input_var + epsilon) * scale + B
    let x = {{ inputs[0] }}.data[index];
    let channel_scale = {{ inputs[1] }}.data[channel];
    let channel_bias = {{ inputs[2] }}.data[channel];
    let channel_mean = {{ inputs[3] }}.data[channel];
    let channel_var = {{ inputs[4] }}.data[channel];

    {{ outputs[0] }}.data[index] = (x - channel_mean) / sqrt(channel_var + {{ epsilon }}) * channel_scale + channel_bias;
}