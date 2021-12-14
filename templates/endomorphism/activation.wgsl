{%- include "structs.wgsl" -%}

[[group(0), binding(0)]]
var<storage, read> {{ inputs[0] }}: ArrayVector;

[[group(0), binding(1)]]
var<storage, write> {{ outputs[0] }}: ArrayVector;

[[stage(compute), workgroup_size(1)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
    let gidx = global_id.x;

{% set activation_input = [ inputs[0], ".data[gidx]"] | join(sep="") %}
{% set activation_output = [ outputs[0], ".data[gidx]"] | join(sep="") %}
{% set activation_type = op_type %}
{%- include "snippets/activation_vec.wgsl" -%}

}