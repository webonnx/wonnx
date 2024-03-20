{%- include "structs.wgsl" -%}

@group(0) @binding(0)
var<storage, read> input_0: ArrayVector;

@group(0) @binding(1)
var<storage, read_write> output_0: ArrayVector;

const pi: f32 = 3.1415;

@compute @workgroup_size({{ workgroup_size_x }})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
	let gidx = global_id.x;

	{% set activation_input = "input_0.data[gidx]" %}
	{% set activation_output = "output_0.data[gidx]" %}
	{% set activation_type = op_type %}
	{%- include "snippets/activation_vec.wgsl" -%}
}