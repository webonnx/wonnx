{%- include "structs.wgsl" -%}

[[group(0), binding(0)]]
var<storage, read> input_0: ArrayVector;

{% if i_lens | length == 2 %}

[[group(0), binding(1)]]
var<storage, read> input_1: ArrayVector;

[[group(0), binding(2)]]
var<storage, write> output_0: ArrayVector;

{% else %}

[[group(0), binding(1)]]
var<storage, write> output_0: ArrayVector;

{% endif %}

[[stage(compute), workgroup_size({{ workgroup_size_x }}, 1, 1)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
	let gidx = global_id.x;

	{% if i_lens | length == 2 %}
		{% if op_type == "Pow" %}
			output_0.data[gidx] = pow(input_0.data[gidx], input_1.data[gidx]);
		{% else %}
			output_0.data[gidx] = input_0.data[gidx] {{ op_type }} input_1.data[gidx];
		{% endif %}

	{% else %}
		output_0.data[gidx] = input_0.data[gidx] {{ op_type }} Vec4(
			Scalar({{ coefficient }}), 
			Scalar({{ coefficient }}),
			Scalar({{ coefficient }}),
			Scalar({{ coefficient }})
		);
	{% endif %}
}