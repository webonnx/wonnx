{%- include "structs.wgsl" -%}
[[group(0), binding(0)]]
var<storage, read> input_0: ArrayVector;

[[group(0), binding(1)]]
var<storage, write> output_0: ArrayVector;

[[stage(compute), workgroup_size({{ workgroup_size_x }})]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
	let gidx = global_id.x;
	
	{% if op_type == "Reciprocal" %}
		let one = Vec4(Scalar(1), Scalar(1), Scalar(1), Scalar(1));
		output_0.data[gidx] = one / (input_0.data[gidx]);

	{% else %}
		output_0.data[gidx] = {{ op_type | lower }}(input_0.data[gidx]);

	{% endif %}
}