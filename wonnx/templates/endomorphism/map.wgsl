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
	{% elif op_type == "Tanh" %}
		{# Tanh will produce NaNs when fed with inputs that are much larger than +10.0 or smaller than -10.0. As the output
		for these inputs converges to 1.0 and -1.0 respectively, we clamp the inputs first. #}
		let one = Vec4(Scalar(1), Scalar(1), Scalar(1), Scalar(1));
		let boundary = one * Scalar(10);
		let intermediate = max(-boundary, min(boundary, input_0.data[gidx]));
		output_0.data[gidx] = tanh(intermediate);
	{% else %}
		output_0.data[gidx] = {{ op_type | lower }}(input_0.data[gidx]);

	{% endif %}
}