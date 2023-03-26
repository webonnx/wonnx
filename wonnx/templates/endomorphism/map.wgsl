{%- include "structs.wgsl" -%}
@group(0) @binding(0)
var<storage, read> input_0: ArrayVector;

@group(0) @binding(1)
var<storage, read_write> output_0: ArrayVector;

@compute @workgroup_size({{ workgroup_size_x }})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
	let gidx = global_id.x;
	
	{% if op_type == "Reciprocal" %}
		let one_scalar = {{scalar_type}}(1);
		let one = Vec4(one_scalar, one_scalar, one_scalar, one_scalar);
		output_0.data[gidx] = one / (input_0.data[gidx]);
	{% elif op_type == "Neg" %}
		let zero_scalar = {{scalar_type}}(0);
		let zeroes = Vec4(zero_scalar, zero_scalar, zero_scalar, zero_scalar);
		output_0.data[gidx] = zeroes - (input_0.data[gidx]);
	{% elif op_type == "Tanh" %}
		{# Tanh will produce NaNs when fed with inputs that are much larger than +10.0 or smaller than -10.0. As the output
		for these inputs converges to 1.0 and -1.0 respectively, we clamp the inputs first. #}
		let one_scalar = {{scalar_type}}(1);
		let one = Vec4(one_scalar, one_scalar, one_scalar, one_scalar);
		let boundary = one * {{ scalar_type }}(10);
		let intermediate = max(-boundary, min(boundary, input_0.data[gidx]));
		output_0.data[gidx] = tanh(intermediate);
	{% elif op_type == "Sign" %}
		{# sign(input_0.data[gidx]) should work but for some reason fails on Windows. Therefore we implement it here the slow way... #}
		{% for i in range(end = 4) %}
			if input_0.data[gidx][{{i}}] < {{ scalar_type }}(0) {
				output_0.data[gidx][{{i}}] = {{ scalar_type }}(-1);
			}
			else if input_0.data[gidx][{{i}}] > {{ scalar_type }}(0) {
				output_0.data[gidx][{{i}}] = {{ scalar_type }}(1);
			}
			else {
				output_0.data[gidx][{{i}}] = {{ scalar_type }}(0);
			}
		{% endfor %}
	{% else %}
		output_0.data[gidx] = {{ op_type | lower }}(input_0.data[gidx]);

	{% endif %}
}