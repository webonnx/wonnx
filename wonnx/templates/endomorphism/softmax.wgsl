{%- include "structs.wgsl" -%}

[[group(0), binding(0)]]
var<storage, read> input_0: Array;

[[group(0), binding(1)]]
var<storage, write> output_0: Array;

[[stage(compute), workgroup_size(1)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
	let gidx = global_id.x;

	{% if op_type == "Softmax" %}
		// Softmax = exp(input - max(input)) / sum(exp(input - max(input)))
		
		// First, determine max(input)
		// WGSL doesn't have a way to write -Infinity (https://github.com/gpuweb/gpuweb/issues/1769)
		// Therefore we use log(0) instead which returns -Infinity
		var max_element: Scalar = log(Scalar(0));
		for(var k: u32 = 0u; k < {{ i_lens[0] }}u; k = k + 1u) {
			let element = input_0.data[gidx + k];
			max_element = max(max_element, element);
		}

		// Calculate sum(exp(input - max(input)))
		var sum: Scalar = Scalar(0);
		for(var k: u32 = 0u; k < {{ i_lens[0] }}u; k = k + 1u) {
			let element = input_0.data[gidx + k];
			sum  = sum + exp(element - max_element);
		}

		// Calculate elements and write to output
		for(var k: u32 = 0u; k < {{ i_lens[0] }}u; k = k + 1u) {
			let element = input_0.data[gidx + k];
			output_0.data[gidx + k] = exp(element - max_element) / sum;
		}
	{% endif %}
}