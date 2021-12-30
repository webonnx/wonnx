
{%- include "structs.wgsl" -%}

[[group(0), binding(0)]]
var<storage, read> {{ inputs[0] }}: Array;

[[group(0), binding(1)]]
var<storage, write> {{ outputs[0] }}: Array;


[[stage(compute), workgroup_size(1)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
	let gidx = global_id.x;

	{% if op_type == "Softmax" %}

	var max_element: f32 = -9999999.9;
	for(var k: u32 = 0u; k < {{ i_lens[0] }}u; k = k + 1u) {
		let element = {{ inputs[0] }}.data[gidx + k];
		max_element = max(max_element, element);
	}

	 var sum: f32 = 0.0;
	 for(var k: u32 = 0u; k < {{ i_lens[0] }}u; k = k + 1u) {
		let element = {{ inputs[0] }}.data[gidx + k];
		sum  = sum + exp(element - max_element);
	}

	for(var k: u32 = 0u; k < {{ i_lens[0] }}u; k = k + 1u) {
		let element = {{ inputs[0] }}.data[gidx + k];
		{{ outputs[0] }}.data[gidx + k] = exp(element - max_element) / sum;
	}
	{% endif %}
}