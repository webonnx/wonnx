{%- include "structs.wgsl" -%}

@group(0) @binding(0)
var<storage, read> input_0: Array;

@group(0) @binding(1)
var<storage, write> output_0: Array;

@stage(compute) @workgroup_size({{ workgroup_size_x }}, {{ workgroup_size_y }})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
	let gidx = global_id.x;
	let chunk_start = gidx * {{ axis_chunk }}u + global_id.y;

	{% if opset_version < 13 %}
		let n_elements = {{ axis_chunk }}u;
		let element_stride = 1u;
	{% else %}
		let n_elements = {{  axis_dims }}u;
		let element_stride = {{ right_of_axis_chunk }}u;
	{% endif %}

	// Softmax = exp(input - max(input)) / sum(exp(input - max(input)))
	
	// First, determine max(input)
	// WGSL doesn't have a way to write -Infinity (https://github.com/gpuweb/gpuweb/issues/1769)
	// Therefore we use log(0) instead which returns -Infinity
	var max_element: Scalar = log(Scalar(0));
	for(var k: u32 = 0u; k < n_elements; k = k + 1u) {
		let element = input_0.data[chunk_start + (k * element_stride)];
		max_element = max(max_element, element);
	}

	// Calculate sum(exp(input - max(input)))
	var sum: Scalar = Scalar(0);
	for(var k: u32 = 0u; k < n_elements; k = k + 1u) {
		let element = input_0.data[chunk_start + (k * element_stride)];
		sum  = sum + exp(element - max_element);
	}

	// Calculate elements and write to output
	for(var k: u32 = 0u; k < n_elements; k = k + 1u) {
		let element = input_0.data[chunk_start + (k * element_stride)];
		output_0.data[chunk_start + (k * element_stride)] = exp(element - max_element) / sum;
	}
}