{%- include "structs.wgsl" -%}

@group(0) @binding(0)
var<storage, read> input_0: Array;

@group(0) @binding(1)
var<storage, read> input_1: Array;

@group(0) @binding(2)
var<storage, write> output_0: Array;

@compute @workgroup_size({{ workgroup_size_x }}, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
	let gidx = global_id.x;

	{# We will be called for each element in the output tensor. Determine the corresponding indices in the source tensors #}
	var lhs_index = 0u;
	var rhs_index = 0u;
	var rest = gidx;
	{% for dim in o_shape[0] %}
		{% if dim > 1 %}
			let out_index = rest / {{ o_chunks[0][loop.index0] }}u;

			{% if lhs_padded_shape[loop.index0] > 1 %}
				lhs_index = lhs_index + (out_index * {{ lhs_padded_chunks[loop.index0] }}u);
			{% endif %}

			{% if rhs_padded_shape[loop.index0] > 1 %}
				rhs_index = rhs_index + (out_index * {{ rhs_padded_chunks[loop.index0] }}u);
			{% endif %}
			rest = rest % {{ o_chunks[0][loop.index0] }}u;
		{% endif %}
	{% endfor %}

	let lhs = input_0.data[lhs_index];
	let rhs = input_1.data[rhs_index];

	{% if op_type == "Pow" %}
		output_0.data[gidx] = pow(lhs, rhs);
	{% elif op_type == "PRelu" %}
		output_0.data[gidx] = max(lhs, Scalar())
							+ min(lhs, Scalar()) * rhs;
	{% else %}
		output_0.data[gidx] = (lhs {{ op_type }} rhs);
	{% endif %}
}
