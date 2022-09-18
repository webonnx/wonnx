{%- include "structs.wgsl" -%}

@group(0) @binding(0)
var<storage, read> input_0: Array;

@group(0) @binding(1)
var<storage, read> input_1: ArrayMatrix;

{%- if i_lens | length == 3 -%} // Bias
	@group(0) @binding(2)
	var<storage, read> input_2: ArrayVector;

	@group(0) @binding(3)
	var<storage, read_write> output_0: Array;

{%- else -%}
	@group(0) @binding(2)
	var<storage, read_write> output_0: Array;

{%- endif %}


@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
	let gidx = global_id.x;
	if (gidx < {{ o_lens[0]/4 | int }}u) {
		let batch = gidx / {{ o_chunks[0][0] / 4 | int }}u; 
		let rest = gidx % {{ o_chunks[0][0] / 4 | int }}u; 

		let m = rest / {{ o_chunks[0][1] }}u;
		let xy = rest % {{ o_chunks[0][1] }}u;
			
		var result = Vec4(Scalar(), Scalar(), Scalar(), Scalar());
		
		let root_index = batch * {{ i_chunks[0][0] }}u + xy;
		let root_kernel_index = m * {{ channel / 16 * 4 | int }}u;

		for(var c: u32 = 0u; c < {{ channel / 16 | int }}u; c = c + 1u) {
			let base_index = root_index + c * {{ 16 * i_chunks[0][1] }}u;
			var base_kernel_index = root_kernel_index + c;
			
			var matrix_0 = input_1.data[base_kernel_index];
			var matrix_1 = input_1.data[base_kernel_index + {{ channel / 16 | int }}u];
			var matrix_2 = input_1.data[base_kernel_index + {{ 2 * channel / 16 | int }}u];
			var matrix_3 = input_1.data[base_kernel_index + {{ 3 * channel / 16 | int }}u];

			for(var index_c_vec: u32 = 0u; index_c_vec < 4u; index_c_vec = index_c_vec + 1u) {
				let base_index_2 = base_index + index_c_vec * {{ 4 * i_chunks[0][1] }}u;

				let tmp_vec = Vec4(
					input_0.data[base_index_2],
					input_0.data[base_index_2 + {{ i_chunks[0][1] }}u],
					input_0.data[base_index_2 + {{ 2 * i_chunks[0][1] }}u],
					input_0.data[base_index_2 + {{ 3 * i_chunks[0][1] }}u],
				);
			
				result = tmp_vec * Mat4x4(
					matrix_0[index_c_vec],
					matrix_1[index_c_vec],
					matrix_2[index_c_vec],
					matrix_3[index_c_vec],
				) + result; 
			}
		}

		{% if i_lens | length == 3 -%}
			result = result + input_2.data[m];
		{%- endif %}

		{% set activation_input = "result" %}
		{% set activation_output = "result" %}
		{% set activation_type = op_type | replace(from="Conv", to="") %}
		{%- include "snippets/activation_vec.wgsl" %}

		let base_index_3 = batch * {{ o_chunks[0][0] }}u + m * {{ o_chunks[0][1] * 4 }}u + xy;
		for(var index_vec: u32 = 0u; index_vec < 4u; index_vec = index_vec + 1u) {
			let index = base_index_3 + index_vec * {{ o_chunks[0][1] }}u;
			output_0.data[index] = result[index_vec];
		}
	}
}
