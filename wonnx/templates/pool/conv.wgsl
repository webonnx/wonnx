{%- include "structs.wgsl" -%}

@group(0) @binding(0)
var<storage, read> input_0: Array;

@group(0) @binding(1)
var<storage, read> input_1: Array;

{%- if i_lens | length == 3 -%} // Bias
	@group(0) @binding(2)
	var<storage, read> input_2: Array;

	@group(0) @binding(3)
	var<storage, read_write> output_0: Array;

{%- else -%}
	@group(0) @binding(2)
	var<storage, read_write> output_0: Array;

{%- endif %}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
	let gidx = global_id.x;
	if (gidx < {{ o_lens[0] }}u) {
		let batch = gidx / {{ o_chunks[0][0] }}u; 
		var rest = gidx % {{ o_chunks[0][0] }}u; 

		let m = rest / {{ o_chunks[0][1] }}u;
		rest = rest % {{ o_chunks[0][1] }}u;
		
		let y = rest / {{ o_chunks[0][2] }}u;
		let x = rest % {{ o_chunks[0][2] }}u;
		
		var result: Scalar = Scalar();

		let root_index = batch * {{ i_chunks[0][0] }}u;
		let root_kernel_index = m * {{ kernel_channel_len }}u;

		for(var c: u32 = 0u; c < {{ channel }}u; c = c + 1u) {
			let base_index = root_index + c * {{ i_chunks[0][1] }}u;
			let base_kernel_index = root_kernel_index + c * {{ kernel_len }}u;

			for(var i: u32 = 0u; i < {{ kernel_shape[0] }}u; i = i + 1u) {
				let tmp_y = y * {{ stride[0] }}u + i * {{ dilation[0] }}u - {{ pad[0] }}u; 

				if ((tmp_y < {{ original_height }}u) && (tmp_y >= 0u)) {
					for(var j: u32 = 0u; j < {{ kernel_shape[1] }}u; j = j + 1u) { 
						let tmp_x = x * {{ stride[1] }}u + j * {{ dilation[1] }}u - {{ pad[1] }}u;

						if ((tmp_x < {{ original_width }}u) && (tmp_x >= 0u)) {
							let tmp_index = base_index + tmp_y * {{ original_width }}u + tmp_x;
							let index_kernel = base_kernel_index + i * {{ kernel_shape[1] }}u + j;
							result = input_0.data[tmp_index] * input_1.data[index_kernel] + result;			
						}
					}
				}
			}
		}

		{%- if i_lens | length == 3 -%}
			result = result + input_2.data[m];
		{%- endif -%}

		{% set activation_input = "result" %}
		{% set activation_output = "output_0.data[gidx]" %}
		{% set activation_type = op_type | replace(from="Conv", to="") %}
		{%- include "snippets/activation_scalar.wgsl" -%}
	}
}
