{%- include "structs.wgsl" -%}

{# 
// The smallest floating point number that can be represented in IEEE-754. This should be -3.40282347E+38. However, Google 
// Chrome's WGSL compiler (as of July 2022) complains that number cannot be represented in f32. Hence we are using +37f,
// which should be sufficiently low.
#}
{% set_global min_float = scalar_type ~ "(-3.40282347E+37f)" %}

@group(0) @binding(0)
var<storage, read> input_0: Array;

@group(0) @binding(1)
var<storage, read_write> output_0: Array;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
	let gidx = global_id.x;

	{% if (i_shape[0][1] % 4) == 0 %}
	if (gidx < {{ o_lens[0] / 4 | int }}u) {
		let batch = gidx / {{ o_chunks[0][0] / 4 | int }}u; 
		var rest = gidx % {{ o_chunks[0][0] / 4 | int }}u; 

		let m = rest / {{ o_chunks[0][1] }}u;
		rest = rest % {{ o_chunks[0][1] }}u;
	
		let y = rest / {{ o_chunks[0][2] }}u;
		let x = rest % {{ o_chunks[0][2] }}u;
		
		{% if op_type == "AveragePool" -%}
		var result = Vec4(Scalar(), Scalar(), Scalar(), Scalar());
		{% else %}
		var result = Vec4(
			{{ min_float }},
			{{ min_float }},
			{{ min_float }},
			{{ min_float }}
		);
		{% endif %}
		var value = result;

		let base_index = batch * {{ i_chunks[0][0] }}u + m * {{ i_chunks[0][1] * 4 }}u ;
		var tmp_y = 0u;
		var tmp_x = 0u;
		var tmp_index = 0u;
		var counter = Scalar();

		for(var i: u32 = 0u; i < {{ kernel_shape[0] }}u; i = i + 1u) {
			tmp_y = y * {{ stride[0] }}u + i * {{ dilation[0] }}u - {{ pad[0] }}u; 

			if ((tmp_y < {{ original_height }}u) && (tmp_y >= 0u)) {
				for(var j: u32 = 0u; j < {{ kernel_shape[1] }}u; j = j + 1u) { 
					tmp_x = x * {{ stride[1] }}u + j * {{ dilation[1] }}u - {{ pad[1] }}u;
						if ((tmp_x < {{ original_width }}u) && (tmp_x >= 0u)) {

							tmp_index  = base_index + tmp_y * {{ original_width }}u + tmp_x;
							value = Vec4(
								input_0.data[tmp_index],
								input_0.data[tmp_index + {{ i_chunks[0][1] }}u],
								input_0.data[tmp_index + {{ 2 * i_chunks[0][1] }}u],
								input_0.data[tmp_index + {{ 3 * i_chunks[0][1] }}u],
							);
							
							{%- if op_type == "MaxPool" -%}
								result = max(result, value);
							{%- elif op_type == "AveragePool" -%}
								result = result + value;
								counter = counter + {{ scalar_type }}(1);
							{%- endif -%}
						}
				}
			}
		}	

		{% if op_type == "AveragePool" -%}
			{% if count_include_pad == 0 %}
			result = result / counter;
			{% else %}
			result = result / {{ kernel_length }}.;
			{% endif %}
		{%- endif %}

		let base_index_2 = batch * {{ o_chunks[0][0] }}u + m * {{ o_chunks[0][1] * 4 }}u + y * {{ width }}u + x;

		for(var index_vec: u32 = 0u; index_vec < 4u; index_vec = index_vec + 1u) {
			let index = base_index_2 + index_vec * {{ o_chunks[0][1] }}u;
			output_0.data[index] = result[index_vec];
		}
	} else {
	if ((gidx >= 4u * {{ o_lens[0] / 4 | int }}u) && ( gidx < {{ o_lens[0] }}u)) {
	{% else %}
	if ( gidx < {{ o_lens[0] | int }}u ) {
	{% endif %}
		let batch = gidx / {{ o_chunks[0][0] }}u; 
		var rest = gidx % {{ o_chunks[0][0] }}u; 

		let m = rest / {{ o_chunks[0][1] }}u;
		rest = rest % {{ o_chunks[0][1] }}u;
	
		let y = rest / {{ o_chunks[0][2] }}u;
		let x = rest % {{ o_chunks[0][2] }}u;
		
		{% if op_type == "AveragePool" -%}
		var result = Scalar();
		{% else %}
		
		var result = {{ scalar_type }}({{ min_float }});
		{% endif %}
		var value = result;
		
		let base_index = batch * {{ i_chunks[0][0] }}u + m * {{ i_chunks[0][1] }}u;
		var tmp_y = 0u;
		var tmp_x = 0u;
		var tmp_index = 0u;
		var counter = Scalar();

		for(var i: u32 = 0u; i < {{ kernel_shape[0] }}u; i = i + 1u) {
			tmp_y = y * {{ stride[0] }}u + i * {{ dilation[0] }}u - {{ pad[0] }}u; 

			if ((tmp_y < {{ original_height }}u) && (tmp_y >= 0u)) {
				for(var j: u32 = 0u; j < {{ kernel_shape[1] }}u; j = j + 1u) { 
					tmp_x = x * {{ stride[1] }}u + j * {{ dilation[1] }}u - {{ pad[1] }}u;
						if ((tmp_x < {{ original_width }}u) && (tmp_x >= 0u)) {
							tmp_index  = base_index + tmp_y * {{ original_width }}u + tmp_x;

							value = input_0.data[tmp_index];
							
							{%- if op_type == "MaxPool" -%}
								result = max(result, value);
							{%- elif op_type == "AveragePool" -%}
								result = result + value;
								counter = counter + {{ scalar_type }}(1);
							{%- endif -%}
						}
				}
			}
		}

		{% if op_type == "AveragePool" -%}
			{% if count_include_pad == 0 %}
			result = result / counter;
			{% else %}
			result = result / {{ kernel_length }}.;
			{% endif %}
		{%- endif %}

		output_0.data[gidx] = result;
	}
	{% if (i_shape[0][1] % 4) == 0 %}
	}
	{% endif %}
}
