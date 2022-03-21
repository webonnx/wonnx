{%- include "structs.wgsl" -%}

[[group(0), binding(0)]]
var<storage, read> input_0: Array;

[[group(0), binding(1)]]
var<storage, write> output_0: Array;

[[stage(compute), workgroup_size(256, 1, 1)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
	let gidx = global_id.x;

	{% if (i_shape[0][1] % 4) == 0 %}
	if (gidx < {{ o_lens[0] / 4 }}u) {
		let batch = gidx / {{ o_chunks[0][0] / 4 }}u; 
		var rest = gidx % {{ o_chunks[0][0] / 4 }}u; 

		let m = rest / {{ o_chunks[0][1] }}u;
		rest = rest % {{ o_chunks[0][1] }}u;
	
		let y = rest / {{ o_chunks[0][2] }}u;
		let x = rest % {{ o_chunks[0][2] }}u;
		
		{% if op_type == "AveragePool" -%}
		var result = Vec4(Scalar(0), Scalar(0), Scalar(0), Scalar(0));
		{% else %}
		var result = Vec4(Scalar(-3.40282347E+38), Scalar(-3.40282347E+38), Scalar(-3.40282347E+38), Scalar(-3.40282347E+38));
		{% endif %}
		var value = result;

		let base_index = batch * {{ i_chunks[0][0] }}u + m * {{ i_chunks[0][1] * 4 }}u ;
		var tmp_y = 0u;
		var tmp_x = 0u;
		var tmp_index = 0u;
		var counter = Scalar(0);

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
								counter = counter + Scalar(1);
							{%- endif -%}
						}
				}
			}
		}	

		{% if op_type == "AveragePool" -%}
			result = result / counter;
		{%- endif %}

		let base_index_2 = batch * {{ o_chunks[0][0] }}u + m * {{ o_chunks[0][1] * 4 }}u + y * {{ width }}u + x;

		for(var index_vec: u32 = 0u; index_vec < 4u; index_vec = index_vec + 1u) {
			let index = base_index_2 + index_vec * {{ o_chunks[0][1] }}u;
			output_0.data[index] = result[index_vec];
		}
	} else {
	if ((gidx >= 4u * {{ o_lens[0] }}u) && ( gidx < {{ o_lens[0] }}u)) {
	{% else %}
	if ( gidx < {{ o_lens[0] }}u ) {
	{% endif %}
		let batch = gidx / {{ o_chunks[0][0] }}u; 
		var rest = gidx % {{ o_chunks[0][0] }}u; 

		let m = rest / {{ o_chunks[0][1] }}u;
		rest = rest % {{ o_chunks[0][1] }}u;
	
		let y = rest / {{ o_chunks[0][2] }}u;
		let x = rest % {{ o_chunks[0][2] }}u;
		
		{% if op_type == "AveragePool" -%}
		var result = Scalar(0);
		{% else %}
		var result = Scalar(-3.40282347E+38);
		{% endif %}
		var value = result;
		
		let base_index = batch * {{ i_chunks[0][0] }}u + m * {{ i_chunks[0][1] }}u;
		var tmp_y = 0u;
		var tmp_x = 0u;
		var tmp_index = 0u;
		var counter = Scalar(0);

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
								counter = counter + Scalar(1);
							{%- endif -%}
						}
				}
			}
		}

		{% if op_type == "AveragePool" -%}
			result = result / counter;
		{%- endif %}

		output_0.data[gidx] = result;
	}
	{% if (i_shape[0][1] % 4) == 0 %}
	}
	{% endif %}
}
