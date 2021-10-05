{% extends "base.wgsl" %}
{% block structs %}{% include "structs.wgsl" %}{% endblock structs %}
{% block main %}
[[stage(compute), workgroup_size(1)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
	
	let batch_number = global_id.x / {{ M_x_H_x_W }}u; 
	let rest = global_id.x / {{ M_x_H_x_W }}u; 

        let m = rest / {{ H_x_W }}u;
        let rest = global_id.x % {{ H_x_W }}u;
        
        let y = rest / {{ width }}u - {{ pad[0] }}u;
        let x = rest % {{ width }}u - {{ pad[1] }}u;
        
        var result: f32 = 0.0;
        var n: f32 = 0.0;
        
        for(var c: u32 = 0u; c < {{ channel }}u; c = c + 1u) {
            for(var i: u32 = 0u; i < {{ kernel_shape[0] }}u; i = i + {{ stride[0] }}u) {
                
		let tmp_y = y + i * {{ dilation[0] }}u; 

        	if ((tmp_y < {{ original_height }}u) && (tmp_y >= 0u)) {
        
	        for(var j: u32 = 0u; j < {{ kernel_shape[1] }}u; j = j + {{ stride[1] }}u) { 

                        let tmp_x = x + j * {{ dilation[1] }}u;

                        if ((tmp_x < {{ original_width }}u) && (tmp_x >= 0u)) {

                                let tmp_index = batch_number * {{ original_C_x_H_x_W }}u + c * {{ original_H_x_W }}u + tmp_y * {{ original_width }}u + tmp_x;
                                let index_div_4 = tmp_index / 4u;
                                let index_rest_4 = tmp_index % 4u;

{% if op_type is matching("conv") %}
                                let index_kernel = m * {{ kernel_channel_len }}u + c * {{ kernel_len }}u + i * {{ kernel_shape[1] }}u + j;

                                result = result + {{ input[0] }}.data[index_div_4][index_rest_4] * {{ input[1] }}.data[index_kernel];
				
{% elif op_type is matching("maxpool") %}
				result = max(result, {{ input[0] }}.data[index_div_4][index_rest_4]);

{% elif op_type is matching("averagepool") %}
				result = result + {{ input[0] }}.data[index_div_4][index_rest_4];
				n = n + 1.0;
{% endif %}
  	              }
  	        }
		}
            }
	}

        let gidx = global_id.x / 4u;
        let gidx_rest_4 = global_id.x % 4u;

{% if op_type is matching("averagepool") %}
        {{ output[0] }}.data[gidx][gidx_rest_4] = result/n;
{% else %}
        {{ output[0] }}.data[gidx][gidx_rest_4] = result{% if input | length == 3 %} + {{ input[2]}}.data[m]{% endif %};
{% endif %}
}
{% endblock main %}