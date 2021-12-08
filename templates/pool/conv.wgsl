{% include "structs.wgsl" %}

[[group(0), binding(0)]]
var<storage, read> var_{{ input[0] }}: Array;

[[group(0), binding(1)]]
var<storage, read> var_{{ input[1] }}: Array;

{% if input | length == 3 %} // Bias
[[group(0), binding(2)]]
var<storage, read> var_{{ input[2] }}: Array;

[[group(0), binding(3)]]
var<storage, write> var_{{ output[0] }}: Array;

{% else %}
[[group(0), binding(2)]]
var<storage, write> var_{{ output[0] }}: Array;
{% endif %}  

[[stage(compute), workgroup_size(256, 1, 1)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
	let gidx = global_id.x;
        if (gidx < {{len_output }}u) {
	let batch = gidx / {{ M_x_H_x_W }}u; 
	let rest = gidx % {{ M_x_H_x_W }}u; 

        let m = rest / {{ H_x_W }}u;
        let rest = rest % {{ H_x_W }}u;
        
        let y = rest / {{ width }}u;
        let x = rest % {{ width }}u;
        
        var result: f32 = 0.0;

        let root_index = batch * {{ original_C_x_H_x_W }}u;

        let root_kernel_index = m * {{ kernel_channel_len }}u;

        for(var c: u32 = 0u; c < {{ channel }}u; c = c + 1u) {
            
            let base_index = root_index + c * {{ original_H_x_W }}u;
            let base_kernel_index = root_kernel_index + c * {{ kernel_len }}u;

            for(var i: u32 = 0u; i < {{ kernel_shape[0] }}u; i = i + 1u) {
                
		let tmp_y = y * {{ stride[0] }}u + i * {{ dilation[0] }}u - {{ pad[0] }}u; 

        	if ((tmp_y < {{ original_height }}u) && (tmp_y >= 0u)) {
        
	        for(var j: u32 = 0u; j < {{ kernel_shape[1] }}u; j = j + 1u) { 

                        let tmp_x = x * {{ stride[1] }}u + j * {{ dilation[1] }}u - {{ pad[1] }}u;

                        if ((tmp_x < {{ original_width }}u) && (tmp_x >= 0u)) {

                                let tmp_index = base_index + tmp_y * {{ original_width }}u + tmp_x;
                                let index_kernel = base_kernel_index + i * {{ kernel_shape[1] }}u + j;

                                result = var_{{ input[0] }}.data[tmp_index] * var_{{ input[1] }}.data[index_kernel] + result;
				
                        }
  	        }
		}
            }
	}

{% if op_type is matching("convrelu") %}
        var_{{ output[0] }}.data[gidx] = max(result{% if input | length == 3 %} + var_{{ input[2] }}.data[m]{% endif %}, 0.);
{% else %}
        var_{{ output[0] }}.data[gidx] = result{% if input | length == 3 %} + var_{{ input[2] }}.data[m]{% endif %};
{% endif %}
        }
}
