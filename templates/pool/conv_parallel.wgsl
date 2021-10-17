{% include "structs.wgsl" %}

{% for binding in bindings %}
[[group(0), binding({{ binding.counter }})]]
var<storage, read_write> var_{{ binding.tensor }}: Array;
{% endfor %}

[[stage(compute), workgroup_size(1)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
	let gidx = global_id.x;
	let batch_number = gidx / {{ output_dims[1] * input_dims[1] * output_dims[2] * output_dims[3] }}u; 
        
	let rest = gidx % {{ output_dims[1] * input_dims[1] * output_dims[2] * output_dims[3] }}u; 

        let m = rest / {{ input_dims[1] * output_dims[2] * output_dims[3] }}u;
        let rest = rest % {{ input_dims[1] * output_dims[2] * output_dims[3] }}u;
        
        let c = rest / {{ output_dims[2] * output_dims[3] }}u;
        let rest = rest % {{ output_dims[2] * output_dims[3] }}u;
        
        let y = rest / {{ output_dims[3] }}u;
        let x = rest % {{ output_dims[3] }}u;
        
        var result: f32 = 0.0;

        for(var i: u32 = 0u; i < {{ kernel_shape[0] }}u; i = i + 1u) {
                
	        let tmp_y = y * {{ stride[0] }}u + i * {{ dilation[0] }}u - {{ pad[0] }}u; 

        	if ((tmp_y < {{ original_height }}u) && (tmp_y >= 0u)) {
        
	        for(var j: u32 = 0u; j < {{ kernel_shape[1] }}u; j = j + 1u) { 

                        let tmp_x = x * {{ stride[1] }}u + j * {{ dilation[1] }}u - {{ pad[1] }}u;

                        if ((tmp_x < {{ original_width }}u) && (tmp_x >= 0u)) {

                                let tmp_index = batch_number * {{ original_C_x_H_x_W }}u + c * {{ original_H_x_W }}u + tmp_y * {{ original_width }}u + tmp_x;
                                let index_kernel = m * {{ kernel_channel_len }}u + c * {{ kernel_len }}u + i * {{ kernel_shape[1] }}u + j;

                                result = fma(var_{{ input[0] }}.data[tmp_index],var_{{ input[1] }}.data[index_kernel], result);
                        }
  	        }
		}
        }

{% if op_type is matching("convrelu") %}
        var_{{ output[0] }}.data[gidx] = max(result{% if input | length == 3 %} + var_{{ input[2] }}.data[m]{% endif %}, 0.);
{% else %}
        var_{{ output[0] }}.data[gidx] = result{% if input | length == 3 %} + var_{{ input[2] }}.data[m]{% endif %};
{% endif %}

        for(var step: u32 = 1u; step < {{ input_dims[1] }}u; step = step * 2u ) {
                if(gidx % (step * 2u) == 0u){
                        var_{{ output[0] }}.data[gidx] = var_{{ output[0] }}.data[gidx] + var_{{ output[0] }}.data[gidx + step];
                }
                storageBarrier()
        }
}
