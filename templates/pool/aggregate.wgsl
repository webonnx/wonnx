{% include "structs.wgsl" %}

{% for binding in bindings %}
[[group(0), binding({{ binding.counter }})]]
var<storage, read_write> var_{{ binding.tensor }}: Array;
{% endfor %}

[[stage(compute), workgroup_size(1)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
	let gidx = global_id.x;
	let batch_number = gidx / {{ M_x_H_x_W }}u; 
	let rest = gidx % {{ M_x_H_x_W }}u; 

        let m = rest / {{ H_x_W }}u;
        let rest = rest % {{ H_x_W }}u;
        
        let y = rest / {{ width }}u;
        let x = rest % {{ width }}u;
        
        var result: f32 = 0.0;
        let base_index = batch_number * {{ original_C_x_H_x_W }}u + m * {{ original_H_x_W }}u + y * {{ stride[0] }}u * {{ original_width }}u+ x * {{ stride[1] }}u;
            for(var i: u32 = 0u; i < {{ kernel_shape[0] }}u; i = i + 1u) {
                
		let tmp_y = i * {{ dilation[0] }}u; 
        
	        for(var j: u32 = 0u; j < {{ kernel_shape[1] }}u; j = j + 1u) { 

                        let tmp_x = j * {{ dilation[1] }}u;

{% if op_type is matching("maxpool") %}
                                let tmp_index = base_index + tmp_y * {{ original_width }}u + tmp_x;
				result = max(result, var_{{ input[0] }}.data[tmp_index]);

{% elif op_type is matching("averagepool") %}
                                let tmp_index = base_index + tmp_y * {{ original_width }}u + tmp_x;
				result = result + var_{{ input[0] }}.data[tmp_index];
{% endif %}
                        }
  	        }
		
            

{% if op_type is matching("averagepool") %}
        var_{{ output[0] }}.data[gidx] = result/{{ kernel_len }}.;
{% else %}
        var_{{ output[0] }}.data[gidx] = result;
{% endif %}
}
