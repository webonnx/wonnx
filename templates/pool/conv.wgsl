{% include "structs.wgsl" %}


{% for binding in bindings %}
[[group(0), binding({{ binding.counter }})]]
var<storage, read_write> _{{ binding.tensor }}: Array;
{% endfor %}

[[stage(compute), workgroup_size(1)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
	let gidx = global_id.x;
	let batch_number = gidx / {{ M_x_H_x_W }}u; 
	let rest = gidx % {{ M_x_H_x_W }}u; 

        let m = rest / {{ H_x_W }}u;
        let rest = rest % {{ H_x_W }}u;
        
        let y = rest / {{ width }}u - {{ pad[0] }}u;
        let x = rest % {{ width }}u - {{ pad[1] }}u;
        
        var result: f32 = 0.0;
{% if op_type is matching("averagepool") %}
        var n: f32 = 0.0;      
{% endif %}

        for(var c: u32 = 0u; c < {{ channel }}u; c = c + 1u) {
            for(var i: u32 = 0u; i < {{ kernel_shape[0] }}u; i = i + 1u) {
                
		let tmp_y = y * {{ stride[0] }}u + i * {{ dilation[0] }}u; 

        	if ((tmp_y < {{ original_height }}u) && (tmp_y >= 0u)) {
        
	        for(var j: u32 = 0u; j < {{ kernel_shape[1] }}u; j = j + 1u) { 

                        let tmp_x = x * {{ stride[1] }}u + j * {{ dilation[1] }}u;

                        if ((tmp_x < {{ original_width }}u) && (tmp_x >= 0u)) {

                                let tmp_index = batch_number * {{ original_C_x_H_x_W }}u + c * {{ original_H_x_W }}u + tmp_y * {{ original_width }}u + tmp_x;

{% if op_type is matching("conv") or op_type is matching("convrelu") %}
                                let index_kernel = m * {{ kernel_channel_len }}u + c * {{ kernel_len }}u + i * {{ kernel_shape[1] }}u + j;

                                result = result + _{{ input[0] }}.data[tmp_index] * _{{ input[1] }}.data[index_kernel];
				
{% elif op_type is matching("maxpool") %}
				result = max(result, _{{ input[0] }}.data[tmp_index]);

{% elif op_type is matching("averagepool") %}
				result = result + _{{ input[0] }}.data[tmp_index];
				n = n + 1.0;
{% endif %}
  	              }
  	        }
		}
            }
	}


{% if op_type is matching("averagepool") %}
        _{{ output[0] }}.data[gidx] = result/n;
{% elif op_type is matching("convrelu") %}
        _{{ output[0] }}.data[gidx] = max(result{% if input | length == 3 %} + _{{ input[2] }}.data[m]{% endif %}, 0.0);
{% else %}
        _{{ output[0] }}.data[gidx] = result{% if input | length == 3 %} + _{{ input[2] }}.data[m]{% endif %};
{% endif %}
}
