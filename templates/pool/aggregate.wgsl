{% include "structs.wgsl" %}

[[group(0), binding(0)]]
var<storage, read> var_{{ input[0] }}: Array;

[[group(0), binding(1)]]
var<storage, write> var_{{ output[0] }}: Array;

[[stage(compute), workgroup_size(256, 1, 1)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
	let gidx = global_id.x;
        if (gidx < {{ len_output / 4 }}u) {

	let batch = gidx / {{ M_x_H_x_W / 4 }}u; 
	let rest = gidx % {{ M_x_H_x_W / 4 }}u; 

        let m = rest / {{ H_x_W }}u;
        let rest = rest % {{ H_x_W }}u;
        
        let y = rest / {{ width }}u;
        let x = rest % {{ width }}u;
        
        var result = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        
        let base_index = batch * {{ original_C_x_H_x_W }}u + m * {{ original_H_x_W * 4 }}u + y * {{ stride[0] }}u * {{ original_width }}u+ x * {{ stride[1] }}u;
        for(var i: u32 = 0u; i < {{ kernel_shape[0] }}u; i = i + 1u) {
                
		let tmp_y = i * {{ dilation[0] }}u; 
        
	        for(var j: u32 = 0u; j < {{ kernel_shape[1] }}u; j = j + 1u) { 

                        let tmp_x = j * {{ dilation[1] }}u;

                        let tmp_index  = base_index + tmp_y * {{ original_width }}u + tmp_x;
                        let vector = vec4<f32>(
                                var_{{ input[0] }}.data[tmp_index],
                                var_{{ input[0] }}.data[tmp_index + {{ original_H_x_W }}u],
                                var_{{ input[0] }}.data[tmp_index + {{ 2 * original_H_x_W }}u],
                                var_{{ input[0] }}.data[tmp_index + {{ 3 * original_H_x_W }}u],
                        );
{% if op_type is matching("maxpool") %}

			result = max(result, vector);

{% elif op_type is matching("averagepool") %}
			result = result + vector;
{% endif %}
                        }
  	        }
		
            

{% if op_type is matching("averagepool") %}
        result = result/ {{ kernel_len }}.;
{% endif %}

        let base_index = batch * {{ M_x_H_x_W }}u + m * {{ H_x_W * 4 }}u + y * {{ width }}u + x;
        for(var index_vec: u32 = 0u; index_vec < 4u; index_vec = index_vec + 1u) {
                let index = base_index + index_vec * {{ H_x_W }}u;

                var_{{ output[0] }}.data[index] = result[index_vec];
        }
        }
}
