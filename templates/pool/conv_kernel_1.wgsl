{% include "structs.wgsl" %}

[[group(0), binding({{ bindings[0].counter }})]]
var<storage, read_write> var_{{ bindings[0].tensor }}: Array;

[[group(0), binding({{ bindings[1].counter }})]]
var<storage, read_write> var_{{ bindings[1].tensor }}: ArrayMatrix;

[[group(0), binding({{ bindings[2].counter }})]]
var<storage, read_write> var_{{ bindings[2].tensor }}: Array;

{% if input | length == 3 %} // Bias
[[group(0), binding({{ bindings[3].counter }})]]
var<storage, read_write> var_{{ bindings[3].tensor }}: Array;
{% endif %}  

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
        for(var c: u32 = 0u; c < {{ channel / 16 }}u; c = c + 1u) {

                let index_kernel = m * {{ kernel_channel_len / 16 }}u +  c;
                var matrix = var_{{ input[1] }}.data[index_kernel];
                
                let root_index = batch_number * {{ original_C_x_H_x_W }}u + 16u * c * {{ original_H_x_W }}u + y * {{ original_width }}u + x;
                for(var index_vec: u32 = 0u; index_vec < 4u; index_vec = index_vec + 1u) {
                        
                        let base_index = root_index + index_vec * {{ 4 * original_H_x_W }}u;
                        let tmp_data_0 = var_{{ input[0] }}.data[base_index];
                        let tmp_data_1 = var_{{ input[0] }}.data[base_index + {{ original_H_x_W }}u];
                        let tmp_data_2 = var_{{ input[0] }}.data[base_index + {{ 2 * original_H_x_W }}u];
                        let tmp_data_3 = var_{{ input[0] }}.data[base_index + {{ 3 * original_H_x_W }}u];
                        let tmp_vec = vec4<f32>(tmp_data_0, tmp_data_1, tmp_data_2, tmp_data_3);
                
                        result = dot(tmp_vec, matrix[index_vec]) + result;
                }
	}

{% if op_type is matching("convrelu") %}
        var_{{ output[0] }}.data[gidx] = max(result{% if input | length == 3 %} + var_{{ input[2] }}.data[m]{% endif %}, 0.);
{% else %}
        var_{{ output[0] }}.data[gidx] = result{% if input | length == 3 %} + var_{{ input[2] }}.data[m]{% endif %};
{% endif %}
}
