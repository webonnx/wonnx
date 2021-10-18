{% include "structs.wgsl" %}

[[group(0), binding({{ bindings[0].counter }})]]
var<storage, read_write> var_{{ bindings[0].tensor }}: Array;

[[group(0), binding({{ bindings[1].counter }})]]
var<storage, read_write> var_{{ bindings[1].tensor }}: ArrayMatrix;

{% if input | length == 3 %} // Bias
[[group(0), binding({{ bindings[2].counter }})]]
var<storage, read_write> var_{{ bindings[2].tensor }}: Array;
{% endif %}  

[[group(0), binding({{ bindings[3].counter }})]]
var<storage, read_write> var_{{ bindings[3].tensor }}: Array;

[[stage(compute), workgroup_size(1)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
	let gidx = global_id.x;
	let batch_number = gidx / {{ M_x_H_x_W / 4 }}u; 
	let rest = gidx % {{ M_x_H_x_W / 4 }}u; 

        let m = rest / {{ H_x_W }}u;
        let xy = rest % {{ H_x_W }}u;
                
        var result = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        
        let root_index = batch_number * {{ original_C_x_H_x_W }}u + xy;
        let root_kernel_index = m * {{ channel / 16 * 4 }}u;

        for(var c: u32 = 0u; c < {{ channel / 16 }}u; c = c + 1u) {

                var base_kernel_index = root_kernel_index + c;
                var matrix_0 = var_{{ input[1] }}.data[base_kernel_index];

                base_kernel_index = base_kernel_index + {{ channel / 16 }}u;
                var matrix_1 = var_{{ input[1] }}.data[base_kernel_index];

                base_kernel_index = base_kernel_index + {{ channel / 16 }}u;
                var matrix_2 = var_{{ input[1] }}.data[base_kernel_index];

                base_kernel_index = base_kernel_index + {{ channel / 16 }}u;
                var matrix_3 = var_{{ input[1] }}.data[base_kernel_index];

                let base_index = root_index + c * {{ 16 * original_H_x_W }}u;
                for(var index_c_vec: u32 = 0u; index_c_vec < 4u; index_c_vec = index_c_vec + 1u) {
                        
                        let base_index = base_index + index_c_vec * {{ 4 * original_H_x_W }}u;
                        let tmp_vec = vec4<f32>(
                                var_{{ input[0] }}.data[base_index],
                                var_{{ input[0] }}.data[base_index + {{ original_H_x_W }}u],
                                var_{{ input[0] }}.data[base_index + {{ 2 * original_H_x_W }}u],
                                var_{{ input[0] }}.data[base_index + {{ 3 * original_H_x_W }}u],
                        );
                
                        result = tmp_vec * mat4x4<f32>(
                             matrix_0[index_c_vec],
                             matrix_1[index_c_vec],
                             matrix_2[index_c_vec],
                             matrix_3[index_c_vec],                
                        ) + result; 
                }
	}

{% if op_type is matching("convrelu") %}
        result = max(result{% if input | length == 3 %} + var_{{ input[2] }}.data[m]{% endif %}, vec4<f32>(0., 0., 0., 0.));
{% else %}
        {% if input | length == 3 %}result = result + var_{{ input[2] }}.data[m]{% endif %};
{% endif %}

        let base_index = batch_number * {{ M_x_H_x_W }}u + m * {{ H_x_W * 4 }}u + xy;
        for(var index_vec: u32 = 0u; index_vec < 4u; index_vec = index_vec + 1u) {
                let index = base_index + index_vec * {{ H_x_W }}u;

                var_{{ output[0] }}.data[index] = result[index_vec];
        }
}
