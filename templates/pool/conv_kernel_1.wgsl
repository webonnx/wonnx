{% include "structs.wgsl" %}

[[group(0), binding(0)]]
var<storage, read> var_{{ input[0] }}: Array;

[[group(0), binding(1)]]
var<storage, read> var_{{ input[1] }}: ArrayMatrix;

{% if input | length == 3 %} // Bias
[[group(0), binding(2)]]
var<storage, read> var_{{ input[2] }}: ArrayVector;

[[group(0), binding(3)]]
var<storage, write> var_{{ output[0] }}: Array;
{% else %}
[[group(0), binding(2)]]
var<storage, write> var_{{ output[0] }}: Array;
{% endif %}  

[[stage(compute), workgroup_size(256, 1, 1)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
        let gidx = global_id.x;
        if (gidx < {{ len_output }}u) {
	let batch = gidx / {{ M_x_H_x_W / 4 }}u; 
	let rest = gidx % {{ M_x_H_x_W / 4 }}u; 

        let m = rest / {{ H_x_W }}u;
        let xy = rest % {{ H_x_W }}u;
                
        var result = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        
        let root_index = batch * {{ original_C_x_H_x_W }}u + xy;
        let root_kernel_index = m * {{ channel / 16 * 4 }}u;

        for(var c: u32 = 0u; c < {{ channel / 16 }}u; c = c + 1u) {

                let base_index = root_index + c * {{ 16 * original_H_x_W }}u;
                var base_kernel_index = root_kernel_index + c;
                
                var matrix_0 = var_{{ input[1] }}.data[base_kernel_index];
                var matrix_1 = var_{{ input[1] }}.data[base_kernel_index + {{ channel / 16 }}u];
                var matrix_2 = var_{{ input[1] }}.data[base_kernel_index + {{ 2 * channel / 16 }}u];
                var matrix_3 = var_{{ input[1] }}.data[base_kernel_index + {{ 3 * channel / 16 }}u];

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

        let base_index = batch * {{ M_x_H_x_W }}u + m * {{ H_x_W * 4 }}u + xy;
        for(var index_vec: u32 = 0u; index_vec < 4u; index_vec = index_vec + 1u) {
                let index = base_index + index_vec * {{ H_x_W }}u;

                var_{{ output[0] }}.data[index] = result[index_vec];
        }
        }
}
