{% extends "base.wgsl" %}
{% block structs %}{% include "struct-ArrayVector.wgsl" %}{% endblock structs %}
{% block main %}
[[stage(compute), workgroup_size(1)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
    let y = global_id.x % {{ right_columns / 4 | int }}u;
    let x = global_id.x / {{ right_columns / 4 | int }}u;
    let index = x * {{ right_columns }}u + y;

    var tmpsum = mat4x4<f32>(vec4<f32>(0.0, 0.0, 0.0, 0.0), vec4<f32>(0.0, 0.0, 0.0, 0.0), vec4<f32>(0.0, 0.0, 0.0, 0.0), vec4<f32>(0.0, 0.0, 0.0, 0.0));
    var product = mat4x4<f32>(vec4<f32>(0.0, 0.0, 0.0, 0.0), vec4<f32>(0.0, 0.0, 0.0, 0.0), vec4<f32>(0.0, 0.0, 0.0, 0.0), vec4<f32>(0.0, 0.0, 0.0, 0.0));

    for(var k: u32 = 0u; k < {{ left_columns / 4 | int }}u; k = k + 1u) {
        let index_left = x * {{ left_columns }}u + k; 
        let index_right = k * {{ left_columns }}u + y; 

        let mat_left = mat4x4<f32>(
                              {{ input[0] }}.data[index_left], 
                              {{ input[0] }}.data[index_left + {{ left_columns / 4| int }}u],
                              {{ input[0] }}.data[index_left + {{ 2 * left_columns / 4 | int }}u],
                              {{ input[0] }}.data[index_left + {{ 3 * left_columns / 4 | int }}u],
                          );
          
        let mat_right = mat4x4<f32>(
                              {{ input[1] }}.data[index_right], 
                              {{ input[1] }}.data[index_right + {{ right_columns / 4 | int }}u],
                              {{ input[1] }}.data[index_right + {{ 2 * right_columns / 4 | int }}u],
                              {{ input[1] }}.data[index_right + {{ 3 * right_columns / 4 | int }}u],
                          );
	
        product = mat_right * mat_left;
	
        for(var index_mat: u32 = 0u; index_mat < 4u; index_mat = index_mat + 1u) {
	        tmpsum[index_mat] = tmpsum[index_mat] + product[index_mat];
	}
    }
    
{% if input | length == 3 %}
    let bias_row = {{ input[2] }}.data[x]; 
    var bias = transpose(mat4x4<f32>(bias_row, bias_row, bias_row, bias_row));
    for(var index_mat: u32 = 0u; index_mat < 4u; index_mat = index_mat + 1u) {
            {{ output[0] }}.data[index + index_mat * {{ right_columns / 4 | int }}u] = {% if alpha != 1 %} {{ alpha | float }} * {% endif %}tmpsum[index_mat] + {% if beta != 1 %} {{ beta | float }} * {% endif %}bias[index_mat];
    }       
{% else %}
    for(var index_mat: u32 = 0u; index_mat < 4u; index_mat = index_mat + 1u) {
            {{ output[0] }}.data[index + index_mat * {{ right_columns / 4 | int }}u] = {% if alpha != 1 %} {{ alpha | float }} * {% endif %}tmpsum[index_mat];
    }       
{% endif %}  
}
{% endblock main %}