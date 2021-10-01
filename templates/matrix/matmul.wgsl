{% extends "base.wgsl" %}
{% block structs %}{% include "struct-ArrayVector.wgsl" %}{% endblock structs %}
{% block main %}
[[stage(compute), workgroup_size(1)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
    let y = global_id.x % {right_columns_div_4}u;
    let x = global_id.x / {right_columns_div_4}u;
    let index = x * {right_columns}u + y;

    var tmpsum = mat4x4<f32>(vec4<f32>(0.0, 0.0, 0.0, 0.0), vec4<f32>(0.0, 0.0, 0.0, 0.0), vec4<f32>(0.0, 0.0, 0.0, 0.0), vec4<f32>(0.0, 0.0, 0.0, 0.0));
    var product = mat4x4<f32>(vec4<f32>(0.0, 0.0, 0.0, 0.0), vec4<f32>(0.0, 0.0, 0.0, 0.0), vec4<f32>(0.0, 0.0, 0.0, 0.0), vec4<f32>(0.0, 0.0, 0.0, 0.0));

    for(var k: u32 = 0u; k < {left_columns_div_4}u; k = k + 1u) {{
        let index_left = x * {left_columns}u + k; 
        let index_right = k * {left_columns}u + y; 

        let mat_left = mat4x4<f32>(
                              {input_left}.data[index_left], 
                              {input_left}.data[index_left + {left_columns_div_4}u],
                              {input_left}.data[index_left + 2u * {left_columns_div_4}u],
                              {input_left}.data[index_left + 3u * {left_columns_div_4}u],
                          );
          
        let mat_right = mat4x4<f32>(
                              {input_right}.data[index_right], 
                              {input_right}.data[index_right + {right_columns_div_4}u],
                              {input_right}.data[index_right + 2u * {right_columns_div_4}u],
                              {input_right}.data[index_right + 3u * {right_columns_div_4}u],
                          );
	
        product = mat_right * mat_left;
	
        for(var index_mat: u32 = 0u; index_mat < 4u; index_mat = index_mat + 1u) {{
	        tmpsum[index_mat] = tmpsum[index_mat] + product[index_mat];
	    }}
    }}

    {output}.data[index] = tmpsum[0u];
    {output}.data[index + {right_columns_div_4}u] = tmpsum[1u];
    {output}.data[index + 2u * {right_columns_div_4}u] = tmpsum[2u];
    {output}.data[index + 3u * {right_columns_div_4}u] = tmpsum[3u];
      
}
{% endblock main %}