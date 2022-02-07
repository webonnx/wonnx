{%- include "structs.wgsl" -%}

[[group(0), binding(0)]]
var<storage, read> input_0: ArrayVector;

[[group(0), binding(1)]]
var<storage, read> input_1: ArrayVector;

{%- if i_lens | length == 3 -%} // Bias
[[group(0), binding(2)]]
var<storage, read> input_2: ArrayVector;

[[group(0), binding(3)]]
var<storage, write> output_0: ArrayVector;
{%- else -%}
[[group(0), binding(2)]]
var<storage, write> output_0: ArrayVector;
{%- endif -%}  

[[stage(compute), workgroup_size(1)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
    let y = global_id.x % {{ i_shape[1][1] / 4 | int }}u;
    let x = global_id.x / {{ i_shape[1][1] / 4 | int }}u;
    let index = x * {{ i_shape[1][1] }}u + y;

    var tmpsum = mat4x4<f32>(vec4<f32>(0.0, 0.0, 0.0, 0.0), vec4<f32>(0.0, 0.0, 0.0, 0.0), vec4<f32>(0.0, 0.0, 0.0, 0.0), vec4<f32>(0.0, 0.0, 0.0, 0.0));
    var product = mat4x4<f32>(vec4<f32>(0.0, 0.0, 0.0, 0.0), vec4<f32>(0.0, 0.0, 0.0, 0.0), vec4<f32>(0.0, 0.0, 0.0, 0.0), vec4<f32>(0.0, 0.0, 0.0, 0.0));

    for(var k: u32 = 0u; k < {{ i_shape[0][1] / 4 | int }}u; k = k + 1u) {
        let index_left = x * {{ i_shape[0][1] }}u + k; 
        let index_right = k * {{ i_shape[0][1] }}u + y; 

        let mat_left = mat4x4<f32>(
                              input_0.data[index_left], 
                              input_0.data[index_left + {{ i_shape[0][1] / 4 | int }}u],
                              input_0.data[index_left + {{ 2 * i_shape[0][1] / 4 | int }}u],
                              input_0.data[index_left + {{ 3 * i_shape[0][1] / 4 | int }}u],
                          );
          
        let mat_right = mat4x4<f32>(
                              input_1.data[index_right], 
                              input_1.data[index_right + {{ i_shape[1][1] / 4 | int }}u],
                              input_1.data[index_right + {{ 2 * i_shape[1][1] / 4 | int }}u],
                              input_1.data[index_right + {{ 3 * i_shape[1][1] / 4 | int }}u],
                          );
	
        product = mat_right * mat_left;
	
        for(var index_mat: u32 = 0u; index_mat < 4u; index_mat = index_mat + 1u) {
	        tmpsum[index_mat] = tmpsum[index_mat] + product[index_mat];
	}
    }
    
{%- if i_lens | length == 3 -%}
    let bias_row = input_2.data[x]; 
    var bias = transpose(mat4x4<f32>(bias_row, bias_row, bias_row, bias_row));
    for(var index_mat: u32 = 0u; index_mat < 4u; index_mat = index_mat + 1u) {
            output_0.data[index + index_mat * {{ i_shape[1][1] / 4 | int }}u] = {%- if alpha != 1 -%} {{ alpha | float }} * {%- endif -%}tmpsum[index_mat] + {%- if beta != 1 -%} {{ beta | float }} * {%- endif -%}bias[index_mat];
    }       
{%- else -%}
    for(var index_mat: u32 = 0u; index_mat < 4u; index_mat = index_mat + 1u) {
            output_0.data[index + index_mat * {{ i_shape[1][1] / 4 | int }}u] = {%- if alpha != 1 -%} {{ alpha | float }} * {%- endif -%}tmpsum[index_mat];
    }       
{%- endif -%}  
}