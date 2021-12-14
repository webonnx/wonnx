{%- include "structs.wgsl" -%}

[[group(0), binding(0)]]
var<storage, read> var_{{ inputs[0] }}: ArrayVector;

[[group(0), binding(1)]]
var<storage, read> var_{{ inputs[1] }}: Array;

{%- if inputs | length == 3 -%} // Bias
[[group(0), binding(2)]]
var<storage, read> var_{{ inputs[2] }}: Array;

[[group(0), binding(3)]]
var<storage, write> var_{{ outputs[0] }}: Array;

{%- else -%}
[[group(0), binding(2)]]
var<storage, write> var_{{ outputs[0] }}: Array;

{%- endif -%}  

[[stage(compute), workgroup_size(1)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
    let gidx = global_id.x;

    var tmpsum = 0.0;
    var product = 0.0;

    for(var k: u32 = 0u; k < {{ i_dims[0][1] / 4 | int }}u; k = k + 1u) {

        let index_left = k; 
        let index_right = k * {{ i_dims[1][1] * 4 }}u + gidx; 

        let vec_left = var_{{ inputs[0] }}.data[index_left];

        let vec_right = vec4<f32>(
                              var_{{ inputs[1] }}.data[index_right], 
                              var_{{ inputs[1] }}.data[index_right + {{ i_dims[1][1] }}u],
                              var_{{ inputs[1] }}.data[index_right + {{ 2 * i_dims[1][1] }}u],
                              var_{{ inputs[1] }}.data[index_right + {{ 3 * i_dims[1][1] }}u],
                          );
	
        product = dot(vec_left, vec_right);
	
	    tmpsum = tmpsum + product;
    }
    
    var_{{ outputs[0] }}.data[gidx] = {%- if alpha != 1 -%}{{ alpha | float }} * {%- endif -%}tmpsum
{%- if inputs | length == 3 -%}
 + {%- if beta != 1 -%}{{ beta | float }} * {%- endif -%}var_{{ inputs[2] }}.data[gidx];
{%- endif -%};
}
