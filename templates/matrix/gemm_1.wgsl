{% include "structs.wgsl" %}

[[group(0), binding({{ bindings[0].counter }})]]
var<storage, read_write> _{{ bindings[0].tensor }}: ArrayVector;

[[group(0), binding({{ bindings[1].counter }})]]
var<storage, read_write> _{{ bindings[1].tensor }}: Array;

[[group(0), binding({{ bindings[2].counter }})]]
var<storage, read_write> _{{ bindings[2].tensor }}: Array;

{% if input | length == 3 %} // Bias
[[group(0), binding({{ bindings[3].counter }})]]
var<storage, read_write> _{{ bindings[3].tensor }}: Array;
{% endif %}  

[[stage(compute), workgroup_size(1)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
    let gidx = global_id.x;

    var tmpsum = 0.0;
    var product = 0.0;

    for(var k: u32 = 0u; k < {{ left_columns / 4 | int }}u; k = k + 1u) {

        let index_left = k; 
        let index_right = k * {{ right_columns * 4 }}u + gidx; 

        let vec_left = _{{ input[0] }}.data[index_left];

        let vec_right = vec4<f32>(
                              _{{ input[1] }}.data[index_right], 
                              _{{ input[1] }}.data[index_right + {{ right_columns }}u],
                              _{{ input[1] }}.data[index_right + {{ 2 * right_columns }}u],
                              _{{ input[1] }}.data[index_right + {{ 3 * right_columns }}u],
                          );
	
        product = dot(vec_left, vec_right);
	
	    tmpsum = tmpsum + product;
    }
    
{% if input | length == 3 %} // Bias
    let bias_row = _{{ input[2] }}.data[gidx]; 
    
    _{{ output[0] }}.data[gidx] = {% if alpha != 1 %} {{ alpha | float }} * {% endif %}tmpsum + {% if beta != 1 %} {{ beta | float }} * {% endif %}bias_row;
       
{% else %}
    _{{ output[0] }}.data[gidx] = {% if alpha != 1 %} {{ alpha | float }} * {% endif %}tmpsum;
{% endif %}  
}
