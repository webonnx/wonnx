{%- include "structs.wgsl" -%}

struct ArrayMatrix3 {
    data: [[stride(48)]] array<mat3x3<f32>>;
}; // this is used as both input and output for convenience

[[group(0), binding(0)]]
var<storage, read> {{ inputs[0] }}: Array;

[[group(0), binding(1)]]
var<storage, read> {{ inputs[1] }}: ArrayMatrix3;

{%- if inputs | length == 3 -%} // Bias
[[group(0), binding(2)]]
var<storage, read> {{ inputs[2] }}: ArrayVector;

[[group(0), binding(3)]]
var<storage, write> {{ outputs[0] }}: Array;

{%- else -%}
[[group(0), binding(2)]]
var<storage, write> {{ outputs[0] }}: Array;
{%- endif %}  

// conv_kernel_3.wgsl
[[stage(compute), workgroup_size(256, 1, 1)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
	let gidx = global_id.x;
        if (gidx < {{ o_lens[0]/4 }}u) {
	let batch = gidx / {{ o_chunks[0][0] / 4 }}u; 
	let rest = gidx % {{ o_chunks[0][0] / 4 }}u; 

        let m = rest / {{ o_chunks[0][1] }}u;
        let rest = rest % {{ o_chunks[0][1] }}u;
        
        let y = rest / {{ o_chunks[0][2] }}u;
        let x = rest % {{ o_chunks[0][2] }}u;
        
        var result = vec4<f32>(0., 0., 0., 0.);
        
        let root_index = batch * {{ i_chunks[0][0] }}u;
        let root_kernel_index = m * {{ channel * 4 }}u;

        for(var c: u32 = 0u; c < {{ channel }}u; c = c + 1u) {
            
            let base_index = root_index + c * {{ i_chunks[0][1] }}u;
            let base_kernel_index = root_kernel_index + c;

            var kernel_matrix_0 = {{ inputs[1] }}.data[base_kernel_index];
            var kernel_matrix_1 = {{ inputs[1] }}.data[base_kernel_index + {{ channel }}u];
            var kernel_matrix_2 = {{ inputs[1] }}.data[base_kernel_index + {{ 2 * channel }}u];
            var kernel_matrix_3 = {{ inputs[1] }}.data[base_kernel_index + {{ 3 * channel }}u];

            for(var i: u32 = 0u; i < {{ kernel_shape[0] }}u; i = i + 1u) {

                var tmp_vec = vec3<f32>(0., 0., 0.);

		let tmp_y = y * {{ stride[0] }}u + i * {{ dilation[0] }}u - {{ pad[0] }}u; 

        	if ((tmp_y < {{ original_height }}u) && (tmp_y >= 0u)) {
        
	        for(var j: u32 = 0u; j < {{ kernel_shape[1] }}u; j = j + 1u) { 

                        let tmp_x = x * {{ stride[1] }}u + j * {{ dilation[1] }}u - {{ pad[1] }}u;

                        if ((tmp_x < {{ original_width }}u) && (tmp_x >= 0u)) {

                                let tmp_index = base_index + tmp_y * {{ original_width }}u + tmp_x;
                                let index_kernel = base_kernel_index + i * {{ kernel_shape[1] }}u + j;
                                
                                tmp_vec[j] = {{ inputs[0] }}.data[tmp_index];

                        }
  	        }
		}

               result = tmp_vec * mat4x3<f32>(
                       kernel_matrix_0[i],
                       kernel_matrix_1[i],
                       kernel_matrix_2[i],
                       kernel_matrix_3[i]
               ) + result;
            }

        }       
	
        {% if inputs | length == 3 -%}
        result = result + {{ inputs[2] }}.data[m];
        {%- endif %}

{% set activation_input = "result" %}
{% set activation_output = "result" %}
{% set activation_type = op_type | replace(from="Conv", to="") %}
{%- include "snippets/activation_vec.wgsl" -%}

        let base_index = batch * {{ o_chunks[0][0] }}u + m * {{ o_chunks[0][1] * 4 }}u + y * {{ width }}u + x;
        for(var index_vec: u32 = 0u; index_vec < 4u; index_vec = index_vec + 1u) {
                let index = base_index + index_vec * {{ o_chunks[0][1] }}u;

                {{ outputs[0] }}.data[index] = result[index_vec];
        }
        }
}
