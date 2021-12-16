{%- include "structs.wgsl" -%}

[[group(0), binding(0)]]
var<storage, read> {{ inputs[0] }}: Array;

[[group(0), binding(1)]]
var<storage, write> {{ outputs[0] }}: Array;

// aggregate.wgsl
[[stage(compute), workgroup_size(256, 1, 1)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
	let gidx = global_id.x;
        if (gidx < {{ o_lens[0] / 4 }}u) {

	let batch = gidx / {{ o_chunks[0][0] / 4 }}u; 
	let rest = gidx % {{ o_chunks[0][0] / 4 }}u; 

        let m = rest / {{ o_chunks[0][1] }}u;
        let rest = rest % {{ o_chunks[0][1] }}u;
 
        let y = rest / {{ o_chunks[0][2] }}u;
        let x = rest % {{ o_chunks[0][2] }}u;
        
        var result = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        
        let base_index = batch * {{ i_chunks[0][0] }}u + m * {{ i_chunks[0][1] * 4 }}u + y * {{ stride[0] }}u * {{ original_width }}u+ x * {{ stride[1] }}u;
        for(var i: u32 = 0u; i < {{ kernel_shape[0] }}u; i = i + 1u) {
                
		let tmp_y = i * {{ dilation[0] }}u; 
        
	        for(var j: u32 = 0u; j < {{ kernel_shape[1] }}u; j = j + 1u) { 

                        let tmp_x = j * {{ dilation[1] }}u;

                        let tmp_index  = base_index + tmp_y * {{ original_width }}u + tmp_x;
                        let vector = vec4<f32>(
                                {{ inputs[0] }}.data[tmp_index],
                                {{ inputs[0] }}.data[tmp_index + {{ i_chunks[0][1] }}u],
                                {{ inputs[0] }}.data[tmp_index + {{ 2 * i_chunks[0][1] }}u],
                                {{ inputs[0] }}.data[tmp_index + {{ 3 * i_chunks[0][1] }}u],
                        );
{%- if op_type == "MaxPool" -%}

			result = max(result, vector);

{%- elif op_type == "AveragePool" -%}
			result = result + vector;
{%- endif -%}
                        }
  	        }
		
            

{% if op_type == "AveragePool" -%}
        result = result / {{ kernel_len }}.;
{%- endif %}

        let base_index = batch * {{ o_chunks[0][0] }}u + m * {{ o_chunks[0][1] * 4 }}u + y * {{ width }}u + x;
        for(var index_vec: u32 = 0u; index_vec < 4u; index_vec = index_vec + 1u) {
                let index = base_index + index_vec * {{ o_chunks[0][1] }}u;

                {{ outputs[0] }}.data[index] = result[index_vec];
        }
        }
}
