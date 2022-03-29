{%- include "structs.wgsl" -%}

@group(0) @binding(0)
var<storage, read> input_0: ArrayVector;

@group(0) @binding(1)
var<storage, read> input_1: Array;

{%- if i_lens | length == 3 -%} // Bias
@group(0) @binding(2)
var<storage, read> input_2: Array;

@group(0) @binding(3)
var<storage, write> output_0: Array;

{%- else -%}
@group(0) @binding(2)
var<storage, write> output_0: Array;

{%- endif -%}  

@stage(compute) @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
	let gidx = global_id.x;

	var tmpsum = Scalar(0);
	var product = Scalar(0);

	for(var k: u32 = 0u; k < {{ i_shape[0][1] / 4 | int }}u; k = k + 1u) {
		let index_left = k; 
		let index_right = k * {{ i_shape[1][1] * 4 }}u + gidx; 

		let vec_left = input_0.data[index_left];

		let vec_right = Vec4(
			input_1.data[index_right], 
			input_1.data[index_right + {{ i_shape[1][1] }}u],
			input_1.data[index_right + {{ 2 * i_shape[1][1] }}u],
			input_1.data[index_right + {{ 3 * i_shape[1][1] }}u],
		);
	
		product = dot(vec_left, vec_right);
		tmpsum = tmpsum + product;
	}
	
	output_0.data[gidx] = 
		{%- if alpha != 1 -%} {{ alpha | float }} * {%- endif -%} 
		tmpsum
		{%- if i_lens | length == 3 -%}
			+ {%- if beta != 1 -%} {{ beta | float }} * {%- endif -%}
			input_2.data[gidx];
		{%- endif -%}
	;
}
