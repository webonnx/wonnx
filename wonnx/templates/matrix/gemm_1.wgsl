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

@compute @workgroup_size(1, {{ workgroup_size_y }})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
	let gidx = global_id.x;

	{# Calculate stacking offsets #}
	let left_offset = global_id.y * {{ stack_left_stride / 4 }}u;
	let right_offset = global_id.y * {{ stack_right_stride }}u;
	let output_offset = global_id.y * {{ stack_output_stride }}u;

	var tmpsum = Scalar();
	var product = Scalar();

	for(var k: u32 = 0u; k < {{ left_shape[1] / 4 | int }}u; k = k + 1u) {
		let index_left = left_offset + k; 
		let index_right = right_offset + (k * {{ right_shape[1] * 4 }}u) + gidx; 

		let vec_left = input_0.data[index_left];

		let vec_right = Vec4(
			input_1.data[index_right], 
			input_1.data[index_right + {{ right_shape[1] }}u],
			input_1.data[index_right + {{ 2 * right_shape[1] }}u],
			input_1.data[index_right + {{ 3 * right_shape[1] }}u],
		);
	
		product = dot(vec_left, vec_right);
		tmpsum = tmpsum + product;
	}
	
	output_0.data[output_offset + gidx] = 
		{%- if alpha != 1 -%} {{ alpha | float }} * {%- endif -%} 
		tmpsum
		{%- if i_lens | length == 3 -%}
			+ {%- if beta != 1 -%} {{ beta | float }} * {%- endif -%}
			input_2.data[gidx];
		{%- endif -%}
	;
}
