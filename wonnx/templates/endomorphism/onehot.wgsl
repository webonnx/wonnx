{%- include "structs.wgsl" -%}

struct Indices {
	data: [[stride(4)]] array<i32>;
};

struct Depth {
	data: i32;
};

[[group(0), binding(0)]]
var<storage, read> input_indexes: Indices;

[[group(0), binding(1)]]
var<storage, read> input_depth: Depth;

[[group(0), binding(2)]]
var<storage, read> input_values: Array;

[[group(0), binding(3)]]
var<storage, write> output_0: Array;

[[stage(compute), workgroup_size(1)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
	let index_of_index = global_id.x;
	let depth = u32(input_depth.data);
	var index = input_indexes.data[index_of_index];
	if(index < 0) {
		index = i32(depth) + index;
	}
	
	let offset = index_of_index * depth;
	let off_value = input_values.data[0];
	let on_value = input_values.data[1];

	for(var i = 0; i < i32(depth); i = i + 1) {
		let ii = offset + u32(i);
		if(i == index) {
			output_0.data[ii] = on_value;
		}
		else {
			output_0.data[ii] = off_value;
		}
	}
}