{%- include "structs.wgsl" -%}

struct Indices {
	data: [[stride(4)]] array<i32>;
};

struct Batch {
	data: [[stride({{ scalar_stride * batch_size }})]] array<{{ batch_type }}>;
};

[[group(0), binding(0)]]
var<storage, read> input_0: Batch; // data

[[group(0), binding(1)]]
var<storage, read> input_1: Indices; // indices

[[group(0), binding(2)]]
var<storage, write> output_0: Batch;

[[stage(compute), workgroup_size({{ workgroup_size_x }}, {{ workgroup_size_y }})]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
	let index_index = global_id.x; // Index of the index in the indices array that we are currently processing
	let batch_index = global_id.y; // Batch of elements that we are copying for this index (batch size determined dynamically)

	// Negative indexing is apparently allowed; see https://github.com/onnx/onnx/blob/main/docs/Operators.md#inputs-38
	var index = input_1.data[index_index];
	if(index < 0) {
		index = {{ i_shape[0][0] }} + index;
	}

	output_0.data[(index_index * u32({{ elements_per_index / batch_size }})) + batch_index] = input_0.data[index * i32({{ elements_per_index / batch_size }}) + i32(batch_index)];
}