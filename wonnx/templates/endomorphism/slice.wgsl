{%- include "structs.wgsl" -%}

struct Indices {
	data: array<i32>
};

struct Chunk {
	data: array<{{ chunk_type }}>
};

@group(0) @binding(0)
var<storage, read> input_0: Chunk; // data

@group(0) @binding(1)
var<storage, read> input_1: Indices; // indices

@group(0) @binding(2)
var<storage, read_write> output_0: Chunk;

@compute @workgroup_size({{ workgroup_size_x }}, {{ workgroup_size_y }})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
	// TODO @ Raphael
}