{%- include "structs.wgsl" -%}
[[group(0), binding(0)]]
var<storage, read> input_0: ArrayVector;

struct OutputArrayVector {
	data: [[stride({{ vec4_stride }})]] array<vec4<{{ cast_to_type }}>>;
}; 

[[group(0), binding(1)]]
var<storage, write> output_0: OutputArrayVector;

[[stage(compute), workgroup_size({{ workgroup_size_x }})]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
    let gidx = global_id.x;
    output_0.data[gidx] = vec4<{{ cast_to_type }}>(input_0.data[gidx]);
}