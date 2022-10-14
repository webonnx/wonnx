{%- include "structs.wgsl" -%}


struct ArrayVector {
	data: array<Vec4>
};


@group(0) @binding(0)
var<storage, read> input_0: ArrayVector;

struct OutputArray {
    data: array<i32>
}

@group(0) @binding(1)
var<storage, read_write> output_0: OutputArray;


@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
	let gidx = global_id.x;
    let xxxx = input_0.data[0];
     {% for idx in idxs %}
         output_0.data[{{ loop.index0 }}] = {{ i_shape[0][idx] }};
     {% endfor %}

}
