{%- include "structs.wgsl" -%}
@group(0) @binding(0)
var<storage, read> input_0: ArrayVector;

const pi: f32 = 3.1415;

@group(0) @binding(1)
var<storage, read_write> output_0: ArrayVector;

@compute @workgroup_size({{ workgroup_size_x }})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
	let gidx = global_id.x;
    var intermediate = 2.0/sqrt(pi)*(input_0.data[gidx]+ pow(input_0.data[gidx],vec4(3.0,3.0,3.0,3.0))*0.08943 );
    intermediate = clamp(intermediate,vec4(-10.0,-10.0,-10.0,-10.0),vec4(10.0,10.0,10.0,10.0));
	output_0.data[gidx] = tanh(intermediate);
}