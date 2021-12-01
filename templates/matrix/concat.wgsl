
{% include "structs.wgsl" %}

[[group(0), binding(0)]]
var<storage, read> var_{{ input[0] }}: ArrayMatrix;

[[group(0), binding(1)]]
var<storage, read> var_{{ input[1] }}: ArrayMatrix;

[[group(0), binding(2)]]
var<storage, write> var_{{ output[0] }}: ArrayMatrix;


[[stage(compute), workgroup_size(1)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
    let gidx = global_id.x;
    if (gidx < {{ len_0 / 16 }}u) {
	    var_{{ output[0] }}.data[gidx] = var_{{ input[0] }}.data[gidx];
    } else {
    	var_{{ output[0] }}.data[gidx] = var_{{ input[1] }}.data[gidx - {{ len_0 / 16 }}u];
    }
}
