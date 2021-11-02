
{% include "structs.wgsl" %}

[[group(0), binding({{ bindings[0].counter }})]]
var<storage, read> var_{{ bindings[0].tensor }}: ArrayMatrix;

[[group(0), binding({{ bindings[1].counter }})]]
var<storage, write> var_{{ bindings[1].tensor }}: ArrayMatrix;


[[stage(compute), workgroup_size(1)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
    let gidx = global_id.x;
    var_{{ output[0] }}.data[gidx] = var_{{ input[0] }}.data[gidx];
}