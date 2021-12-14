
{%- include "structs.wgsl" -%}

[[group(0), binding(0)]]
var<storage, read> var_{{ inputs[0] }}: ArrayMatrix;

[[group(0), binding(1)]]
var<storage, write> var_{{ outputs[0] }}: ArrayMatrix;


[[stage(compute), workgroup_size(1)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
    let gidx = global_id.x;
    var_{{ outputs[0] }}.data[gidx] = var_{{ inputs[0] }}.data[gidx];
}