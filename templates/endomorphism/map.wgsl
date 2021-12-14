{%- include "structs.wgsl" -%}
[[group(0), binding(0)]]
var<storage, read> var_{{ inputs[0] }}: ArrayVector;

[[group(0), binding(1)]]
var<storage, write> var_{{ outputs[0] }}: ArrayVector;

[[stage(compute), workgroup_size(1)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
    let gidx = global_id.x;
    var_{{ outputs[0] }}.data[gidx] = {{ op_type | lower }}(var_{{ inputs[0] }}.data[gidx]);
}