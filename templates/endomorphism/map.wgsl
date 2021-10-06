{% extends "base.wgsl" %}
{% block structs %}{% include "struct-ArrayVector.wgsl" %}{% endblock structs %}
{% block main %}
[[stage(compute), workgroup_size(1)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
    let gidx = global_id.x;
    _{{ output[0] }}.data[gidx] = {{ op_type }}(_{{ input[0] }}.data[gidx]);
}
{% endblock main %}