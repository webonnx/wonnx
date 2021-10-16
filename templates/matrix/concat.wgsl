
{% extends "base.wgsl" %}
{% block structs %}{% include "struct-ArrayVector.wgsl" %}{% endblock structs %}
{% block main %}
[[stage(compute), workgroup_size(1)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
    let gidx = global_id.x;
    if (gidx < {{ len_0 / 4 | int }}u) {
	_{{ output[0] }}.data[gidx] = _{{ input[0] }}.data[gidx];
    } else {
    	_{{ output[0] }}.data[gidx] = _{{ input[1] }}.data[gidx];
    }
}
{% endblock main %}