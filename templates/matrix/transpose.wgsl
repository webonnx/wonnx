{% extends "base.wgsl" %}
{% block structs %}{% include "struct-ArrayVector.wgsl" %}{% endblock structs %}
{% block main %}
[[stage(compute), workgroup_size(1)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
        let y = global_id.x % {{ len_1 }}u;
        let x = global_id.x / {{ len_1 }}u;
        let index = x * {{ len_1 * 4 }}u + y; 
        
        let tmpMat_{{ input[0] }} = transpose(mat4x4<f32>({{ input[0] }}.data[index], 
                            {{ input[0] }}.data[index + {{ len_1 }}u],
                            {{ input[0] }}.data[index + 2u * {{ len_1 }}u],
                            {{ input[0] }}.data[index + 3u * {{ len_1 }}u],
                        ));

        let index = y * {{ len_0 }}u + x;

        {{ output[0] }}.data[index] = tmpMat_{{ input[0] }}[0u];
        {{ output[0] }}.data[index + {{ len_0 / 4 }}u] = tmpMat_{{ input[0] }}[1u];
        {{ output[0] }}.data[index + 2u * {{ len_0 / 4 }}u] = tmpMat_{{ input[0] }}[2u];
        {{ output[0] }}.data[index + 3u * {{ len_0 / 4 }}u] = tmpMat_{{ input[0] }}[3u];
}
{% endblock main %}