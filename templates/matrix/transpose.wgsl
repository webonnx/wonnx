{% extends "base.wgsl" %}
{% block structs %}{% include "struct-ArrayVector.wgsl" %}{% endblock structs %}
{% block main %}
[[stage(compute), workgroup_size(1)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
        let y = global_id.x % {{ len_1 }}u;
        let x = global_id.x / {{ len_1 }}u;
        let index = x * {{ len_1 * 4 }}u + y; 
        
        let tmpMatvar_{{ input[0] }} = transpose(mat4x4<f32>(var_{{ input[0] }}.data[index], 
                            var_{{ input[0] }}.data[index + {{ len_1 }}u],
                            var_{{ input[0] }}.data[index + {{ 2 * len_1 }}u],
                            var_{{ input[0] }}.data[index + {{ 3 * len_1 }}u],
                        ));

        let index = y * {{ len_0 }}u + x;

        var_{{ output[0] }}.data[index] = tmpMatvar_{{ input[0] }}[0u];
        var_{{ output[0] }}.data[index + {{ len_0 / 4 | int }}u] = tmpMatvar_{{ input[0] }}[1u];
        var_{{ output[0] }}.data[index + {{ 2 * len_0 / 4 | int }}u] = tmpMatvar_{{ input[0] }}[2u];
        var_{{ output[0] }}.data[index + {{ 3 * len_0 / 4 | int }}u] = tmpMatvar_{{ input[0] }}[3u];
}
{% endblock main %}