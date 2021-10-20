{% extends "base.wgsl" %}
{% block structs %}{% include "structs.wgsl" %}{% endblock structs %}
{% block main %}
[[stage(compute), workgroup_size(1)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
    let gidx = global_id.x;

{% if op_type is matching("relu") %}
    var_{{ output[0] }}.data[gidx] = max(var_{{ input[0] }}.data[gidx], vec4<f32>(0.0, 0.0, 0.0, 0.0));

{% elif op_type is matching("sigmoid") %}
    var_{{ output[0] }}.data[gidx] = vec4<f32>(1.0, 1.0, 1.0, 1.0) / (vec4<f32>(1.0, 1.0, 1.0, 1.0) + exp(-var_{{ input[0] }}.data[gidx]));

{% elif op_type is matching("softsign") %}
    let input = var_{{ input[0] }}.data[gidx]; 
    var_{{ output[0] }}.data[gidx] = input / (vec4<f32>(1.0, 1.0, 1.0, 1.0) + abs(input));

{% elif op_type is matching("softplus") %}
    var_{{ output[0] }}.data[gidx] = log(vec4<f32>(1.0, 1.0, 1.0, 1.0) + exp(var_{{ input[0] }}.data[gidx]));

{% elif op_type is matching("clip") %}
    let min_clip = var_{{ input[1] }}.data[0u];
    let max_clip = var_{{ input[2] }}.data[0u];
    var_{{ output[0] }}.data[gidx] = clamp(
        {input[0]}.data[gidx], 
        vec4<f32>(min_clip, min_clip, min_clip, min_clip),
        vec4<f32>(max_clip, max_clip, max_clip, max_clip),
    );

{% elif op_type is matching("celu") %}
    let input_vec = var_{{ input[0] }}.data[gidx]; 
    var_{{ output[0] }}.data[gidx] = max(
            vec4<f32>(0.0, 0.0, 0.0, 0.0), 
            input_vec
        ) + min(
            vec4<f32>(0.0, 0.0, 0.0, 0.0), 
            {{ alpha }} * (exp(input_vec / {{ alpha }}) - vec4<f32>(1.0, 1.0, 1.0, 1.0))
        );

{% elif op_type is matching("elu") %}
        let input_vec = var_{{ input[0] }}.data[gidx]; 
        var_{{ output[0] }}.data[gidx] = max(
            vec4<f32>(0.0, 0.0, 0.0, 0.0), 
            input_vec
        ) + min(
            vec4<f32>(0.0, 0.0, 0.0, 0.0), 
            {{ alpha }} * (exp(input_vec) - vec4<f32>(1.0, 1.0, 1.0, 1.0))
        );
{% endif %}
}
{% endblock main %}