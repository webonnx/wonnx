{%- include "structs.wgsl" -%}

[[group(0), binding(0)]]
var<storage, read> var_{{ inputs[0] }}: ArrayVector;

[[group(0), binding(1)]]
var<storage, write> var_{{ outputs[0] }}: ArrayVector;

[[stage(compute), workgroup_size(1)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
    let gidx = global_id.x;

{%- if op_type is matching("Relu") -%}
    var_{{ outputs[0] }}.data[gidx] = max(var_{{ inputs[0] }}.data[gidx], vec4<f32>(0.0, 0.0, 0.0, 0.0));

{%- elif op_type is matching("Sigmoid") -%}
    var_{{ outputs[0] }}.data[gidx] = vec4<f32>(1.0, 1.0, 1.0, 1.0) / (vec4<f32>(1.0, 1.0, 1.0, 1.0) + exp(-var_{{ inputs[0] }}.data[gidx]));

{%- elif op_type is matching("Softsign") -%}
    let input = var_{{ inputs[0] }}.data[gidx]; 
    var_{{ outputs[0] }}.data[gidx] = input / (vec4<f32>(1.0, 1.0, 1.0, 1.0) + abs(input));

{%- elif op_type is matching("Softplus") -%}
    var_{{ outputs[0] }}.data[gidx] = log(vec4<f32>(1.0, 1.0, 1.0, 1.0) + exp(var_{{ inputs[0] }}.data[gidx]));

{%- elif op_type is matching("Clip") -%}
    let min_clip = var_{{ inputs[1] }}.data[0u];
    let max_clip = var_{{ inputs[2] }}.data[0u];
    var_{{ outputs[0] }}.data[gidx] = clamp(
        {inputs[0]}.data[gidx], 
        vec4<f32>(min_clip, min_clip, min_clip, min_clip),
        vec4<f32>(max_clip, max_clip, max_clip, max_clip),
    );

{%- elif op_type is matching("Celu") -%}
    let input_vec = var_{{ inputs[0] }}.data[gidx]; 
    var_{{ outputs[0] }}.data[gidx] = max(
            vec4<f32>(0.0, 0.0, 0.0, 0.0), 
            input_vec
        ) + min(
            vec4<f32>(0.0, 0.0, 0.0, 0.0), 
            {{ alpha }} * (exp(input_vec / {{ alpha }}) - vec4<f32>(1.0, 1.0, 1.0, 1.0))
        );

{%- elif op_type is matching("Elu") -%}
        let input_vec = var_{{ inputs[0] }}.data[gidx]; 
        var_{{ output[0] }}.data[gidx] = max(
            vec4<f32>(0.0, 0.0, 0.0, 0.0), 
            input_vec
        ) + min(
            vec4<f32>(0.0, 0.0, 0.0, 0.0), 
            {{ alpha }} * (exp(input_vec) - vec4<f32>(1.0, 1.0, 1.0, 1.0))
        );
{%- endif -%}
}