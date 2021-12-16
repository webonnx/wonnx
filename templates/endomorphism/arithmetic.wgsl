{%- include "structs.wgsl" -%}
[[group(0), binding(0)]]
var<storage, read> {{ inputs[0] }}: ArrayVector;

{% if inputs | length == 2 %}
[[group(0), binding(1)]]
var<storage, read> {{ inputs[1] }}: ArrayVector;

[[group(0), binding(2)]]
var<storage, write> {{ outputs[0] }}: ArrayVector;

{% else %}

[[group(0), binding(1)]]
var<storage, write> {{ outputs[0] }}: ArrayVector;

{% endif %}
[[stage(compute), workgroup_size(256, 1, 1)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
    let gidx = global_id.x;
{% if inputs | length == 2 %}
    {{ outputs[0] }}.data[gidx] = {{ inputs[0] }}.data[gidx] {{ op_type }} {{ inputs[1] }}.data[gidx];
{% else %}
    {{ outputs[0] }}.data[gidx] = {{ inputs[0] }}.data[gidx] {{ op_type }} vec4<f32>({{ coefficient }}., {{ coefficient }}., {{ coefficient }}., {{ coefficient }}.);
{% endif %}
}