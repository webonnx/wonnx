
{%- include "structs.wgsl" -%}


{% for input in inputs %}
[[group({{ loop.index0 / 4 | int }}), binding({{ loop.index0 % 4}})]]
var<storage, read> {{ input }}: Array;
{% endfor %}

{% set binding_len = inputs | length %}
[[group({{ binding_len  / 4 | int }}), binding({{ binding_len % 4 }})]]
var<storage, write> {{ outputs[0] }}: Array;

// concat.wgsl
[[stage(compute), workgroup_size(256, 1, 1)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
    let gidx = global_id.x;
{% for input in inputs %}
{% if loop.first %}
    if (gidx < {{ i_lens[0] }}u) {
	    {{ outputs[0] }}.data[gidx] = {{ inputs[0] }}.data[gidx];
    }
{% else %}
	if ((gidx >= {{ cum_len | nth(n=loop.index0 -1) }}u) && (gidx < {{ cum_len | nth(n=loop.index0)}}u)) {
	    {{ outputs[0] }}.data[gidx] = {{ input }}.data[gidx - {{ cum_len | nth(n=loop.index0 -1) }}u];
    }
{% endif %}
{% endfor %}
}
