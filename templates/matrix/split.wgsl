
{%- include "structs.wgsl" -%}


[[group(0), binding(0)]]
var<storage, read> {{ inputs[0] }}: Array;


{% for output in outputs %}
[[group({{ loop.index / 4 | int }}), binding({{ loop.index % 4}})]]
var<storage, write> {{ output }}: Array;
{% endfor %}

// concat.wgsl
[[stage(compute), workgroup_size(256, 1, 1)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
    let gidx = global_id.x;

    if (gidx < {{ i_lens[0] }}u) {

        var rest = gidx;
        {%- for chunks in i_chunks[0] -%}
        {% if loop.last %}
        let d_{{ loop.index0 }} = rest; 
        {% else %}
        let d_{{ loop.index0 }} = rest / {{ chunks }}u; 
        rest = gidx % {{ chunks }}u; 
        {% endif %}
        {%- endfor -%}

{% for output in outputs %}
{%- if loop.first %}
    if (d_{{ axis }} < {{ split | first }}u) {
	    {{ output }}.data[gidx] = {{ inputs[0] }}.data[gidx];
    }
{%- else %}
	if ((d_{{ axis }} >= {{ split | nth(n=loop.index0 -1) }}u) && (d_{{ axis }} < {{ split | nth(n=loop.index0)}}u)) {
	    {{ output }}.data[gidx - {{ split | nth(n=loop.index0 -1) * i_chunks[0] | nth(n=axis) }}u] = {{ inputs[0] }}.data[gidx];
    }
{% endif %}
{% endfor %}
    }
}
