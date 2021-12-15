
{%- include "structs.wgsl" -%}


[[group(0), binding(0)]]
var<storage, read> {{ inputs[0] }}: Array;


{% for output in outputs %}
[[group({{ loop.index / 4 | int }}), binding({{ loop.index % 4}})]]
var<storage, write> {{ output }}: Array;
{% endfor %}

// split.wgsl
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

        let index = {%- for chunk in o_chunks | first -%}
        {%- if not loop.first %}
        +
        {%- endif -%}
        d_{{ loop.index0 }} * {{ chunk }}u
        {%- endfor -%};

	    {{ output }}.data[index] = {{ inputs[0] }}.data[gidx];
    }
{%- else %}
{% set split_output = split | nth(n=loop.index0 -1) %}
	if ((d_{{ axis }} >= {{ split_output }}u) && (d_{{ axis }} < {{ split | nth(n=loop.index0)}}u)) {

        let index = {%- for chunk in o_chunks | nth(n=loop.index0) -%}
        {%- if not loop.first %}
        +
        {%- endif -%}
        {%- if loop.index0 == axis %}
        (d_{{ loop.index0 }} - {{ split_output }}u) * {{ chunk }}u
        {% else %}
        d_{{ loop.index0 }} * {{ chunk }}u
        {%- endif -%}
        {%- endfor -%};

	    {{ output }}.data[index] = {{ inputs[0] }}.data[gidx];
    }
{% endif %}
{% endfor %}
    }
}
