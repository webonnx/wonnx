
{%- include "structs.wgsl" -%}

[[group(0), binding(0)]]
var<storage, read> {{ inputs[0] }}: Array;


[[group(0), binding(1)]]
var<storage, write> {{ outputs[0] }}: Array;

// resize.wgsl
[[stage(compute), workgroup_size(256, 1, 1)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
    let gidx = global_id.x;

    if (gidx < {{ o_lens[0] }}u) {

        var rest = gidx;
        {%- for chunk in o_chunks[0] -%}
        {% if loop.last %}
        let d_{{ loop.index0 }} = rest; 
        {% else %}
        let d_{{ loop.index0 }} = rest / {{ chunk }}u; 
        rest = gidx % {{ chunk }}u; 
        {% endif %}
        {%- endfor %}


        let index = {%- for chunks in i_chunks[0] -%}
				{% set scale = scales | nth(n=loop.index0) %}
        {%- if not loop.first %}
				+ 
				{%- endif -%}
        u32(floor(
				  (f32(d_{{ loop.index0 }}) + 0.5) / {{ scale }} - 0.5 
			  )) * {{ chunks  }}u 
        {%- endfor -%};

	    {{ outputs[0] }}.data[gidx] = {{ inputs[0] }}.data[index];
    }
}
