{%- include "structs.wgsl" -%}

[[group(0), binding(0)]]
var<storage, read> {{ inputs[0] }}: Array;

[[group(0), binding(1)]]
var<storage, write> {{ outputs[0] }}: Array;


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

        let index = {%- for perm in permuted_chunks -%}
        {% if not loop.first %}
        +
        {% endif %}
        d_{{ loop.index0 }} * {{ perm }}u
        {%- endfor %};

        {{ outputs[0] }}.data[index] = {{ inputs[0] }}.data[gidx];
    }
}