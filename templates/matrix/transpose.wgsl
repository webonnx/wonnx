{%- include "structs.wgsl" -%}

[[group(0), binding(0)]]
var<storage, read> var_{{ inputs[0] }}: Array;

[[group(0), binding(1)]]
var<storage, write> var_{{ outputs[0] }}: Array;


[[stage(compute), workgroup_size(256, 1, 1)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
    let gidx = global_id.x;

    if (gidx < {{ o_lens[0] }}u) {

        var rest = gidx;
        {%- for chunks in i_chunks_0 -%}
        {%- if loop.last -%}
        let d_{{ loop.index0 }} = rest; 
        {%- else -%}
        let d_{{ loop.index0 }} = rest / {{ chunks }}u; 
        rest = gidx % {{ chunks }}u; 
        {%- endif -%}{%- endfor -%}

        let index = {%- for perm in permuted_chunks -%}{%- if loop.first -%}
        d_{{ loop.index0 }} * {{ perm }}u
        {%- else -%}
        + d_{{ loop.index0 }} * {{ perm }}u
        {%- endif -%}{%- endfor -%};

        var_{{ outputs[0] }}.data[index] = var_{{ inputs[0] }}.data[gidx];
    }
}