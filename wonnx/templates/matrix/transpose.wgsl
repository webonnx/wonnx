{%- include "structs.wgsl" -%}

[[group(0), binding(0)]]
var<storage, read> input_0: Array;

[[group(0), binding(1)]]
var<storage, write> output_0: Array;

[[stage(compute), workgroup_size(256, 1, 1)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
	let gidx = global_id.x;

	if (gidx < {{ o_lens[0] }}u) {
		var rest = gidx;

		{%- for chunks in o_chunks[0] -%}
			{% if loop.last %}
				let d_{{ loop.index0 }} = rest; 
			{% else %}
				let d_{{ loop.index0 }} = rest / {{ chunks }}u; 
				rest = gidx % {{ chunks }}u; 
			{% endif %}
		{%- endfor -%}

		let index = 
			{%- for perm in permuted_chunks -%}
				{% if not loop.first %}
					+
				{% endif %}
				
				d_{{ loop.index0 }} * {{ perm }}u
			{%- endfor %}
		;

		output_0.data[gidx] = input_0.data[index];
	}
}