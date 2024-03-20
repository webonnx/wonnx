
{%- include "structs.wgsl" -%}

@group(0) @binding(0)
var<storage, read> input_0: Array;

@group(0) @binding(1)
var<storage, read_write> output_0: Array;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
	let gidx = global_id.x;

	if (gidx < {{ o_lens[0] }}u) {
		var rest = gidx;

		rest = gidx;
		{%- for chunks in o_chunks[0] -%}
			{% if loop.last %}
				let d_{{ loop.index0 }} = rest; 
			{% else %}
				let d_{{ loop.index0 }} = rest / {{ chunks }}u; 
				rest = gidx % {{ chunks }}u; 
			{% endif %}
		{%- endfor -%}

		var pad = false;
		{% for pad in pad_info %}
                        var id_{{ loop.index0 }} = 0u;

			if (d_{{ loop.index0 }} < {{ pad.copy_start }}u) {
                                {% if mode == "reflect" %}
                                        id_{{ loop.index0 }} = ({{ pad.copy_start }}u - d_{{ loop.index0 }}) % {{ i_shape[0][loop.index0] }}u;
                                {% else %}
                                        id_{{ loop.index0 }} = d_{{ loop.index0 }} - {{ pad.copy_start }}u;
                                        pad = true;
                                {% endif %}
			}
			else if (d_{{ loop.index0 }} > {{ pad.end_pad_start }}u) {
                                {% if mode == "reflect" %}
                                        id_{{ loop.index0 }} = 2u * {{ pad.end_pad_start }}u - d_{{ loop.index0 }};
                                {% else %}
                                        id_{{ loop.index0 }} = d_{{ loop.index0 }} - {{ pad.copy_start }}u;
                                        pad = true;
                                {% endif %}
			} else {
                                id_{{ loop.index0 }} = d_{{ loop.index0 }} - {{ pad.copy_start }}u;
                        }
		{% endfor %}

		if (pad) {
                        output_0.data[gidx] = {{ scalar_type }}({{ constant_value }});
		} else {
			let index = 
				{%- for chunk in i_chunks | first -%}
					{%- if not loop.first %}
						+
					{%- endif -%}
					id_{{ loop.index0 }} * {{ chunk }}u
				{%- endfor -%}
			;

			output_0.data[gidx] = input_0.data[index];
		}
	}
}
