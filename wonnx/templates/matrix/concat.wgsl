
{%- include "structs.wgsl" -%}

{% for input in i_lens %}

@group({{ loop.index0 / 4 | int }}) @binding({{ loop.index0 % 4}})
var<storage, read> input_{{ loop.index0 }}: Array;

{% endfor %}

{% set binding_len = i_lens | length %}
@group({{ binding_len  / 4 | int }}) @binding({{ binding_len % 4 }})
var<storage, write> output_0: Array;

@stage(compute) @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
	let gidx = global_id.x;
	
	{% for input in i_lens %}
		{% if loop.first %}
			if (gidx < {{ i_lens[0] }}u) {
				output_0.data[gidx] = input_0.data[gidx];
			}

		{% else %}
			if ((gidx >= {{ cum_len | nth(n=loop.index0 -1) }}u) && (gidx < {{ cum_len | nth(n=loop.index0)}}u)) {
				output_0.data[gidx] = input_{{ loop.index0 }}.data[gidx - {{ cum_len | nth(n=loop.index0 -1) }}u];
			}
			
		{% endif %}
	{% endfor %}
}
