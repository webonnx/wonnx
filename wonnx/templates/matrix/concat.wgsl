
{%- include "structs.wgsl" -%}

{% for input in i_lens %}

@group({{ loop.index0 / 4 | int }}) @binding({{ loop.index0 % 4}})
var<storage, read> input_{{ loop.index0 }}: Array;

{% endfor %}

{% set binding_len = i_lens | length %}
@group({{ binding_len  / 4 | int }}) @binding({{ binding_len % 4 }})
var<storage, read_write> output_0: Array;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
	let gidx = global_id.x;
        let gidy = global_id.y;

        let x_executions = num_workgroups.x * 16u;

        let actual_idx = gidx + gidy * x_executions;
	
	{% for input in i_lens %}
		{% if loop.first %}
			if (actual_idx < {{ i_lens[0] }}u) {
				output_0.data[actual_idx] = input_0.data[actual_idx];
			}

		{% else %}
			if ((actual_idx >= {{ cum_len | nth(n=loop.index0 -1) }}u) && (actual_idx < {{ cum_len | nth(n=loop.index0)}}u)) {
				output_0.data[actual_idx] = input_{{ loop.index0 }}.data[actual_idx - {{ cum_len | nth(n=loop.index0 -1) }}u];
			}
			
		{% endif %}
	{% endfor %}
}
