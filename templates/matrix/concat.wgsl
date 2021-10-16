
{% include "structs.wgsl" %}

{% for binding in bindings %}
[[group(0), binding({{ binding.counter }})]]
var<storage, read_write> _{{ binding.tensor }}: Array;
{% endfor %}

[[stage(compute), workgroup_size(1)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
    let gidx = global_id.x;
    if (gidx < {{ len_0 }}u) {
	    _{{ output[0] }}.data[gidx] = _{{ input[0] }}.data[gidx];
    } else {
    	_{{ output[0] }}.data[gidx] = _{{ input[1] }}.data[gidx - {{ len_0 }}u];
    }
}
