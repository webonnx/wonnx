{% include "structs.wgsl" %}

@group(0) @binding(0)
var<storage, read> input_0: Array;

@group(0) @binding(1)
var<storage, read_write> output_0: Array;

@compute @workgroup_size({{ workgroup_size_x }}, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
	let gidx = global_id.x;

	{# We will be invoked once for each scalar in the output (output_0.data[gidx]) which represents one reduce operation.
	Find out which output cell we are calculating (d_X refers to the index of dimension X in the input) #}
	if (gidx < {{ o_lens[0] }}u) {
		var rest = gidx;

		{#- chunks_with_dims_preserved is the input dims, but the reduced axes have their size set to 1, e.g. an input 
		tensor of dimension [3,3,2] with axes=[1] will have chunks_with_dims_preserved=[3,1,2] (This is equal to the 
		shape of the output when keepdims = true) -#}

		{% for chunks in chunks_with_dims_preserved %}
			{% if loop.last %}
				{% if not loop.index0 in axes %}
					let d_{{ loop.index0 }}: u32 = rest; 
				{% endif %}
			{% else %}
				{% if not loop.index0 in axes %}
					let d_{{ loop.index0 }}: u32 = rest / {{ chunks }}u; 
				{% endif %}
				{% if chunks != 1 %}
				rest = gidx % {{ chunks }}u; 
				{% endif %}
			{% endif %}
		{% endfor %}

		{#- At this point we have d_* variables set to the indexes on the fixed (non reduced) axes. In the above example,
		the base indexes [d_0,d_1,d_2] will be [0,0,0], [0,0,1], [0,0,2], [1,0,0], [1,0,1], ..  
		
		Now for each reduced axis, iterate all values and reduce. Note, starting value may not always be zero. For 
		ReduceMin/Max we should initialize as NaN and keep a flag to check if we have seen at least one element -#}

		var accumulator = {% if op_type == "ReduceProd" %} {{ scalar_type }}(1) {% else %} Scalar() {% endif %};
	    var max_element: Scalar = log(Scalar());

		var count = 0u;

		{% for reducing_axis in axes %}
			for(var d_{{reducing_axis}} = 0u; d_{{reducing_axis}} < {{i_shape[0][reducing_axis]}}u; d_{{reducing_axis}} = d_{{reducing_axis}} + 1u) {
		{% endfor %}

				let input_val = input_0.data[
					{% for _axis in i_shape[0] %}
						(d_{{loop.index0}} * {{i_chunks[0][loop.index0]}}u) {% if not loop.last %} + {% endif %}
					{% endfor %}
				];

				{% if op_type == "ReduceMean" or op_type == "ReduceSum" %}
					accumulator = accumulator + input_val;
				{% elif op_type == "ReduceL1" %}
					accumulator = accumulator + abs(input_val);
				{% elif op_type == "ReduceL2" or op_type == "ReduceSumSquare" %}
					accumulator = accumulator + (input_val * input_val);
				{% elif op_type == "ReduceLogSum" %}
					accumulator = accumulator + input_val;
				{% elif op_type == "ReduceLogSumExp" %}
					accumulator = accumulator + exp(input_val);
				{% elif op_type == "ReduceProd" %}
					accumulator = accumulator * input_val;
				{% elif op_type == "ReduceMin" %}
					if(count == 0u) {
						accumulator = input_val;
					}
					else if(accumulator > input_val) {
						accumulator = input_val;
					}
				{% elif op_type == "ReduceMax" %}
					if(count == 0u) {
						accumulator = input_val;
					}
					else if(accumulator < input_val) {
						accumulator = input_val;
					}
				{% elif op_type == "ArgMax" %}
					if(input_val > max_element) {
                        max_element = input_val;
                        accumulator = f32(count);
                    }
				{% endif %}

				count = count + 1u;

		{% for reducing_axis in axes %}
			}
		{% endfor %}

		{#- Post-processing -#}
		{% if op_type == "ReduceMean" %}
			accumulator = accumulator / {{ scalar_type }}(count);
		{% elif op_type == "ReduceL2" %}
			accumulator = sqrt(accumulator);
		{% elif op_type == "ReduceLogSum" or op_type == "ReduceLogSumExp" %}
			accumulator = log(accumulator);
		{% endif %}

		output_0.data[gidx] = accumulator;
	}
}