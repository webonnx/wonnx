type Scalar = {{ scalar_type }};
type GemmVec = vec{{ kernel_size }}<{{ scalar_type }}>;
type GemmMat = mat{{ kernel_size }}x{{ kernel_size }}<{{ scalar_type }}>;

struct GemmArrayVector {
	data: array<GemmVec>;
};

[[group(0), binding(0)]]
var<storage, read> input_left: GemmArrayVector;

[[group(0), binding(1)]]
var<storage, read> input_right: GemmArrayVector;

{% if i_lens | length == 3 %} // Bias
	[[group(0), binding(2)]]
	var<storage, read> input_2: GemmArrayVector;

	[[group(0), binding(3)]]
	var<storage, write> output_0: GemmArrayVector;
{% else %}
	[[group(0), binding(2)]]
	var<storage, write> output_0: GemmArrayVector;
{% endif %}

[[stage(compute), workgroup_size({{ workgroup_size_x }}, {{ workgroup_size_y }})]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
	let y = global_id.x % {{ n_chunks }}u;
	let x = global_id.x / {{ n_chunks }}u;

	{# Calculate stacking offsets #}
	let stack_index = global_id.y;
	let left_offset = stack_index * {{ stack_left_stride / kernel_size }}u;
	let right_offset = stack_index * {{ stack_right_stride / kernel_size }}u;
	let output_offset = stack_index * {{ stack_output_stride / kernel_size }}u;

	let index = output_offset + (x * {{ right_shape[1] }}u) + y;

	{# Create zero vector and matrix of the correct size for later use #}
	let zero_vec = GemmVec(
		{% for i in range(end = kernel_size) %}
			Scalar(0) {%-if not loop.last -%},{%- endif -%}
		{% endfor %}
	);

	let zero_matrix = GemmMat(
		{% for i in range(end = kernel_size) %}
			zero_vec {%-if not loop.last -%},{%- endif -%}
		{% endfor %}
	);

	var tmpsum = zero_matrix;
	var product = zero_matrix;

	for(var k: u32 = 0u; k < {{ k_chunks }}u; k = k + 1u) {
		let index_left = left_offset + (x * {{ left_shape[1] }}u) + k;
		let index_right = right_offset + (k * {{ right_shape[1] }}u) + y;

		let mat_left = GemmMat(
			{% for i in range(end = kernel_size) %}
				input_left.data[index_left + {{ i * k_chunks }}u] {%-if not loop.last -%},{%- endif -%}
			{% endfor %}
		);
		
		let mat_right = GemmMat(
			{% for i in range(end = kernel_size) %}
				input_right.data[index_right + ({{ i * n_chunks }}u)] {%-if not loop.last -%},{%- endif -%}
			{% endfor %}
		);
	
		product = mat_right * mat_left;
	
		for(var index_mat: u32 = 0u; index_mat < {{ kernel_size }}u; index_mat = index_mat + 1u) {
			tmpsum[index_mat] = tmpsum[index_mat] + product[index_mat];
		}
	}
	
	{% if i_lens | length == 3 %}
		let bias_index =
			{% if not bias_broadcast_x  %} (x * {{ bias_shape[1] }}u) + {% endif %} 
			{% if not bias_broadcast_y %} y {% else  %} 0u {% endif %};

		for(var index_mat: u32 = 0u; index_mat < {{ kernel_size }}u; index_mat = index_mat + 1u) {
			let bias = input_2.data[bias_index {% if not bias_broadcast_x %} +(index_mat * {{ n_chunks }}u) {% endif %}];

			output_0.data[index + (index_mat * {{ n_chunks }}u)] = 
				{%- if alpha != 1 -%} {{ alpha | float }} * {%- endif -%} 
				tmpsum[index_mat] + 
				{%- if beta != 1 -%} {{ beta | float }} * {%- endif -%} 
				bias;
		}
	{% else %}
		for(var index_mat: u32 = 0u; index_mat < {{ kernel_size }}u; index_mat = index_mat + 1u) {
			output_0.data[index + (index_mat * {{ n_chunks }}u)] = {% if alpha != 1 -%} {{ alpha | float }} * {%- endif -%} tmpsum[index_mat];
		}
	{% endif %}
}