type Scalar = {{ scalar_type }};
type GemmVec = vec{{ kernel_size }}<{{ scalar_type }}>;
type GemmMat = mat{{ kernel_size }}x{{ kernel_size }}<{{ scalar_type }}>;

struct GemmArrayVector {
	data: array<GemmVec>;
};

[[group(0), binding(0)]]
var<storage, read> input_0: GemmArrayVector;

[[group(0), binding(1)]]
var<storage, read> input_1: GemmArrayVector;

{% if i_lens | length == 3 %} // Bias
	[[group(0), binding(2)]]
	var<storage, read> input_2: GemmArrayVector;

	[[group(0), binding(3)]]
	var<storage, write> output_0: GemmArrayVector;
{% else %}
	[[group(0), binding(2)]]
	var<storage, write> output_0: GemmArrayVector;
{% endif %}

[[stage(compute), workgroup_size({{ workgroup_size_x }})]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
	let y = global_id.x % {{ n_chunks }}u;
	let x = global_id.x / {{ n_chunks }}u;
	let index = x * {{ i_shape[1][1] }}u + y;

	// Create zero vector and matrix of the correct size for later use
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
		let index_left = (x * {{ i_shape[0][1] }}u) + k;
		let index_right = k * {{ i_shape[1][1] }}u + y;

		let mat_left = GemmMat(
			{% for i in range(end = kernel_size) %}
				input_0.data[index_left + ({{ i }}u * {{ k_chunks }}u)] {%-if not loop.last -%},{%- endif -%}
			{% endfor %}
		);
		
		let mat_right = GemmMat(
			{% for i in range(end = kernel_size) %}
				input_1.data[index_right + ({{ i * n_chunks }}u)] {%-if not loop.last -%},{%- endif -%}
			{% endfor %}
		);
	
		product = mat_right * mat_left;
	
		for(var index_mat: u32 = 0u; index_mat < {{ kernel_size }}u; index_mat = index_mat + 1u) {
			tmpsum[index_mat] = tmpsum[index_mat] + product[index_mat];
		}
	}
	
	{% if i_lens | length == 3 %}
		let bias_row = input_2.data[x]; 
		var bias = transpose(GemmMat(
			{% for i in range(end=kernel_size) %}
				bias_row {%-if not loop.last -%},{%- endif -%}
			{% endfor %}
		));
		for(var index_mat: u32 = 0u; index_mat < {{ kernel_size }}u; index_mat = index_mat + 1u) {
			output_0.data[index + index_mat * {{ n_chunks }}u] = 
				{%- if alpha != 1 -%} {{ alpha | float }} * {%- endif -%} 
				tmpsum[index_mat] + 
				{%- if beta != 1 -%} {{ beta | float }} * {%- endif -%} 
				bias[index_mat]
			;
		}
	{% else %}
		for(var index_mat: u32 = 0u; index_mat < {{ kernel_size }}u; index_mat = index_mat + 1u) {
			output_0.data[index + (index_mat * {{ n_chunks }}u)] = {% if alpha != 1 -%} {{ alpha | float }} * {%- endif -%} tmpsum[index_mat];
		}
	{% endif %}
}