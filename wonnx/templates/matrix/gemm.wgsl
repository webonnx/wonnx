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

[[stage(compute), workgroup_size(1)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
	{% if i_shape[0][1] < kernel_size %}
		let n_k = 1u;
	{% else %}
		let n_k = {{ i_shape[0][1] / kernel_size | int }}u;
	{% endif %}

	{% if i_shape[1][1] < kernel_size %}
		let n_m = 1u;
	{% else %}
		let n_m = {{ i_shape[1][1] / kernel_size | int }}u;
	{% endif %}

	let y = global_id.x % n_m;
	let x = global_id.x / n_m;
	let index = x * {{ i_shape[1][1] }}u + y;

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

	for(var k: u32 = 0u; k < n_k; k = k + 1u) {
		let index_left = x * n_k + k;
		let index_right = k * n_m + y;

		let mat_left = GemmMat(
			{% for i in range(end = kernel_size) %}
				input_0.data[index_left + ({{ i }}u * n_k)] {%-if not loop.last -%},{%- endif -%}
			{% endfor %}
		);
		
		let mat_right = GemmMat(
			{% for i in range(end = kernel_size) %}
				input_1.data[index_right + ({{ i }}u * n_m)] {%-if not loop.last -%},{%- endif -%}
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
			output_0.data[index + index_mat * n_m] = 
				{%- if alpha != 1 -%} {{ alpha | float }} * {%- endif -%} 
				tmpsum[index_mat] + 
				{%- if beta != 1 -%} {{ beta | float }} * {%- endif -%} 
				bias[index_mat]
			;
		}
	{% else %}
		for(var index_mat: u32 = 0u; index_mat < {{ kernel_size }}u; index_mat = index_mat + 1u) {
			output_0.data[index + (index_mat * n_m)] = {% if alpha != 1 -%} {{ alpha | float }} * {%- endif -%} tmpsum[index_mat];
		}
	{% endif %}
}