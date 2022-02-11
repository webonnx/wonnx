{%- if activation_type == "Relu" -%}
	{{ activation_output }} = max({{ activation_input }}, Scalar(0));

{%- elif activation_type == "Sigmoid" -%}
	{{ activation_output }} = Scalar(1) / (Scalar(1) + exp(-{{ activation_input }}));

{%- elif activation_type == "Softsign" -%}
	let input = {{ activation_input }}; 
	{{ activation_output }} = input / (Scalar(1) + abs(input));

{%- elif activation_type == "Softplus" -%}
	{{ activation_output }} = log(Scalar(1) + exp({{ activation_input }}));

{%- elif activation_type == "Clip" -%}
	let min_clip = input_1.data[0u];
	let max_clip = input_2.data[0u];

	{{ activation_output }} = clamp(
		{{ activation_input }}, 
		Vec4(min_clip, min_clip, min_clip, min_clip),
		Vec4(max_clip, max_clip, max_clip, max_clip),
	);

{%- elif activation_type == "Celu" -%}
	let input_vec = {{ activation_input }};

	{{ activation_output }} = max(Scalar(0), 
			input_vec
		) + min(
			Scalar(0), 
			{{ alpha }} * (exp(input_vec / {{ alpha }}) - Scalar(1))
		);

{%- elif activation_type == "Elu" -%}
		let input_vec = {{ activation_input }}; 
		{{ activation_output }} = max(
			Scalar(0), 
			input_vec
		) + min(
			Scalar(0), 
			{{ alpha }} * (exp(input_vec) - Scalar(1))
		);

{%- elif activation_output != activation_input -%}
	{{ activation_output }} = {{ activation_input }};

{%- endif -%}