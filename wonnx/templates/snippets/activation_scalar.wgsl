{%- if activation_type == "Relu" -%}
	{{ activation_output }} = max({{ activation_input }}, Scalar());

{%- elif activation_type == "Sigmoid" -%}
	{{ activation_output }} = {{ scalar_type }}(1) / ({{ scalar_type }}(1) + exp(-{{ activation_input }}));

{%- elif activation_type == "Softsign" -%}
	let input = {{ activation_input }}; 
	{{ activation_output }} = input / ({{ scalar_type }}(1) + abs(input));

{%- elif activation_type == "Softplus" -%}
	{{ activation_output }} = log({{ scalar_type }}(1) + exp({{ activation_input }}));

{%- elif activation_type == "Clip" -%}
	{{ activation_output }} = clamp(
		{{ activation_input }}, 
		{{ min }},
		{{ max }},
	);

{%- elif activation_type == "Celu" -%}
	let input_vec = {{ activation_input }};

	{{ activation_output }} = max(Scalar(), 
			input_vec
		) + min(
			Scalar(), 
			{{ scalar_type }}({{ alpha }}) * (exp(input_vec / {{ scalar_type }}({{ alpha }})) - {{ scalar_type }}(1))
		);

{%- elif activation_type == "Elu" -%}
		let input_vec = {{ activation_input }}; 
		{{ activation_output }} = max(
			Scalar(), 
			input_vec
		) + min(
			Scalar(), 
			{{ scalar_type }}({{ alpha }}) * (exp(input_vec) - {{ scalar_type }}(1))
		);

{%- elif activation_type == "HardSigmoid" -%}
	{{ activation_output }} = max(
		{{ scalar_type }}(0),
		min(
			{{ scalar_type }}(1),
			{{ scalar_type }}({{ alpha }}) * {{ activation_input }} + {{ scalar_type }}({{ beta }})
		)
	);

{%- elif activation_output != activation_input -%}
	{{ activation_output }} = {{ activation_input }};

{%- endif -%}
