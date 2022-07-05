{# 
// Operations usually work with a single scalar data type (typically f32). This data type is set by the compiler as the
// 'scalar_type' variable. Here we define several other useful data types that shader code can use to make it more portable.
#}
type Scalar = {{ scalar_type }};
type Vec3 = vec3<{{ scalar_type }}>;
type Vec4 = vec4<{{ scalar_type }}>;

struct Array {
	data: array<Scalar>
};

struct ArrayVector {
	data: array<Vec4>
};

{# 
// WGSL only supports matrixes for floating point types at this point 
#}
{% if scalar_type_is_float %}
	type Mat3x3 = mat3x3<{{ scalar_type }}>;
	type Mat4x4 = mat4x4<{{ scalar_type }}>;
	type Mat4x3 = mat4x3<{{ scalar_type }}>;

	struct ArrayMatrix {
		data: array<Mat4x4>
	};

	struct ArrayMatrix3 {
		data: array<Mat3x3>
	};
{% endif %}

