// Operations usually work with a single scalar data type (typically f32). This data type is set by the compiler as the 
// 'scalar_type' variable. Here we define several other useful data types that shader code can use to make it more portable.
type Scalar = {{ scalar_type }};
type Vec3 = vec3<{{ scalar_type }}>;
type Vec4 = vec4<{{ scalar_type }}>;
type Mat3x3 = mat3x3<{{ scalar_type }}>;
type Mat4x4 = mat4x4<{{ scalar_type }}>;
type Mat4x3 = mat4x3<{{ scalar_type }}>;

struct Array {
	data: [[stride({{ scalar_stride }})]] array<Scalar>;
}; 

struct ArrayVector {
	data: [[stride({{ vec4_stride }})]] array<Vec4>;
}; 

struct ArrayMatrix {
	data: [[stride({{ mat4x4_stride }})]] array<Mat4x4>;
}; 

struct ArrayMatrix3 {
	data: [[stride({{ mat3x3_stride }})]] array<Mat3x3>;
};
