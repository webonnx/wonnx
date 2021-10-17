[[block]]
struct Array {
    data: [[stride(4)]] array<f32>;
}; 

[[block]]
struct ArrayVector {
    data: [[stride(16)]] array<vec4<f32>>;
}; 

[[block]]
struct ArrayMatrix {
    data: [[stride(64)]] array<mat4x4<f32>>;
}; 

[[block]]
struct ArrayMatrix3 {
    data: [[stride(48)]] array<mat3x3<f32>>;
}; 

[[block]]
struct ArrayVector3 {
    data: [[stride(16)]] array<vec3<f32>>;
}; 