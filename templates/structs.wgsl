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
