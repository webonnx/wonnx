pub static INIT: &str = r#"[[block]]
struct Array {
    data: [[stride(4)]] array<f32>;
}; 

[[block]]
struct BigArray {
    data: [[stride(16)]] array<vec4<f32>>;
}; 


"#;
