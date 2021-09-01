[[block]]
struct PrimeIndices {
    data: [[stride(4)]] array<f32>;
}; // this is used as both input and output for convenience

[[group(0), binding(0)]]
var<storage, read_write> v_indices: PrimeIndices;

[[stage(compute), workgroup_size(1)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
    v_indices.data[global_id.x] = v_indices.data[global_id.x];
}