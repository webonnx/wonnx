{% include "structs.wgsl" %}

[[group(0), binding({{ bindings[0].counter }})]]
var<storage, read> var_{{ bindings[0].tensor }}: ArrayVector;

[[group(0), binding({{ bindings[1].counter }})]]
var<storage, write> var_{{ bindings[1].tensor }}: ArrayVector;


[[stage(compute), workgroup_size(1)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {
        let y = global_id.x % {{ len_1 * 4 }}u;
        let x = global_id.x / {{ len_1 * 4 }}u;
        let index = x * {{ len_1 * 4 }}u + y; 
        
        let tmp_mat = transpose(mat4x4<f32>(var_{{ input[0] }}.data[index], 
                            var_{{ input[0] }}.data[index + {{ len_1 }}u],
                            var_{{ input[0] }}.data[index + {{ 2 * len_1 }}u],
                            var_{{ input[0] }}.data[index + {{ 3 * len_1 }}u],
                        ));

        let index = y * {{ len_0 }}u + x;

        var_{{ output[0] }}.data[index] = tmp_mat[0u];
        var_{{ output[0] }}.data[index + {{ len_0 / 4 | int }}u] = tmp_mat[1u];
        var_{{ output[0] }}.data[index + {{ 2 * len_0 / 4 | int }}u] = tmp_mat[2u];
        var_{{ output[0] }}.data[index + {{ 3 * len_0 / 4 | int }}u] = tmp_mat[3u];
}