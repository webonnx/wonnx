{%- include "structs.wgsl" -%}

@group(0) @binding(0)
var<storage, read> input_0: Array;

@group(0) @binding(1)
var<storage, read_write> output_0: Array;

@compute @workgroup_size({{ workgroup_size_x }})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let c = global_id.x;
    //let chunk_start = {{ i_chunks[0][1] }}u * c;
    let start = (c / {{ i_shape[0][1] }}u) * {{ i_shape[0][1] }}u;
    let end = start + {{ i_shape[0][1] - 1 }}u;

    var square_sum: Scalar = Scalar();
    for (var i = max(start, c - {{left_size}}u); i <= min(end, c + {{right_size}}u); i++) {
        let I = input_0.data[i];
        square_sum += I * I;
    }

    output_0.data[c] = input_0.data[ c ] / pow({{ scalar_type }}({{ bias }}) + ({{ scalar_type }}({{ alpha }}) / {{ scalar_type }}({{ size }})) * square_sum, {{ scalar_type }}({{ beta }}));
}
