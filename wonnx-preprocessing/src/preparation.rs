use std::collections::HashMap;

use wonnx::onnx::{GraphProto, TypeProto_oneof_value, ValueInfoProto};

pub fn apply_dynamic_dimensions(graph: &mut GraphProto, dynamic_dims: &HashMap<String, i64>) {
    // Apply to values
    for value_info in graph.mut_value_info() {
        apply_dynamic_dimensions_value(value_info, dynamic_dims);
    }

    for value_info in graph.mut_input() {
        apply_dynamic_dimensions_value(value_info, dynamic_dims);
    }

    for value_info in graph.mut_output() {
        apply_dynamic_dimensions_value(value_info, dynamic_dims);
    }
}

/// Replaces dimension params with provided values
fn apply_dynamic_dimensions_value(
    value_info: &mut ValueInfoProto,
    dynamic_dims: &HashMap<String, i64>,
) {
    let name = value_info.get_name().to_string();
    let field_type = value_info.mut_field_type();

    if let Some(TypeProto_oneof_value::tensor_type(field_type_value)) = &mut field_type.value {
        let dims = field_type_value.mut_shape().mut_dim();

        for (idx, dim) in dims.iter_mut().enumerate() {
            if let Some(new_dim_value) = dynamic_dims.get(dim.get_dim_param()) {
                println!(
                    "Setting dimension param {idx} ({}) to value {new_dim_value} for {name}",
                    dim.get_dim_param()
                );
                dim.clear_dim_param();
                dim.set_dim_value(*new_dim_value);
            }
        }
    }
}
