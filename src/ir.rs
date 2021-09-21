pub fn get_value_info() {}
use std::collections::HashMap;

pub fn format_node(
    node: &crate::onnx::NodeProto,
    graph: &crate::onnx::GraphProto,
) -> (String, u32, u32, u32) {
    let inputs = node.get_input();
    let outputs = node.get_output();

    let inputs_value_info = graph.get_input();
    let outputs_value_info = graph.get_output();
    let attributes_value_info = graph.get_value_info();

    let mut value_infos = HashMap::new();

    for input in inputs {
        if let Some(value_info) = inputs_value_info
            .iter()
            .filter(|v| v.get_name() == input)
            .next()
        {
            value_infos.insert(input, value_info);
        } else if let Some(value_info) = attributes_value_info
            .iter()
            .filter(|v| v.get_name() == input)
            .next()
        {
            value_infos.insert(input, value_info);
        } else {
            panic!("Invalid input!")
        }
    }

    for output in outputs {
        if let Some(value_info) = outputs_value_info
            .iter()
            .filter(|v| v.get_name() == output)
            .next()
        {
            value_infos.insert(output, value_info);
        } else if let Some(value_info) = attributes_value_info
            .iter()
            .filter(|v| v.get_name() == output)
            .next()
        {
            value_infos.insert(output, value_info);
        } else if let Some(value_info) = inputs_value_info
            .iter()
            .filter(|v| v.get_name() == output)
            .next()
        {
            value_infos.insert(output, value_info);
        } else {
            panic!("Invalid output!")
        }
    }

    match node.get_op_type() {
        "Abs" => (
            (format!(
                "{output}.data[global_id.x] = {op_type}({input}.data[global_id.x]);",
                input = inputs[0],
                output = outputs[0],
                op_type = node.get_op_type().to_lowercase()
            ),
            (crate::resource::len(value_infos.get(&inputs[0]).unwrap())/4) as _,
            1,
            1)
        ),
        "Add" => {
            (format!(
                        "{output}.data[global_id.x] = {input_0}.data[global_id.x] + {input_1}.data[global_id.x];",
                        input_0 = inputs[0],
                        input_1 = inputs[1],
                        output = outputs[0],
                    ),
            (crate::resource::len(value_infos.get(&inputs[0]).unwrap())/4) as _,
            1,
            1)
        }
        "Matmul" => {
            (format!(
                r#"
            var i: u32 = global_id.x * {len}u + global_id.y;
		    var tmpsum = {output}.data[i];
		    var product = {output}.data[i];
		    for(var k: u32 = 0u; k < {len}u; k = k + 1u) {{
			product = {input_0}.data[global_id.x * {len}u + k] * {input_1}.data[global_id.y * {len}u + k];
			for(var index_mat: u32 = 0u; index_mat < 4u; index_mat = index_mat + 1u) {{
			    tmpsum[index_mat] = tmpsum[index_mat] + product[index_mat];
			}}
		    }}
		    {output}.data[i] = tmpsum;"#,
                input_0 = inputs[0],
                input_1 = inputs[1],
                output = outputs[0],
                len = node
                    .get_attribute()
                    .iter()
                    .filter(|x| x.get_name() == "len")
                    .next()
                    .expect("length attribute not found for matrix multiplication")
                    .get_i()
            ),
            1,
            1,
            1)
        }
        "Relu" => {
            (format!(
                        "{output}.data[global_id.x] = max({input}.data[global_id.x], vec4<f32>(0.0, 0.0, 0.0, 0.0));",
                        input = inputs[0],
                        output = outputs[0],
                    ),
            (crate::resource::len(value_infos.get(&inputs[0]).unwrap())/4) as _,
            1,
            1)
        }
        "Sum" => {
            unimplemented!()
        }
        "Transpose" => {
            let len = crate::resource::len_index(value_infos.get(&inputs[0]).unwrap(), 1)/4; 
            (format!(
                r#"
                let tmpMat = transpose(mat4x4<f32>({input}.data[global_id.x * {len}u + global_id.y], 
                                    {input}.data[(global_id.x + 1u) * {len}u + global_id.y],
                                    {input}.data[(global_id.x + 2u) * {len}u + global_id.y],
                                    {input}.data[(global_id.x + 3u) * {len}u + global_id.y],
                                ));
                {output}.data[global_id.y * {len4}u + global_id.x] = tmpMat[0u];
                {output}.data[global_id.y * {len4}u + 1u * {len}u + global_id.x] = tmpMat[1u];
                {output}.data[global_id.y * {len4}u + 2u * {len}u + global_id.x] = tmpMat[2u];
                {output}.data[global_id.y * {len4}u + 3u * {len}u + global_id.x] = tmpMat[3u];
                "#,
                input = inputs[0],
                output = outputs[0],
                len = len,
                len4 = len * 4
            ),
            (crate::resource::len_index(value_infos.get(&inputs[0]).unwrap(), 0)/4) as _,
            len as _,
            1)
        }
        _ => unimplemented!(),
    }
}

pub fn format_tensor(
    binding_group: u32,
    tensor: &crate::onnx::ValueInfoProto,
    inner_type: &crate::compute::InnerType,
) -> String {
    format!(
        r#"
[[group(0), binding({i})]]
var<storage, read_write> {tensor}: {inner_type};

"#,
        i = binding_group,
        tensor = tensor.get_name(),
        inner_type = inner_type,
    )
}
