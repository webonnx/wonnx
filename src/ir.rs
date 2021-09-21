pub fn format_node(node: &crate::onnx::NodeProto) -> String {
    match node.get_op_type() {
            "_Convert_Vec_to_Block_Matrix" => {
                    format!(
                        r#"
                        
                    var i: u32 = global_id.x * {len}u + global_id.y;

		            for(var index_mat: u32 = 0u; index_mat < 4u; index_mat = index_mat + 1u) {{
		        	    {output}.data[i][index_mat] = {input}.data[global_id.x * {len}u * 4u + global_id.y + index_mat];
		            }}
                        "#,
                        input = node.get_input()[0],
                        output = node.get_output()[0],
                        len = todo!()
                    )
            }
            "abs" => {
                crate::op::as_vec(
                    format!(
                        "{output}.data[global_id.x][index_mat] = {op_type}({input}.data[global_id.x][index_mat]);",
                        input = node.get_input()[0],
                        output = node.get_output()[0],
                        op_type = node.get_op_type().to_lowercase()
                    )
                    .as_str(),
                )
            }
            "add" => {
                crate::op::as_vec(
                    format!(
                        "{output}.data[global_id.x][index_mat] = {input_0}.data[global_id.x][index_mat] + {input_1}.data[global_id.x][index_mat];",
                        input_0 = node.get_input()[0],
                        input_1 = node.get_input()[1],
                        output = node.get_output()[0],
                    ).as_str())
            }
            "matmul" => {
                format!(
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
                        input_0 = node.get_input()[0],
                        input_1 = node.get_input()[1],
                        output = node.get_output()[0],
                    len = node
                        .get_attribute()
                        .iter()
                        .filter(|x| x.get_name() == "len")
                        .next()
                        .expect("length attribute not found for matrix multiplication")
                        .get_i()
                    )
            },
            "relu" => {
                crate::op::as_vec(
                    format!(
                        "{output}.data[global_id.x][index_mat] = max({input}.data[global_id.x][index_mat], vec4<f32>(0.0, 0.0, 0.0, 0.0));",
                        input = node.get_input()[0],
                        output = node.get_output()[0],
                    )
                    .as_str(),
                )
            },
            "sum" => {
                unimplemented!()
            },
            "transpose" => {
                    format!(
                        r#"
                        
                        {output}.data[global_id.x] = max({input}.data[global_id.x], vec4<f32>(0.0, 0.0, 0.0, 0.0));"#,
                        input = node.get_input()[0],
                        output = node.get_output()[0],
                    )
            },
            _ => unimplemented!(),
        }
}

pub fn format_tensor(binding_group: u32, tensor: &crate::onnx::ValueInfoProto) -> String {
    format!(
        r#"
[[group(0), binding({i})]]
var<storage, read_write> {tensor}: ArrayMatrix;

"#,
        i = binding_group,
        tensor = tensor.get_name()
    )
}
