pub fn get_value_info() {}
use std::collections::HashMap;

pub fn format_node(
    node: &crate::onnx::NodeProto,
    inner_infos: &HashMap<&str, crate::InnerInfo>,
) -> (String, u32, u32, u32) {
    let inputs = node.get_input();
    let outputs = node.get_output();

    let dims = inner_infos
        .get(&inputs[0].as_str())
        .unwrap()
        .dims
        .as_ref()
        .unwrap();

    let length = crate::resource::len(dims);

    match node.get_op_type() {
        "Abs" => (
            format!("let gidx = global_id.x * {}u + global_id.y;", length)
                + format!(
                    "{output}.data[gidx] = {op_type}({input}.data[gidx]);",
                    input = inputs[0],
                    output = outputs[0],
                    op_type = node.get_op_type().to_lowercase()
                )
                .as_str(),
            length as _,
            1,
            1,
        ),
        "Add" => (
            format!("let gidx = global_id.x * {}u + global_id.y;", length)
                + format!(
                    "{output}.data[gidx] = {input_0}.data[gidx] + {input_1}.data[gidx];",
                    input_0 = inputs[0],
                    input_1 = inputs[1],
                    output = outputs[0],
                )
                .as_str(),
            length as _,
            1,
            1,
        ),
        "Matmul" => (
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
            1,
        ),
        "Relu" => (
            format!("let gidx = global_id.x * {}u + global_id.y;", length)
                + format!(
                    "{output}.data[gidx] = max({input}.data[gidx], vec4<f32>(0.0, 0.0, 0.0, 0.0));",
                    input = inputs[0],
                    output = outputs[0],
                )
                .as_str(),
            length as _,
            1,
            1,
        ),
        "Sum" => {
            unimplemented!()
        }
        "Transpose" => {
            let len_0 = dims[0];
            let len_1 = dims[1] / 4;

            (
                format!(
                    r#"

                let y = global_id.x % {len_1}u;
                let x = global_id.x / {len_1}u;
                let index = x * {len_1_x_4}u + y; 
                
                let tmpMat_{input} = transpose(mat4x4<f32>({input}.data[index], 
                                    {input}.data[index + {len_1}u],
                                    {input}.data[index + 2u * {len_1}u],
                                    {input}.data[index + 3u * {len_1}u],
                                ));

                let y = global_id.x % {len_0_div_4}u;
                let x = global_id.x / {len_0_div_4}u;
                let index = x * {len_0}u + y;

                {output}.data[index] = tmpMat_{input}[0u];
                {output}.data[index + {len_0_div_4}u] = tmpMat_{input}[1u];
                {output}.data[index + 2u * {len_0_div_4}u] = tmpMat_{input}[2u];
                {output}.data[index + 3u * {len_0_div_4}u] = tmpMat_{input}[3u];
                "#,
                    input = inputs[0],
                    output = outputs[0],
                    len_1 = len_1,
                    len_1_x_4 = len_1 * 4,
                    len_0 = len_0,
                    len_0_div_4 = len_0 / 4
                ),
                length as _,
                1,
                1,
            )
        }
        _ => unimplemented!(),
    }
}

pub fn format_tensor(
    binding_group: u32,
    tensor: &str,
    inner_type: &crate::compute::InnerType,
) -> String {
    format!(
        r#"
[[group(0), binding({i})]]
var<storage, read_write> {tensor}: {inner_type};

"#,
        i = binding_group,
        tensor = tensor,
        inner_type = inner_type,
    )
}
