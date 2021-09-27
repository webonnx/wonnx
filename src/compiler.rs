

use std::collections::HashMap;
use crate::onnx;
use crate::get_attribute;

pub fn format_node(
    node: &crate::onnx::NodeProto,
    inner_infos: &HashMap<String, crate::InnerInfo>,
) -> (String, u32, u32, u32) {
    let inputs = node.get_input();
    let outputs = node.get_output();

    let dims = &inner_infos.get(&inputs[0]).unwrap().dims;

    let length = crate::utils::len(dims);

    match node.get_op_type() {
        "Abs" | "Acos" | "Asin" | "Atan" | "Ceil" | "Cos" | "Cosh" | "Exp" | "Floor" | "Log"
        | "Round" | "Sign" | "Sin" | "Sinh" | "Sqrt" | "Tan" | "Tanh" => (
            "let gidx = global_id.x;".to_string()
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
        "Add" | "And" | "Div" | "Equal" | "Greater" | "GreaterOrEqual" | "Less" | "LessOrEqual"
        | "Mod" | "Mul" | "Or" | "Sub" => (
            "let gidx = global_id.x;".to_string()
                + format!(
                    "{output}.data[gidx] = {input_0}.data[gidx] {op_type} {input_1}.data[gidx];",
                    input_0 = inputs[0],
                    input_1 = inputs[1],
                    output = outputs[0],
                    op_type = match node.get_op_type() {
                        "Add" => "+",
                        "And" => "&",
                        "Div" => "/",
                        "Equal" => "==",
                        "Greater" => ">",
                        "GreaterOrEqual" => ">=",
                        "Less" => "<",
                        "LessOrEqual" => "<=",
                        "Mod" => "%",
                        "Mul" => "*",
                        "Or" => "|",
                        "Sub" => "-",
                        _ => unimplemented!(),
                    }
                )
                .as_str(),
            length as _,
            1,
            1,
        ),
        "Celu" | "Elu" => {
            
            let mut alpha_default = onnx::AttributeProto::new();
            alpha_default.set_f(1.0);

            let alpha = get_attribute("alpha", Some(&alpha_default), node).get_f();
            
            (
            "let gidx = global_id.x;".to_string()
                + match node.get_op_type() {
                    "Celu" => 
                 format!(
                    "{output}.data[gidx] = max(vec4<f32>(0.0, 0.0, 0.0, 0.0), {input_0}.data[gidx]) + min(
                        vec4<f32>(0.0, 0.0, 0.0, 0.0), 
                        {alpha} * (exp({input_0}.data[gidx] / {alpha} ) - vec4<f32>(1.0, 1.0, 1.0, 1.0) ));",
                    input_0 = inputs[0],
                    alpha = alpha,
                    output = outputs[0],
                ),
                    "Elu" =>  format!(
                    r#"
        let tmp_vec = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        let input_vec = {input_0}.data[gidx]; 
        for(var index_vec: u32 = 0u; index_vec < 4u; index_vec = index_vec + 1u) {{
            if (input_vec[index_vec] < 0.0) {{
	            tmp_vec[index_vec] = {alpha} * (exp(input_vec[index_vec]) - 1.0);
            }} else {{  
	            tmp_vec[index_vec] = input_vec[index_vec];
            }}
	    }}
        {output}.data[gidx] = tmp_vec;
                    "#,
                    input_0 = inputs[0],
                    alpha = alpha,
                    output = outputs[0],
                ),
                _ => unimplemented!()
            }.as_str(),
            length as _,
            1,
            1,
        )},
            "Conv" => {
                // TODO: Conv only support NxCxHxW for the moment.
                debug_assert!(input_dims.len() == 4usize);

                let mut auto_pad_default = onnx::AttributeProto::new();
                auto_pad_default.set_s("NOTSET".to_string().into_bytes());

                let auto_pad =
                    from_utf8(get_attribute("auto_pad", Some(&auto_pad_default), node).get_s())
                        .unwrap();

                let mut dilations_default = onnx::AttributeProto::new();
                dilations_default.set_ints(vec![1, 1]);

                let dilations =
                    get_attribute("dilations", Some(&dilations_default), node).get_ints();

                let kernel_shape = get_attribute("kernel_shape", None, node).get_ints();

                let mut strides_default = onnx::AttributeProto::new();
                strides_default.set_ints(vec![1, 1]);

                let strides = get_attribute("strides", Some(&strides_default), node).get_ints();

                let mut pads_default = onnx::AttributeProto::new();
                pads_default.set_ints(vec![0, 0, 0, 0]);

                let pads = get_attribute("pads", Some(&pads_default), node).get_ints();

                let mut output_dims = input_dims.clone();

                let w = initializers
                    .iter()
                    .find(|x| x.get_name() == input[1].as_str())
                    .expect(
                        format!("Did not find initializer for input Conv {}", input[1]).as_str(),
                    );
                let w_data = w.get_float_data();
                let w_dims = w.get_dims();

                inner_infos.insert(
                    input[1].clone(),
                    InnerInfo {
                        buffer: resource::create_buffer_init(device, w_data, input[1].as_str()),
                        dims: w_dims.to_vec(),
                    },
                );

                match auto_pad {
                    "NOTSET" => {
                        output_dims[0] = input_dims[0];
                        output_dims[1] = w_dims[0];
                        output_dims[2] = (input_dims[2]
                            - ((kernel_shape[0] - 1) * dilations[0] + 1)
                            + pads[0]
                            + pads[2])
                            / strides[0]
                            + 1;
                        output_dims[3] = (input_dims[3]
                            - ((kernel_shape[1] - 1) * dilations[1] + 1)
                            + pads[1]
                            + pads[3])
                            / strides[1]
                            + 1;
                    }
                    "SAME_UPPER" => {
                        output_dims[0] = input_dims[0];
                        output_dims[1] = w_dims[0];
                        output_dims[2] = input_dims[2] / strides[0];
                        output_dims[3] = input_dims[3] / strides[1];
                    }
                    "SAME_LOWER" => {
                        output_dims[0] = input_dims[0];
                        output_dims[1] = w_dims[0];
                        output_dims[2] = input_dims[2] / strides[0];
                        output_dims[3] = input_dims[3] / strides[1];
                    }
                    _ => unimplemented!(),
                }
                inner_infos.insert(
                    output[0].clone(),
                    InnerInfo {
                        buffer: resource::create_buffer(
                            device,
                            resource::size(&input_dims) as _,
                            output[0].as_str(),
                        ),
                        dims: output_dims,
                    },
                );
		unimplemented!();
		
            },

        "Gemm" => {
            let mut alpha_default = onnx::AttributeProto::new();
            alpha_default.set_f(1.0);

            let alpha = get_attribute("alpha", Some(&alpha_default), node).get_f();

            let mut beta_default = onnx::AttributeProto::new();
            beta_default.set_f(1.0);

            let beta = get_attribute("beta", Some(&beta_default), node).get_f();

            let left_columns = &inner_infos.get(&inputs[0]).unwrap().dims[1];
            let right_columns = &inner_infos.get(&inputs[1]).unwrap().dims[1];
            let threads = (&inner_infos.get(&inputs[0]).unwrap().dims[0] / 4) * right_columns / 4;

            (
                format!(
                    r#"
    let y = global_id.x % {right_columns_div_4}u;
    let x = global_id.x / {right_columns_div_4}u;
    let index = x * {right_columns}u + y;

    var tmpsum = mat4x4<f32>(vec4<f32>(0.0, 0.0, 0.0, 0.0), vec4<f32>(0.0, 0.0, 0.0, 0.0), vec4<f32>(0.0, 0.0, 0.0, 0.0), vec4<f32>(0.0, 0.0, 0.0, 0.0));
    var product = mat4x4<f32>(vec4<f32>(0.0, 0.0, 0.0, 0.0), vec4<f32>(0.0, 0.0, 0.0, 0.0), vec4<f32>(0.0, 0.0, 0.0, 0.0), vec4<f32>(0.0, 0.0, 0.0, 0.0));

    for(var k: u32 = 0u; k < {left_columns_div_4}u; k = k + 1u) {{
        let index_left = x * {left_columns}u + k; 
        let index_right = k * {left_columns}u + y; 

        let mat_left = mat4x4<f32>(
                              {input_left}.data[index_left], 
                              {input_left}.data[index_left + {left_columns_div_4}u],
                              {input_left}.data[index_left + 2u * {left_columns_div_4}u],
                              {input_left}.data[index_left + 3u * {left_columns_div_4}u],
                          );
          
        let mat_right = mat4x4<f32>(
                              {input_right}.data[index_right], 
                              {input_right}.data[index_right + {right_columns_div_4}u],
                              {input_right}.data[index_right + 2u * {right_columns_div_4}u],
                              {input_right}.data[index_right + 3u * {right_columns_div_4}u],
                          );
	
        product = mat_right * mat_left;
	
        for(var index_mat: u32 = 0u; index_mat < 4u; index_mat = index_mat + 1u) {{
	        tmpsum[index_mat] = tmpsum[index_mat] + product[index_mat];
	    }}
    }}

    {output}.data[index] = tmpsum[0];
    {output}.data[index + {right_columns_div_4}u] = tmpsum[1];
    {output}.data[index + 2u * {right_columns_div_4}u] = tmpsum[2];
    {output}.data[index + 3u * {right_columns_div_4}u] = tmpsum[3];
      
            "#,
                    input_left = inputs[0],
                    input_right = inputs[1],
                    output = outputs[0],
                    left_columns = left_columns,
                    left_columns_div_4 = left_columns / 4,
                    // The right columns is composed of 4 vector of size 4
                    right_columns = right_columns,
                    right_columns_div_4 = right_columns / 4,
                ),
                threads as _,
                1,
                1,
            )
        }
        "MatMul" => {
            let left_columns = &inner_infos.get(&inputs[0]).unwrap().dims[1];
            let right_columns = &inner_infos.get(&inputs[1]).unwrap().dims[1];
            let threads = (&inner_infos.get(&inputs[0]).unwrap().dims[0] / 4) * right_columns / 4;

            (
                format!(
                    r#"
    let y = global_id.x % {right_columns_div_4}u;
    let x = global_id.x / {right_columns_div_4}u;
    let index = x * {right_columns}u + y;

    var tmpsum = mat4x4<f32>(vec4<f32>(0.0, 0.0, 0.0, 0.0), vec4<f32>(0.0, 0.0, 0.0, 0.0), vec4<f32>(0.0, 0.0, 0.0, 0.0), vec4<f32>(0.0, 0.0, 0.0, 0.0));
    var product = mat4x4<f32>(vec4<f32>(0.0, 0.0, 0.0, 0.0), vec4<f32>(0.0, 0.0, 0.0, 0.0), vec4<f32>(0.0, 0.0, 0.0, 0.0), vec4<f32>(0.0, 0.0, 0.0, 0.0));

    for(var k: u32 = 0u; k < {left_columns_div_4}u; k = k + 1u) {{
        let index_left = x * {left_columns}u + k; 
        let index_right = k * {left_columns}u + y; 

        let mat_left = mat4x4<f32>(
                              {input_left}.data[index_left], 
                              {input_left}.data[index_left + {left_columns_div_4}u],
                              {input_left}.data[index_left + 2u * {left_columns_div_4}u],
                              {input_left}.data[index_left + 3u * {left_columns_div_4}u],
                          );
          
        let mat_right = mat4x4<f32>(
                              {input_right}.data[index_right], 
                              {input_right}.data[index_right + {right_columns_div_4}u],
                              {input_right}.data[index_right + 2u * {right_columns_div_4}u],
                              {input_right}.data[index_right + 3u * {right_columns_div_4}u],
                          );
	
        product = mat_right * mat_left;
	
        for(var index_mat: u32 = 0u; index_mat < 4u; index_mat = index_mat + 1u) {{
	        tmpsum[index_mat] = tmpsum[index_mat] + product[index_mat];
	    }}
    }}

    {output}.data[index] = tmpsum[0];
    {output}.data[index + {right_columns_div_4}u] = tmpsum[1];
    {output}.data[index + 2u * {right_columns_div_4}u] = tmpsum[2];
    {output}.data[index + 3u * {right_columns_div_4}u] = tmpsum[3];
      
            "#,
                    input_left = inputs[0],
                    input_right = inputs[1],
                    output = outputs[0],
                    left_columns = left_columns,
                    left_columns_div_4 = left_columns / 4,
                    // The right columns is composed of 4 vector of size 4
                    right_columns = right_columns,
                    right_columns_div_4 = right_columns / 4,
                ),
                threads as _,
                1,
                1,
            )
        }
        "Relu" | "Sigmoid" | "Softsign" | "Softplus" => (
            "let gidx = global_id.x;".to_string()
                + match node.get_op_type() {
                    "Relu" => 
                        format!(
                    "{output}.data[gidx] = max({input}.data[gidx], vec4<f32>(0.0, 0.0, 0.0, 0.0));",
                    input = inputs[0],
                    output = outputs[0],
                    ),
                    "Sigmoid" => 
                        format!(
                    "{output}.data[gidx] = vec4<f32>(1.0, 1.0, 1.0, 1.0) / (vec4<f32>(1.0, 1.0, 1.0, 1.0) + exp(-{input}.data[gidx]));",
                    input = inputs[0],
                    output = outputs[0],
                    ),
                    "Softsign" => 
                        format!(
                    "{output}.data[gidx] = {input}.data[gidx] / (vec4<f32>(1.0, 1.0, 1.0, 1.0) + abs({input}.data[gidx]));",
                    input = inputs[0],
                    output = outputs[0],
                    ),
                    "Softplus" => 
                        format!(
                    "{output}.data[gidx] = log(vec4<f32>(1.0, 1.0, 1.0, 1.0) + exp({input}.data[gidx]));",
                    input = inputs[0],
                    output = outputs[0],
                    ),
                    _ => unimplemented!("Unsupported activation"),
                }
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

            let perm = get_attribute("perm", None, &node)
                .get_ints();

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

                let index = y * {len_0}u + x;

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
                (length / 4) as _,
                1,
                1,
            )
        }
        _ => unimplemented!(),
    }
}
