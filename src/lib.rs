pub mod boilerplate;
pub mod compute;
use std::error;
pub mod onnx;
pub mod resource;
pub mod utils;
use log::debug;
use protobuf::{self, Message};
use std::collections::HashMap;
// Change the alias to `Box<error::Error>`.
type Result<T> = std::result::Result<T, Box<dyn error::Error>>;

/// Creates a new session connected to the GPU.
///
/// Generate a session that will translate the onnx format into WGSL instructions.
///
/// # Examples
///
/// Basic usage:
///
/// ```ignore
/// let session = Session::from_path("path/to/model.onnx").await.unwrap();
/// ```
pub struct Session {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub model: onnx::ModelProto,
}

impl Session {
    pub async fn from_path(path: &str) -> Result<Session> {
        let (device, queue) = resource::request_device_queue().await;

        let model = onnx::ModelProto::parse_from_bytes(
            &std::fs::read(path).expect("ONNX Model path not found."),
        )
        .expect("Could not deserialize the Model");

        debug!("model: {:#?}", model);

        Ok(Session {
            device,
            queue,
            model,
        })
    }

    pub async fn from_model(model: onnx::ModelProto) -> Result<Session> {
        let (device, queue) = resource::request_device_queue().await;

        debug!("model: {:#?}", model);

        Ok(Session {
            device,
            queue,
            model,
        })
    }

    pub async fn run(&self, input_data: HashMap<String, (&[f32], &[i64])>) -> Option<Vec<f32>> {
        let graph = self.model.get_graph();
        let device = &self.device;
        let queue = &self.queue;

        let inner_infos = generate_buffer(input_data, graph, device);

        compute::wrapper(device, queue, graph, &inner_infos).unwrap();

        let outputs = graph.get_output();
        // TODO: Define behavior for multi output.
        let buffer_slice = inner_infos
            .get(outputs[0].get_name())
            .unwrap()
            .buffer
            .slice(..);
        let buffer_future = buffer_slice.map_async(wgpu::MapMode::Read);
        device.poll(wgpu::Maintain::Wait);

        // OUTPUT

        if let Ok(()) = buffer_future.await {
            // Gets contents of buffer
            let data = buffer_slice.get_mapped_range();
            // Since contents are got in bytes, this converts these bytes back to f32
            let result = bytemuck::cast_slice(&data).to_vec();

            //            drop(data);

            Some(result)
        } else {
            panic!("failed to run compute on gpu!")
        }
    }
}

pub struct InnerInfo {
    buffer: wgpu::Buffer,
    dims: Vec<i64>,
}

pub fn generate_buffer(
    input_data: HashMap<String, (&[f32], &[i64])>,
    graph: &onnx::GraphProto,
    device: &wgpu::Device,
) -> HashMap<String, InnerInfo> {
    let mut inner_infos = HashMap::new();

    for input in graph.get_input().iter() {
        let name = input.get_name();
        let (data, dim) = input_data
            .get(name)
            .unwrap_or_else(|| panic!("Input: {name} was not found in user HashMap.", name = name));
        inner_infos.insert(
            name.to_string(),
            InnerInfo {
                buffer: resource::create_buffer_init(device, data, name),
                dims: dim.to_vec(),
            },
        );
    }

    for node in graph.get_node().iter() {
        let input = node.get_input();
        let output = node.get_output();
        let attributes = node.get_attribute();

        let input_dims = inner_infos
            .get(&input[0])
            .expect(format!("Input: {} has not been provided", input[0]).as_str())
            .dims
            .clone();
        debug!(
            "resource::size(input_dims): {:#?}",
            resource::size(&input_dims)
        );
        match node.get_op_type() {
            "Abs" | "Acos" | "Asin" | "Atan" | "Ceil" | "Cos" | "Cosh" | "Exp" | "Floor"
            | "Log" | "Round" | "Sign" | "Sin" | "Sinh" | "Sqrt" | "Tan" | "Tanh" | "Add"
            | "And" | "Div" | "Equal" | "Greater" | "GreaterOrEqual" | "Less" | "LessOrEqual"
            | "Mod" | "Mul" | "Or" | "Sub" | "Celu" | "Elu" | "Relu" | "Sigmoid" | "Softsign" | "Softplus" => {
                inner_infos.insert(
                    output[0].clone(),
                    InnerInfo {
                        buffer: resource::create_buffer(
                            device,
                            resource::size(&input_dims) as _,
                            output[0].as_str(),
                        ),
                        dims: input_dims,
                    },
                );
            }
            "Transpose" => {
                let perm = attributes
                    .iter()
                    .find(|attr| attr.get_name() == "perm")
                    .expect(format!("Required attribute '{}' not found", "perm").as_str())
                    .get_ints();

                let mut output_dims = input_dims.clone();
                for (i, j) in input_dims.iter().zip(perm) {
                    output_dims[*j as usize] = *i;
                }

                inner_infos.insert(
                    output[0].clone(),
                    InnerInfo {
                        buffer: resource::create_buffer(
                            device,
                            resource::size(&output_dims) as _,
                            output[0].as_str(),
                        ),
                        dims: output_dims,
                    },
                );
            }
            "MatMul" => {
                let mut output_dims = input_dims.clone();
                let input_right_dims = inner_infos
                    .get(&input[1])
                    .expect(format!("Input: {} has not been provided", input[0]).as_str())
                    .dims
                    .clone();
                output_dims[1] = input_right_dims[1];
                inner_infos.insert(
                    output[0].clone(),
                    InnerInfo {
                        buffer: resource::create_buffer(
                            device,
                            resource::size(&output_dims) as _,
                            output[0].as_str(),
                        ),
                        dims: output_dims,
                    },
                );
            }
            _ => unimplemented!(),
        }
    }

    inner_infos
}

pub fn get_attribute<'a>(
    attribute: &'a str,
    defaults: Option<&'a onnx::AttributeProto>,
    node: &'a onnx::NodeProto,
) -> &'a onnx::AttributeProto {
    match defaults {
        Some(default) => node
            .get_attribute()
            .iter()
            .find(|attr| attr.get_name() == attribute)
            .unwrap_or(&default),
        None => node
            .get_attribute()
            .iter()
            .find(|attr| attr.get_name() == attribute)
            .expect("Did not find required attribute"),
    }
}

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
        "Celu" => {
            
            let mut alpha_default = onnx::AttributeProto::new();
            alpha_default.set_f(1.0);

            let alpha = get_attribute("alpha", Some(&alpha_default), node).get_f();
            (
            "let gidx = global_id.x;".to_string()
                + format!(
                    "{output}.data[gidx] = max(vec4<f32>(0.0, 0.0, 0.0, 0.0), {input_0}.data[gidx]) + min(
                        vec4<f32>(0.0, 0.0, 0.0, 0.0), 
                        {alpha} * (exp({input_0}.data[gidx] / {alpha} ) - vec4<f32>(1.0, 1.0, 1.0, 1.0) ));",
                    input_0 = inputs[0],
                    alpha = alpha,
                    output = outputs[0],
                )
                .as_str(),
            length as _,
            1,
            1,
        )},
        "Elu" => {
            
            let mut alpha_default = onnx::AttributeProto::new();
            alpha_default.set_f(1.0);

            let alpha = get_attribute("alpha", Some(&alpha_default), node).get_f();
            (
            "let gidx = global_id.x;".to_string()
                + format!(
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
                )
                .as_str(),
            length as _,
            1,
            1,
        )},
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
