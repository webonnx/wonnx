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

        let input_dims = inner_infos.get(&input[0]).unwrap().dims.clone();
        debug!(
            "resource::size(input_dims): {:#?}",
            resource::size(&input_dims)
        );
        match node.get_op_type() {
            "Abs" | "Add" | "Relu" => {
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
            _ => unimplemented!(),
        }
    }

    inner_infos
}

pub fn get_value_info() {}

pub fn format_node(
    node: &crate::onnx::NodeProto,
    inner_infos: &HashMap<String, crate::InnerInfo>,
) -> (String, u32, u32, u32) {
    let inputs = node.get_input();
    let outputs = node.get_output();

    let dims = &inner_infos.get(&inputs[0]).unwrap().dims;

    let length = crate::utils::len(dims);

    match node.get_op_type() {
        "Abs" => (
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
        "Add" => (
            "let gidx = global_id.x;".to_string()
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
        "Matmul" => {
            let len_left = &inner_infos.get(&inputs[0]).unwrap().dims[1];
            let len_right = &inner_infos.get(&inputs[1]).unwrap().dims[1];

            (
                format!(
                    r#"
            let y = global_id.x % {len_right}u;
            let x = global_id.x / {len_right}u;

		    var tmpsum = {output}.data[i];
		    var product = {output}.data[i];
		    for(var k: u32 = 0u; k < {len_left}u; k = k + 1u) {{
                
                let mat_left = mat4x4<f32>({input_left}.data[x * {len_left_x_4}u + k], 
                                    {input_left}.data[x * {len_left_x_4}u + {len_left}u + k],
                                    {input_left}.data[x * {len_left_x_4}u + 2u * {len_left}u + k],
                                    {input_left}.data[x * {len_left_x_4}u + 3u * {len_left}u + k],
                                );
                
                let mat_right = mat4x4<f32>({input_right}.data[k * {len_right_x_4}u + y], 
                                    {input_right}.data[k * {len_right_x_4}u + {len_right}u + y],
                                    {input_right}.data[k * {len_right_x_4}u + 2u * {len_right}u + y],
                                    {input_right}.data[k * {len_right_x_4}u + 3u * {len_right}u + y],
                                );

			product = mat_left * mat_right;
			for(var index_mat: u32 = 0u; index_mat < 4u; index_mat = index_mat + 1u) {{
			    tmpsum[index_mat] = tmpsum[index_mat] + product[index_mat];
			}}
		    }}
		    {output}.data[i] = tmpsum;"#,
                    input_left = inputs[0],
                    input_right = inputs[1],
                    output = outputs[0],
                    len_left = len_left,
                    len_left_x_4 = len_left * 4,
                    len_right = len_right,
                    len_right_x_4 = len_right * 4,
                ),
                1,
                1,
                1,
            )
        }
        "Relu" => (
            "let gidx = global_id.x;".to_string()
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

            let perm = node
                .get_attribute()
                .iter()
                .find(|attr| attr.get_name() == "perm")
                .expect(format!("Required attribute '{}' not found", "perm").as_str())
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

            //    let y = global_id.x % {len_0_div_4}u;
            //    let x = global_id.x / {len_0_div_4}u;
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
