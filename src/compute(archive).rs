use log::debug;
use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt;

#[derive(Debug)]
pub enum InnerType {
    Array,
    ArrayVector,
}

impl fmt::Display for InnerType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

pub fn wrapper(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    graph: &crate::onnx::GraphProto,
    buffers: &HashMap<&str, wgpu::Buffer>,
) -> Result<(), wgpu::Error> {
    let nodes = graph.get_node();

    let mut binding_counter: u32 = 0;
    // Generating the shader

    for node in nodes.iter() {
        let inputs = node.get_input();
        let outputs = node.get_output();

        let mut binding_key: HashMap<&str, u32> = HashMap::new();
        let mut entries = vec![];
        let mut shader = crate::boilerplate::INIT.to_string();
        for tensor in inputs.iter() {
            shader.push_str(
                crate::ir::format_tensor(
                    binding_counter,
                    tensor,
                    &crate::compute::InnerType::ArrayVector,
                )
                .as_str(),
            );
            binding_key.insert(tensor, binding_counter);
            entries.push(wgpu::BindGroupEntry {
                binding: binding_counter,
                resource: buffers.get(tensor.as_str()).unwrap().as_entire_binding(),
            });
            binding_counter += 1;
        }

        for tensor in outputs.iter() {
            shader.push_str(
                crate::ir::format_tensor(
                    binding_counter,
                    tensor,
                    &crate::compute::InnerType::ArrayVector,
                )
                .as_str(),
            );
            binding_key.insert(tensor, binding_counter);
            entries.push(wgpu::BindGroupEntry {
                binding: binding_counter,
                resource: buffers.get(tensor.as_str()).unwrap().as_entire_binding(),
            });
            binding_counter += 1;
        }

        let (shader_node, x, y, z) = crate::ir::format_node(node, graph);
        shader.push_str(&format!(
            r#"
[[stage(compute), workgroup_size(1)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {{
    
    {main_body}
}}
"#,
            main_body = shader_node
        ));

        debug!("shader: {}", shader);

        // Generating the compute pipeline and binding group.
        let cs_module = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(&shader)),
        });

        // Instantiates the pipeline.
        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &cs_module,
            entry_point: "main",
        });

        // Instantiates the bind group, once again specifying the binding of buffers.
        let bind_group_layout = compute_pipeline.get_bind_group_layout(0);

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &entries,
        });

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        {
            let mut cpass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
            cpass.set_pipeline(&compute_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.insert_debug_marker("compute");
            cpass.dispatch(x, y, z); // Number of cells to run, the (x,y,z) size of item being processed
        }
        queue.submit(Some(encoder.finish()));
    }
    Ok(())
}
