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
    let inputs = graph.get_input();
    let value_infos = graph.get_value_info();
    let outputs = graph.get_output();
    let nodes = graph.get_node();

    // Generating the shader
    let mut shader = crate::boilerplate::INIT.to_string();

    let mut binding_counter: u32 = 0;
    let mut binding_key: HashMap<&str, u32> = HashMap::new();
    let mut entries = vec![];
    for tensor in inputs.iter() {
        shader.push_str(
            crate::ir::format_tensor(
                binding_counter,
                tensor,
                &crate::compute::InnerType::ArrayVector,
            )
            .as_str(),
        );
        binding_key.insert(tensor.get_name(), binding_counter);
        entries.push(wgpu::BindGroupEntry {
            binding: binding_counter,
            resource: buffers.get(tensor.get_name()).unwrap().as_entire_binding(),
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
        binding_key.insert(tensor.get_name(), binding_counter);
        entries.push(wgpu::BindGroupEntry {
            binding: binding_counter,
            resource: buffers.get(tensor.get_name()).unwrap().as_entire_binding(),
        });
        binding_counter += 1;
    }

    for tensor in value_infos.iter() {
        shader.push_str(
            crate::ir::format_tensor(
                binding_counter,
                tensor,
                &crate::compute::InnerType::ArrayVector,
            )
            .as_str(),
        );
        binding_key.insert(tensor.get_name(), binding_counter);
        entries.push(wgpu::BindGroupEntry {
            binding: binding_counter,
            resource: buffers.get(tensor.get_name()).unwrap().as_entire_binding(),
        });
        binding_counter += 1;
    }

    let shaders = nodes
        .iter()
        .map(|node| crate::ir::format_node(node, graph))
        .collect::<Vec<(String, u32, u32, u32)>>();

    shader.push_str(&format!(
        r#"
[[stage(compute), workgroup_size(1)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {{
    
    {main_body}
}}
"#,
        main_body = shaders
            .iter()
            .map(|(shader, _, _, _)| shader)
            .fold("".to_string(), |acc, node| acc + "\n" + &node),
    ));

    debug!("shader: {}", shader);

    // TODO: Make defining threads more clean.
    let (_, x, y, z) = shaders.iter().next().unwrap();

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
    let bind_group_layout = compute_pipeline.get_bind_group_layout(0 as _);

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &entries,
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        debug!("Ready for dispatch!");
        cpass.insert_debug_marker("compute");
        cpass.dispatch(*x, *y, *z); // Number of cells to run, the (x,y,z) size of item being processed
    }
    queue.submit(Some(encoder.finish()));
    Ok(())
}
