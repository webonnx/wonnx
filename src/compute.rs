use log::{debug, info};
use serde_derive::Serialize;
use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt;
use tera::{Context, Tera};

use std::time::Instant;

#[derive(Debug, Serialize)]
pub enum InnerType {
    Array,
    ArrayVector,
}

impl fmt::Display for InnerType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Serialize)]
struct Bindings {
    counter: u32,
    tensor: String,
    inner_type: String,
}

pub fn wrapper(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    node: &crate::onnx::NodeProto,
    inner_infos: &HashMap<String, crate::InnerInfo>,
    tera: &Tera,
) -> Result<(), wgpu::Error> {
    let mut binding_counter: u32 = 0;
    // Generating the shader

    let mut time = std::time::Duration::new(0, 0);
    let mut context = Context::new();
    let inputs = node.get_input();
    let outputs = node.get_output();

    let time_before_render = Instant::now();
    let inputs = if ["Reshape", "Clip", "Squeeze"].contains(&node.get_op_type()) {
        inputs.get(0..1).unwrap()
    } else {
        inputs
    };
    info!(
        "Computing node: {}, with op_type: {}",
        node.get_name(),
        node.get_op_type()
    );
    // Generating the shader
    let mut entries = vec![];
    let mut bindings = vec![];

    for tensor in inputs.iter() {
        let inner_type = &inner_infos
            .get(tensor)
            .unwrap_or_else(|| panic!("Invalid input tensor: {}", tensor))
            .inner_type;
        entries.push(wgpu::BindGroupEntry {
            binding: binding_counter,
            resource: inner_infos
                .get(tensor.as_str())
                .unwrap()
                .buffer
                .as_entire_binding(),
        });
        bindings.push(Bindings {
            counter: binding_counter,
            tensor: tensor.to_string(),
            inner_type: inner_type.to_string(),
        });
        binding_counter += 1;
    }

    for tensor in outputs.iter() {
        entries.push(wgpu::BindGroupEntry {
            binding: binding_counter,
            resource: inner_infos
                .get(tensor.as_str())
                .unwrap()
                .buffer
                .as_entire_binding(),
        });
        bindings.push(Bindings {
            counter: binding_counter,
            tensor: tensor.to_string(),
            inner_type: "ArrayVector".to_string(),
        });
        binding_counter += 1;
        info!(
            "output {} has size: {:?} at counter {}",
            tensor,
            inner_infos.get(tensor.as_str()).unwrap().dims,
            binding_counter
        );
    }
    context.insert("bindings", &bindings);

    // TODO: Add attribute value binding
    let (shader_template, x, y, z) = crate::compiler::format_node(node, inner_infos, &mut context);

    let shader = tera
        .render(&shader_template, &context)
        .expect("failed to render shader");

    debug!("shader: {}", shader);
    // debug!("x: {}", x);
    // TODO: Make defining threads more clean.

    time = Instant::now() - time_before_render + time;
    println!("time: {:#?}", time);
    // Generating the compute pipeline and binding group.

    // Instantiates the pipeline.
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(&shader)),
        }),
        entry_point: "main",
    });

    // Instantiates the bind group, once again specifying the binding of buffers.

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &compute_pipeline.get_bind_group_layout(0u32),
        entries: &entries,
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        // cpass.insert_debug_marker("compute");
        cpass.dispatch(x, y, z); // Number of cells to run, the (x,y,z) size of item being processed
    }
    queue.submit(Some(encoder.finish()));
    Ok(())
}
