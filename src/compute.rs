use log::{debug, info};
use serde_derive::Serialize;
use std::borrow::Cow;
use std::collections::HashMap;
use tera::{Context, Tera};

use std::time::Instant;

#[derive(Serialize)]
struct Bindings {
    counter: u32,
    tensor: String,
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

    for tensor in inputs {
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
        });
        binding_counter += 1;
    }

    for tensor in outputs {
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
    info!("time to compute node: {:#?}", time);
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
        layout: &compute_pipeline.get_bind_group_layout(0),
        entries: &entries,
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch(x, y, z); // Number of cells to run, the (x,y,z) size of item being processed
    }
    queue.submit(Some(encoder.finish()));
    Ok(())
}

pub fn compute(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    graph: &crate::onnx::GraphProto,
    inner_infos: &HashMap<String, crate::InnerInfo>,
    tera: &Tera,
) -> Result<(), wgpu::Error> {
    let mut iter = graph.get_node().iter();
    let mut previous_node = iter.next().unwrap();
    for node in iter {
        let previous_node_op_type = previous_node.get_op_type();
        let node_op_type = node.get_op_type();

        if previous_node_op_type == "Conv" && node_op_type == "Relu" {
            let mut tmp_node = crate::onnx::NodeProto::new();
            tmp_node.set_op_type("ConvRelu".to_string());
            tmp_node.set_name("ConvRelu".to_string());
            tmp_node.set_input(protobuf::RepeatedField::from(
                previous_node.get_input().to_vec(),
            ));
            tmp_node.set_attribute(protobuf::RepeatedField::from(previous_node.get_attribute()));
            tmp_node.set_output(protobuf::RepeatedField::from(node.get_output().to_vec()));

            crate::compute::wrapper(device, queue, &tmp_node, inner_infos, tera).unwrap();
        } else if previous_node_op_type == "Conv" && node_op_type != "Relu" {
            crate::compute::wrapper(device, queue, previous_node, inner_infos, tera).unwrap();
        } else if previous_node_op_type == "Relu" {
            //        } else if ["Dropout"].contains(&node_op_type) {
            //            let mut tmp_node = crate::onnx::NodeProto::new();
            //            tmp_node.set_op_type(previous_node_op_type.to_string());
            //            tmp_node.set_name("Some node".to_string());
            //            tmp_node.set_input(protobuf::RepeatedField::from(
            //                previous_node.get_input().to_vec(),
            //            ));
            //            tmp_node.set_attribute(protobuf::RepeatedField::from(previous_node.get_attribute()));
            //            tmp_node.set_output(protobuf::RepeatedField::from(node.get_output().to_vec()));
            //
            //            compute::wrapper(device, queue, &tmp_node, inner_infos, tera).unwrap();
        } else {
            crate::compute::wrapper(device, queue, previous_node, inner_infos, tera).unwrap();
        }

        previous_node = node;
    }
    crate::compute::wrapper(device, queue, previous_node, inner_infos, tera).unwrap();

    Ok(())
}
