use log::debug;
use std::borrow::Cow;

pub fn wrapper(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffers: &[wgpu::BindGroupEntry],
    bindings: &[i32],
    main: &str,
    x: u32,
    y: u32,
    z: u32,
) -> Result<(), wgpu::Error> {
    // Generating the shader
    let mut shader = crate::boilerplate::INIT.to_string();

    for i in bindings {
        shader.push_str(
            format!(
                r#"
[[group(0), binding({i})]]
var<storage, read_write> b_{i}: Array;

"#,
                i = i
            )
            .as_str(),
        )
    }

    shader.push_str(&format!(
        r#"
{main}    
"#,
        main = main
    ));

    println!("shader: {}", shader);

    // Generating the compute pipeline and binding group.
    let cs_module = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(&shader)),
    });
    debug!("Successfully generated cs module!");
    // Instantiates the pipeline.
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &cs_module,
        entry_point: "main",
    });

    // Instantiates the bind group, once again specifying the binding of buffers.
    let bind_group_layout = compute_pipeline.get_bind_group_layout(bindings[0] as _);
    debug!("Successfully created bind group layout!");
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: buffers,
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        debug!("Ready for dispatch!");
        cpass.insert_debug_marker("compute collatz iterations");
        cpass.dispatch(x, y, z); // Number of cells to run, the (x,y,z) size of item being processed
    }
    queue.submit(Some(encoder.finish()));
    Ok(())
}

pub fn matrix_wrapper(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffers: &[wgpu::BindGroupEntry],
    bindings: &[i32],
    main: &str,
    x: u32,
    y: u32,
    z: u32,
) -> Result<(), wgpu::Error> {
    // Generating the shader
    let mut shader = crate::boilerplate::INIT.to_string();

    for i in bindings {
        shader.push_str(
            format!(
                r#"
[[group(0), binding({i})]]
var<storage, read_write> b_{i}: BigArray;

"#,
                i = i
            )
            .as_str(),
        )
    }

    shader.push_str(&format!(
        r#"
{main}    
"#,
        main = main
    ));

    println!("shader: {}", shader);

    // Generating the compute pipeline and binding group.
    let cs_module = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(&shader)),
    });
    debug!("Successfully generated cs module!");
    // Instantiates the pipeline.
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &cs_module,
        entry_point: "main",
    });

    // Instantiates the bind group, once again specifying the binding of buffers.
    let bind_group_layout = compute_pipeline.get_bind_group_layout(bindings[0] as _);
    debug!("Successfully created bind group layout!");
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: buffers,
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        debug!("Ready for dispatch!");
        cpass.insert_debug_marker("compute collatz iterations");
        cpass.dispatch(x, y, z); // Number of cells to run, the (x,y,z) size of item being processed
    }
    queue.submit(Some(encoder.finish()));
    Ok(())
}
