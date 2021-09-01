use std::borrow::Cow;

fn define_storage(&n: &usize) -> String {
    let mut storage = "".to_string();

    for i in 0..n {
        storage.push_str(
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

    return storage;
}

pub fn unit_compute(
    device: &wgpu::Device,
    buffers: &[wgpu::BindGroupEntry],
    function: &str,
) -> (wgpu::ComputePipeline, wgpu::BindGroup) {
    let storage = define_storage(&buffers.len());

    let shader = format!(
        r#"
[[block]]
struct Array {{
    data: [[stride(4)]] array<f32>;
}}; 

{storage}

[[stage(compute), workgroup_size(1)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {{
    b_0.data[global_id.x] = {function}(b_0.data[global_id.x])
    +0.1;
    
}}
"#,
        function = function,
        storage = storage
    );

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
        entries: buffers,
    });

    (compute_pipeline, bind_group)
}

pub fn conv_compute(
    device: &wgpu::Device,
    buffers: &[wgpu::BindGroupEntry],
    convolution: &[f32],
    stride: &[i32],
) -> (wgpu::ComputePipeline, wgpu::BindGroup) {
    let storage = define_storage(&buffers.len());

    let mut equations: String = "".into();
    let mut i = 0.;
    for conv in convolution {
        equations.push_str(
            format!(
                " {conv}.0 * b_0.data[{stride}u * global_id.x + {idx}u] +",
                conv = conv,
                stride = stride[0],
                idx = i
            )
            .as_str(),
        );

        i += 1.;
    }
    equations.pop();
    let shader = format!(
        r#"
[[block]]
struct Array {{
    data: [[stride(4)]] array<f32>;
}}; 

{storage}

[[stage(compute), workgroup_size(1)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {{
        b_1.data[global_id.x] ={equations};
 }}
"#,
        equations = equations,
        storage = storage
    );

    println!("shaders: {}", shader);

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
        entries: buffers,
    });

    (compute_pipeline, bind_group)
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_compute() {
        let (device, _) = pollster::block_on(crate::ressource::request_device_queue());
        let data = [1.0, 2.0, 3.0, 4.0];
        let buffer = crate::ressource::create_buffer_init(&device, &data);

        let binding_group_entry = wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        };
        crate::compute::unit_compute(&device, &[binding_group_entry], "cos");
    }
}
