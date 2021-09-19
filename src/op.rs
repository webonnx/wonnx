pub fn matrix_map(function: &str) -> String {
    format!(
        r#"
[[stage(compute), workgroup_size(1)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {{
    for(var index_mat: u32 = 0u; index_mat < 4u; index_mat = index_mat + 1u) {{
        b_0.data[global_id.x][index_mat] = {function};
    }}
}}
"#,
        function = function,
    )
}

pub fn main(nodes: &[&crate::ir::Node]) -> String {
    format!(
        r#"
[[stage(compute), workgroup_size(1)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {{
    {main_body}
}}
"#,
        main_body = nodes
            .iter()
            .map(|node| node.to_string())
            .fold("".to_string(), |acc, node| acc + "\n\n" + &node),
    )
}

pub fn as_vec(function: &str) -> String {
    format!(
        r#"
    for(var index_mat: u32 = 0u; index_mat < 4u; index_mat = index_mat + 1u) {{
        {function}
    }}
"#,
        function = function,
    )
}

pub fn map(function: &str) -> String {
    format!(
        r#"
[[stage(compute), workgroup_size(1)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {{
    b_0.data[global_id.x] = {function}(b_0.data[global_id.x]);
}}
"#,
        function = function,
    )
}

pub fn matmul(len: i32) -> String {
    format!(
        r#"
[[stage(compute), workgroup_size(1)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {{
        var i: u32 = global_id.x * {len}u + global_id.y;
        var tmpSum: f32 = 0.0;
        for(var k: u32 = 0u; k < {len}u; k = k + 1u) {{
            tmpSum = fma(b_0.data[global_id.x * {len}u + k], b_1.data[global_id.y + k * {len}u], tmpSum);
        }}
        b_2.data[i] = tmpSum;
}}
    "#,
        len = len
    )
}

pub fn vector_matmul(len: i32) -> String {
    format!(
        r#"
[[stage(compute), workgroup_size(1)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {{
        var i: u32 = global_id.x * {len}u + global_id.y;
        var tmpSum = b_3.data[i];
        var product = b_3.data[i];
        for(var k: u32 = 0u; k < {len}u; k = k + 1u) {{
            product = b_0.data[global_id.x * {len}u + k] * b_1.data[global_id.y * {len}u + k];
            for(var index_mat: u32 = 0u; index_mat < 4u; index_mat = index_mat + 1u) {{
                tmpSum[index_mat] = tmpSum[index_mat] + product[index_mat];
            }}
        }}
        b_3.data[i] = tmpSum;
}}
    "#,
        len = len
    )
}

pub fn scan(workgroup_size: i32, stride: i32) -> String {
    format!(
        r#"
[[stage(compute), workgroup_size({workgroup_size})]]
fn main(
    [[builtin(global_invocation_id)]] global_id: vec3<u32>,
) {{
        b_0.data[{stride_1}u * (global_id.x)] = b_0.data[{stride_1}u * (global_id.x)] 
        + b_0.data[{stride_1}u * (global_id.x) + {stride}u ];
}}
    "#,
        workgroup_size = workgroup_size,
        stride_1 = stride * 2,
        stride = stride,
    )
}

pub fn conv(convolution: &[f32], stride: &[i32], recursive: bool) -> String {
    let recursive = if recursive { 0 } else { 1 };
    let mut main: String = format!(
        "
[[stage(compute), workgroup_size(1)]]
fn main([[builtin(global_invocation_id)]] global_id: vec3<u32>) {{
            b_{recursive}.data[{stride}u * global_id.x] = ",
        stride = &stride[0],
        recursive = recursive
    );

    let mut i = 0;
    for conv in convolution {
        main.push_str(
            format!(
                " {conv:.1} * b_0.data[{stride}u * global_id.x + {idx}u] +",
                conv = conv,
                stride = stride[0],
                idx = i
            )
            .as_str(),
        );

        i += 1;
    }
    main.pop();
    main.push(';');
    main.push('}');
    main
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_map() {
        let (device, queue) = pollster::block_on(crate::ressource::request_device_queue());
        let data = [1.0, 2.0, 3.0, 4.0];
        let buffer = crate::ressource::create_buffer_init(&device, &data);

        let binding_group_entry = wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        };
        crate::compute::wrapper(
            &device,
            &queue,
            &[binding_group_entry],
            &[crate::compute::InnerType::ArrayVector],
            &crate::op::map(&"cos"),
            1,
            1,
            1,
        )
        .unwrap();

        let buffer_slice = buffer.slice(..);
        // Gets the future representing when `staging_buffer` can be read from
        let buffer_future = buffer_slice.map_async(wgpu::MapMode::Read);

        device.poll(wgpu::Maintain::Wait);
        if let Ok(()) = pollster::block_on(buffer_future) {
            // Gets contents of buffer
            let data = buffer_slice.get_mapped_range();
            // Since contents are got in bytes, this converts these bytes back to f32
            let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();

            let expected = vec![f32::cos(1.0), f32::cos(2.), f32::cos(3.), f32::cos(4.)];

            for (res, exp) in result.iter().zip(expected) {
                assert!((res - exp) < f32::EPSILON)
            }
            drop(data);
        } else {
            panic!("failed to run compute on gpu!")
        }
    }

    //#[test]
    // fn test_conv() {
    // let (device, queue) = pollster::block_on(crate::ressource::request_device_queue());
    // let data = [1.0, 2.0, 3.0, 4.0];
    // let buffer = crate::ressource::create_buffer_init(&device, &data);
    // let output = crate::ressource::create_buffer(&device, 4);
    // let binding_group_entry = [
    // wgpu::BindGroupEntry {
    // binding: 0,
    // resource: buffer.as_entire_binding(),
    // },
    // wgpu::BindGroupEntry {
    // binding: 1,
    // resource: output.as_entire_binding(),
    // },
    // ];

    // crate::compute::wrapper(
    // &device,
    // &queue,
    // &binding_group_entry,
    // &[
    // crate::compute::InnerType::Array,
    // crate::compute::InnerType::Array,
    // ],
    // &crate::op::conv(&[1.0, 2.0], &[1], false),
    // 2,
    // 1,
    // 1,
    // )
    // .unwrap();

    // let buffer_slice = output.slice(..);
    // // Gets the future representing when `staging_buffer` can be read from
    // let buffer_future = buffer_slice.map_async(wgpu::MapMode::Read);

    // device.poll(wgpu::Maintain::Wait);
    // if let Ok(()) = pollster::block_on(buffer_future) {
    // // Gets contents of buffer
    // let data = buffer_slice.get_mapped_range();
    // // Since contents are got in bytes, this converts these bytes back to f32
    // let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();

    // let expected = vec![5.0, 11.0];

    // for (res, exp) in result.iter().zip(expected) {
    // assert!((res - exp) < f32::EPSILON)
    // }
    // drop(data);
    // } else {
    // panic!("failed to run compute on gpu!")
    // }
    // }
    #[test]
    fn test_scan() {
        let (device, queue) = pollster::block_on(crate::ressource::request_device_queue());
        let data = vec![1.0; 1_024 as _];
        let buffer = crate::ressource::create_buffer_init(&device, &data);

        let binding_group_entry = [wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }];
        let mut i = 1;
        while i < 1_024 {
            let target = 1_024 / i / 2;

            crate::compute::wrapper(
                &device,
                &queue,
                &binding_group_entry,
                &[crate::compute::InnerType::Array],
                &crate::op::scan(1, i),
                target as u32,
                1,
                1,
            )
            .unwrap();
            i *= 2;
            // cpass.insert_debug_marker("compute collatz iterations");
        }

        // Note that we're not calling `.await` here.
        let buffer_slice = buffer.slice(..);
        // Gets the future representing when `staging_buffer` can be read from
        let buffer_future = buffer_slice.map_async(wgpu::MapMode::Read);

        device.poll(wgpu::Maintain::Wait);
        if let Ok(()) = pollster::block_on(buffer_future) {
            // Gets contents of buffer
            let data = buffer_slice.get_mapped_range();
            // Since contents are got in bytes, this converts these bytes back to f32
            let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
            assert_eq!(1_024f32, result[0]);
            drop(data);
        } else {
            panic!("failed to run compute on gpu!")
        }
    }
}
