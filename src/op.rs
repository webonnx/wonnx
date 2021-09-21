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
