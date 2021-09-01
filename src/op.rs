pub fn map(function: &str) -> String {
    format!(
        r#"
    b_0.data[global_id.x] = {function}(b_0.data[global_id.x]);
"#,
        function = function,
    )
}

pub fn scan(convolution: &[f32], stride: &[i32], recursive: bool) -> String {
    let recursive = if recursive { 0 } else { 1 };
    let mut main: String = format!(
        "    b_{recursive}.data[{stride}u * global_id.x] = ",
        stride = &stride[0] * 2,
        recursive = recursive
    );

    let mut i = 0;
    for conv in convolution {
        main.push_str(
            format!(
                " {conv:.2} * b_0.data[{stride}u * global_id.x + {stride_1}u * {idx}u] +",
                conv = conv,
                stride = stride[0] * 2,
                stride_1 = stride[0],
                idx = i
            )
            .as_str(),
        );

        i += 1;
    }
    main.pop();
    main.push(';');
    main
}

pub fn conv(convolution: &[f32], stride: &[i32], recursive: bool) -> String {
    let recursive = if recursive { 0 } else { 1 };
    let mut main: String = format!(
        "    b_{recursive}.data[{stride}u * global_id.x] = ",
        stride = &stride[0],
        recursive = recursive
    );

    let mut i = 0;
    for conv in convolution {
        main.push_str(
            format!(
                " {conv:.2} * b_0.data[{stride}u * (global_id.x + {idx}u)] +",
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
    main
}
