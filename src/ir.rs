use std::collections::HashMap;
use std::fmt;

use rayon::str::Bytes;

// Computation nodes are comprised of a name, the name of
// an operator that it invokes, a list of named inputs,
// a list of named outputs, and a list of attributes.
pub struct Node {
    pub name: String,
    pub input: Vec<String>,
    pub output: Vec<String>,
    pub op_type: String,
    // domain: String,
    pub attribute: Vec<Attribute>,
    // doc_string: Option<String>,
}

pub struct Attribute {
    name: String,
    doc_string: Option<String>,
    attribute_type: String,
    f: Option<f32>,
    i: Option<i64>,
    s: Option<u8>,
    t: Option<f32>,
}
// A graph is used to describe a side-effect-free computation (function).
// A serialized graph is comprised of a set of metadata fields, a list of
// model parameters, and a list of computation nodes.
struct Graph {
    name: String,
    node: Vec<Node>,
    initializer: Vec<TensorType>,
    doc_string: Option<String>,
    input: Vec<ValueInfo>,
    output: Vec<ValueInfo>,
    value_info: Vec<ValueInfo>,
}

struct TensorType {
    elem_type: u32,
    shape: Shape,
}

struct Shape {
    dim: Vec<Dim>,
}

struct Dim {
    dim_value: u32,
}

struct ValueInfo {
    name: String,
    value_type: String,
    doc_string: Option<String>,
}

impl fmt::Display for Node {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.op_type.as_str() {
            "Abs" => {
                let s = crate::op::as_vec(
                    format!(
                        "{output}.data[global_id.x][index_mat] = {op_type}({input}.data[global_id.x][index_mat]);",
                        input = self.input[0],
                        output = self.output[0],
                        op_type = self.op_type.to_lowercase()
                    )
                    .as_str(),
                );
                write!(f, "{}", s)
            }
            "Add" => {
                let s = crate::op::as_vec(
                    format!(
                        "{output}.data[global_id.x][index_mat] = {input_0}.data[global_id.x][index_mat] + {input_1}.data[global_id.x][index_mat];",
                        input_0 = self.input[0],
                        input_1 = self.input[1],
                        output = self.output[0]
                    )
                    .as_str(),
                );
                write!(f, "{}", s)
            }
            "MatMul" => {
                let s = format!(
                    r#"
            var i: u32 = global_id.x * {len}u + global_id.y;
		    var tmpSum = {output}.data[i];
		    var product = {output}.data[i];
		    for(var k: u32 = 0u; k < {len}u; k = k + 1u) {{
			product = {input_0}.data[global_id.x * {len}u + k] * {input_1}.data[global_id.y * {len}u + k];
			for(var index_mat: u32 = 0u; index_mat < 4u; index_mat = index_mat + 1u) {{
			    tmpSum[index_mat] = tmpSum[index_mat] + product[index_mat];
			}}
		    }}
		    {output}.data[i] = tmpSum;"#,
                    input_0 = self.input[0],
                    input_1 = self.input[1],
                    output = self.output[0],
                    len = self
                        .attribute
                        .iter()
                        .filter(|x| x.name == "len")
                        .next()
                        .expect("Length attribute not found for matrix multiplication")
                        .i
                        .expect("Attribute len did not have integer value")
                );
                write!(f, "{}", s)
            }
            "Sum" => {
                unimplemented!()
            }
            _ => unimplemented!(),
        }
    }
}
