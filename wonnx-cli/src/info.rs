use human_bytes::human_bytes;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use prettytable::{row, table, Table};
use wonnx::{
    onnx::{GraphProto, ModelProto, NodeProto, ValueInfoProto},
    utils::{ScalarType, Shape},
    WonnxError,
};

use crate::utils::ValueInfoProtoUtil;

fn dimensions_infos(
    graph_proto: &GraphProto,
) -> Result<HashMap<String, Option<Shape>>, wonnx::WonnxError> {
    let mut shapes_info = HashMap::new();

    for info in graph_proto.get_input() {
        let shape = info.get_shape().ok();
        shapes_info.insert(info.get_name().to_string(), shape);
    }

    for info in graph_proto.get_output() {
        let shape = info.get_shape().ok();
        shapes_info.insert(info.get_name().to_string(), shape);
    }

    for info in graph_proto.get_value_info() {
        let shape = info.get_shape().ok();
        shapes_info.insert(info.get_name().to_string(), shape);
    }

    for info in graph_proto.get_initializer() {
        let shape = Shape::from(
            ScalarType::from_i32(info.get_data_type())?,
            &info
                .get_dims()
                .iter()
                .map(|x| *x as usize)
                .collect::<Vec<usize>>(),
        );
        shapes_info.insert(info.get_name().to_string(), Some(shape));
    }

    Ok(shapes_info)
}

fn node_identifier(index: usize, node: &NodeProto) -> String {
    format!("\"{} {}\"", index, node.get_op_type())
}

pub fn print_graph(model: &ModelProto) {
    println!("strict digraph {{");
    let graph = model.get_graph();

    let mut outputs: HashMap<String, String> = HashMap::new();
    for (index, node) in graph.get_node().iter().enumerate() {
        for output in node.get_output() {
            outputs.insert(output.clone(), node_identifier(index, node));
        }
    }

    // Special values
    let model_inputs: HashSet<&str> = graph.get_input().iter().map(|v| v.get_name()).collect();
    let initializers: HashSet<&str> = graph
        .get_initializer()
        .iter()
        .map(|x| x.get_name())
        .collect();

    for (index, node) in graph.get_node().iter().enumerate() {
        for input in node.get_input() {
            if initializers.contains(input.as_str()) {
                continue;
            }

            if model_inputs.contains(input.as_str()) {
                println!("\t\"Input {}\" -> {}", input, node_identifier(index, node));
            }
            // Find input node
            else if let Some(out_from_node) = outputs.get(input) {
                println!("\t{} -> {}", out_from_node, node_identifier(index, node));
            } else {
                println!(
                    "\t\"Unknown: {}\" -> {}",
                    input,
                    node_identifier(index, node)
                );
            }
        }
    }
    println!("}}");
}

pub fn sizes_table(model: &ModelProto) -> Result<Table, WonnxError> {
    let mut input_size: usize = 0;
    for input in model.get_graph().get_input() {
        input_size += input
            .get_shape()
            .map(|x| x.buffer_bytes_aligned())
            .unwrap_or(0);
    }

    let mut output_size: usize = 0;
    for output in model.get_graph().get_output() {
        output_size += output
            .get_shape()
            .map(|x| x.buffer_bytes_aligned())
            .unwrap_or(0);
    }

    let mut intermediate_size: usize = 0;
    for output in model.get_graph().get_value_info() {
        intermediate_size += output
            .get_shape()
            .map(|x| x.buffer_bytes_aligned())
            .unwrap_or(0);
    }

    let mut initializer_size: usize = 0;
    for info in model.get_graph().get_initializer() {
        let shape = Shape::from(
            ScalarType::from_i32(info.get_data_type())?,
            &info
                .get_dims()
                .iter()
                .map(|x| *x as usize)
                .collect::<Vec<usize>>(),
        );
        initializer_size += shape.buffer_bytes_aligned();
    }

    Ok(table![
        [b->"Inputs",  r->human_bytes(input_size as f64)],
        [b->"Outputs", r->human_bytes(output_size as f64)],
        [b->"Intermediate", r->human_bytes(intermediate_size as f64)],
        [b->"Weights", r->human_bytes(initializer_size as f64)],
        [b->"Total", r->human_bytes((input_size + output_size + intermediate_size + initializer_size) as f64)]
    ])
}

fn datatype_to_string(dt: &ValueInfoProto) -> String {
    dt.data_type()
        .map(|k| k.to_string())
        .unwrap_or_else(|_| String::from("(unknown)"))
}

pub fn info_table(model: &ModelProto) -> Result<Table, WonnxError> {
    // List initializers
    let initializer_names: HashSet<String> = model
        .get_graph()
        .get_initializer()
        .iter()
        .map(|it| it.get_name().to_string())
        .collect();

    let mut inputs_table = Table::new();
    inputs_table.add_row(row![b->"Name", b->"Description", b->"Shape", b->"Type"]);
    let inputs = model.get_graph().get_input();
    for i in inputs {
        if !initializer_names.contains(i.get_name()) {
            inputs_table.add_row(row![
                i.get_name(),
                i.get_doc_string(),
                i.dimensions_description()
                    .iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<String>>()
                    .join(" x "),
                datatype_to_string(i)
            ]);
        }
    }

    let mut outputs_table = Table::new();
    outputs_table.add_row(row![b->"Name", b->"Description", b->"Shape", b->"Type"]);

    let outputs = model.get_graph().get_output();
    for i in outputs {
        outputs_table.add_row(row![
            i.get_name(),
            i.get_doc_string(),
            i.dimensions_description()
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<String>>()
                .join(" x "),
            datatype_to_string(i)
        ]);
    }

    let opset_string = model
        .get_opset_import()
        .iter()
        .map(|os| format!("{} {}", os.get_version(), os.get_domain()))
        .collect::<Vec<String>>()
        .join(";");

    // List ops used
    struct Usage {
        attributes: BTreeSet<String>,
    }

    let mut usage: BTreeMap<&str, Usage> = BTreeMap::new();

    let graph = model.get_graph();
    for node in graph.get_node() {
        let usage = if let Some(usage) = usage.get_mut(node.get_op_type()) {
            usage
        } else {
            usage.insert(
                node.get_op_type(),
                Usage {
                    attributes: BTreeSet::new(),
                },
            );
            usage.get_mut(node.get_op_type()).unwrap()
        };

        node.get_attribute().iter().for_each(|a| {
            let value = match a.get_field_type() {
                wonnx::onnx::AttributeProto_AttributeType::FLOAT => format!("{}", a.get_f()),
                wonnx::onnx::AttributeProto_AttributeType::INT => format!("{}", a.get_i()),
                wonnx::onnx::AttributeProto_AttributeType::STRING => {
                    String::from_utf8(a.get_s().to_vec()).unwrap()
                }
                _ => format!("<{:?}>", a.get_field_type()),
            };

            let name = format!("{}={}", a.get_name(), value);
            usage.attributes.insert(name);
        });
    }

    let mut usage_table = Table::new();
    usage_table.add_row(row![b->"Op", b->"Attributes"]);
    for (op, usage) in usage.iter() {
        let attrs = usage
            .attributes
            .iter()
            .cloned()
            .collect::<Vec<String>>()
            .join("\n");
        usage_table.add_row(row![op, attrs]);
    }

    // List node inputs/outputs that don't have dimensions
    let shapes = dimensions_infos(graph)?;
    for node in graph.get_node() {
        for input in node.get_input() {
            match shapes.get(input) {
                None => log::error!(
                    "Node '{}' input '{}' has unknown shape",
                    node.get_name(),
                    input
                ),
                Some(_) => {}
            }
        }
    }

    let size_table = sizes_table(model)?;

    Ok(table![
        [b->"Model version", model.get_model_version()],
        [b->"IR version", model.get_ir_version()],
        [b->"Producer name", model.get_producer_name()],
        [b->"Producer version", model.get_producer_version()],
        [b->"Opsets", opset_string],
        [b->"Inputs", inputs_table],
        [b->"Outputs", outputs_table],
        [b->"Ops used", usage_table],
        [b->"Memory usage", size_table]
    ])
}
