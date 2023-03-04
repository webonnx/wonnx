use std::{collections::HashMap, fmt::Display};

use wonnx::onnx::{GraphProto, ModelProto};

use crate::types::TraceOptions;

enum Source {
    NodeOutput {
        node_name: String,
        node_index: usize,
        op_type: String,
        output_name: String,
        output_index: usize,
    },
    Initializer {
        name: String,
    },
    Input {
        name: String,
    },
}

impl Display for Source {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Source::NodeOutput {
                node_name,
                node_index,
                op_type,
                output_name,
                output_index,
            } => write!(
                f,
                "{output_name}: output #{output_index} of node #{node_index} '{node_name}' ({op_type})"
            ),
            Source::Initializer { name } => write!(f, "{name}: initializer"),
            Source::Input { name } => write!(f, "{name}: input"),
        }
    }
}

pub fn trace_command(model: &ModelProto, opt: &TraceOptions) {
    // Build a map of all outputs
    let mut outputs: HashMap<String, Source> = HashMap::new();
    let graph = model.get_graph();

    for (node_index, node) in graph.node.iter().enumerate() {
        let node_name = node.get_name();
        for (output_index, output_name) in node.get_output().iter().enumerate() {
            outputs.insert(
                output_name.clone(),
                Source::NodeOutput {
                    node_name: node_name.to_string(),
                    node_index,
                    op_type: node.get_op_type().to_string(),
                    output_name: output_name.clone(),
                    output_index,
                },
            );
        }
    }

    for input in graph.input.iter() {
        outputs.insert(
            input.get_name().to_string(),
            Source::Input {
                name: input.get_name().to_string(),
            },
        );
    }

    for initializer in graph.initializer.iter() {
        outputs.insert(
            initializer.get_name().to_string(),
            Source::Initializer {
                name: initializer.get_name().to_string(),
            },
        );
    }

    // Print race
    if let Some(source) = outputs.get(&opt.output_name) {
        source.trace(&outputs, graph, 0, opt.maximum_depth);
    }
}

impl Source {
    fn trace(
        &self,
        map: &HashMap<String, Source>,
        graph: &GraphProto,
        depth: usize,
        max_depth: Option<usize>,
    ) {
        if let Some(md) = max_depth {
            if depth >= md {
                return;
            }
        }
        let tab = "| ";
        let tabs = tab.repeat(depth);
        println!("{tabs}+ {}", self);

        match self {
            Source::NodeOutput { node_index, .. } => {
                let node = &graph.get_node()[*node_index];
                for input in &node.input {
                    if let Some(source) = map.get(input) {
                        source.trace(map, graph, depth + 1, max_depth);
                    } else {
                        println!("{tabs}{tab}* '{}' NOT FOUND", input);
                    }
                }
            }
            Source::Initializer { .. } | Source::Input { .. } => (),
        }
    }
}
