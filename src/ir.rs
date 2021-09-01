// Computation nodes are comprised of a name, the name of
// an operator that it invokes, a list of named inputs,
// a list of named outputs, and a list of attributes.
struct Node {
    name: String,
    input: Vec<String>,
    output: Vec<String>,
    op_type: string,
    domain: String,
    attributes: HashMap<String, String>,
    doc_string: Option<String>,
}

// A graph is used to describe a side-effect-free computation (function).
// A serialized graph is comprised of a set of metadata fields, a list of
// model parameters, and a list of computation nodes.
struct Graph {
    name: String,
    node: Vec<Node>,
    initializer: HashMap<String, String>,
    doc_string: Option<String>,
    input: Vec<ValueInfo>,
    output: Vec<ValueInfo>,
    value_info: Vec<ValueInfo>,
}

struct ValueInfo {
    name: String,
    Type: String,
    doc_string: Option<String>,
}
