{%- include "structs.wgsl" -%}

// Input tensor, shape NxCxHxW
@group(0) @binding(0)
var<storage, read> input_tensor: Array;

// Kernel weight tensor, shape CxM/groupxkHxkW
@group(0) @binding(1)
var<storage, read> input_kernel_weights: Array;

{% if i_lens | length == 3 -%}
    @group(0) @binding(2)
    var<storage, read> input_bias: Array;

    @group(0) @binding(3)
    var<storage, read_write> output_0: Array;
{%- else -%}
    @group(0) @binding(2)
    var<storage, read_write> output_0: Array;
{%- endif %}

{% set input_shape = i_shape[0] %}
{% set input_chunks = i_chunks[0] %}
{% set kernel_shape = i_shape[1] %}
{% set kernel_chunks = i_chunks[1] %}

@compute @workgroup_size({{ workgroup_size_x }}, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let output_idx = global_id.x;

    if (output_idx < {{ o_lens[0] }}u) {
        // Calculate the output coordinates we are responsible for
        let batch = output_idx / {{ o_chunks[0][0] }}u;
        var rest = output_idx % {{ o_chunks[0][0] }}u;

        let channel = rest / {{ o_chunks[0][1] }}u;
        rest = rest % {{ o_chunks[0][1] }}u;

        let y = rest / {{ o_chunks[0][2] }}u;
        let x = rest % {{ o_chunks[0][2] }}u;

        // The actual output is a slice of the full output,
        // where the given padding values are removed on each end.
        // We don't need to worry about this at the upper coordinate end,
        // but we need to consider it on the lower end and calculate
        // virtual output coordinates to calculate the input coordinate range later.
        let unpadded_y = y + {{ pads[0] }}u;
        let unpadded_x = x + {{ pads[1] }}u;

        let sample_root_index = batch * {{ input_chunks[0] }}u;

        // Calculate the input coordinate range for our output coordinate
        let min_in_y = select(0u, (unpadded_y - {{ kernel_shape[2] }}u) / {{ stride[0] }}u, unpadded_y > {{ kernel_shape[2] }}u);
        let max_in_y = select({{ input_shape[2] }}u - 1u, unpadded_y / {{ stride[0] }}u, unpadded_y / {{ stride[0] }}u < {{ input_shape[3] }}u);
        let min_in_x = select(0u, (unpadded_x - {{ kernel_shape[3] }}u) / {{ stride[1] }}u, unpadded_x > {{ kernel_shape[3] }}u);
        let max_in_x = select({{ input_shape[3] }}u - 1u, unpadded_x / {{ stride[1] }}u, unpadded_x / {{ stride[1] }}u < {{ input_shape[3] }}u);

        var result: Scalar = Scalar();

        // Now, go over each input channel and apply the corresponing kernel for that channel
        // to calculate the output piece by piece.
        for(var ichannel: u32 = 0u; ichannel < {{ input_shape[1] }}u; ichannel = ichannel + 1u) {
            // Base index for the 2D data in the input data
            let base_index = sample_root_index + ichannel * {{ input_chunks[1] }}u;
            // Get the starting position of the kernel for the given input and output channel
            let base_kernel_index = ichannel *{{ kernel_chunks[0] }}u + channel * {{ kernel_chunks[1] }}u;

            // Iterate of all potential input values
            for(var in_y: u32 = min_in_y; in_y <= max_in_y; in_y = in_y + 1u) {
                for(var in_x: u32 = min_in_x; in_x <= max_in_x; in_x = in_x + 1u) {
                    let kernel_y = unpadded_y - (in_y * {{ stride[0] }}u);
                    let kernel_x = unpadded_x - (in_x * {{ stride[1] }}u);

                    if(kernel_y < {{ kernel_shape[2] }}u && kernel_x < {{ kernel_shape[3] }}u) {
                        result = result + (input_tensor.data[base_index + (in_y * {{ input_chunks[2] }}u) + in_x]
                                           * input_kernel_weights.data[base_kernel_index + kernel_y * {{ kernel_chunks[2] }}u + kernel_x]);
                    }
                }
            }
        }
        {% if i_lens | length == 3 -%}
            // Apply Bias if specified
            result = result + input_bias.data[channel];
        {%- endif %}

        output_0.data[output_idx] = result;
    }
}
