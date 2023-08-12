# SPDX-License-Identifier: MIT OR Apache-2.0

import unittest
import onnx.backend.base
import onnx.backend.test

from onnx.backend.base import BackendRep, Device, DeviceType
import onnx.shape_inference
import onnx.version_converter
from typing import Optional, Text, Any
from onnx import ModelProto, TensorProto
import numpy as np
import wonnx

# The following just executes the fake backend through the backend test
# infrastructure. Since we don't have full reference implementation of all ops
# in ONNX repo, it's impossible to produce the proper results. However, we can
# run 'checker' (that's what base Backend class does) to verify that all tests
# fed are actually well-formed ONNX models.
#
# If everything is fine, all the tests would be marked as "skipped".
#
# We don't enable report in this test because the report collection logic itself
# fails when models are mal-formed.


# This is a pytest magic variable to load extra plugins
pytest_plugins = ("onnx.backend.test.report",)


class DummyRep(BackendRep):
    def __init__(self, inputs, outputs, outputs_shape, model):
        self.inputs = inputs
        self.outputs = outputs
        self.outputs_shape = outputs_shape
        self.session = wonnx.Session.from_bytes(onnx._serialize(model))
        self.rtol = 1
        pass

    def run(self, inputs, rtol=1.0, **kwargs):
        dicts = {}
        for k, v in zip(self.inputs, inputs):
            if isinstance(v, np.ndarray):
                dicts[k] = v.flatten()
            else:
                tmp_v = np.array(v)
                np.reshape(tmp_v, self.outputs_shape[k])
                dicts[k] = tmp_v

        results = self.session.run(dicts)

        outputs = []
        for item in results.items():
            tmp_v = np.array(item[1])
            tmp_v = np.reshape(tmp_v, self.outputs_shape[item[0]])
            if tmp_v.dtype == "float64":
                tmp_v = tmp_v.astype("float32")
            outputs.append(tmp_v)
        return outputs


class DummyBackend(onnx.backend.base.Backend):
    @classmethod
    def prepare(
        cls,
        model,  # type: ModelProto
        inputs,
        device="CPU",  # type: Text
        **kwargs,  # type: Any
    ):  # type: (...) -> Optional[onnx.backend.base.BackendRep]
        super(DummyBackend, cls).prepare(model, device, **kwargs)

        # test shape inference
        model = onnx.shape_inference.infer_shapes(model)
        inputs = [input.name for input in model.graph.input]
        outputs = [output.name for output in model.graph.output]

        outputs_shape = {}
        for output in model.graph.output:
            outputs_shape[output.name] = [
                shape.dim_value for shape in output.type.tensor_type.shape.dim
            ]

        return DummyRep(
            inputs=inputs,
            outputs=outputs,
            model=model,
            outputs_shape=outputs_shape,
        )

    @classmethod
    def supports_device(cls, device):  # type: (Text) -> bool
        d = Device(device)
        if d.type == DeviceType.CPU:
            return True
        return False


test_coverage_safelist = set(
    [
        "bvlc_alexnet",
        "densenet121",
        "inception_v1",
        "inception_v2",
        "resnet50",
        "shufflenet",
        "SingleRelu",
        "squeezenet_old",
        "vgg19",
        "zfnet",
    ]
)


def do_enforce_test_coverage_safelist(model):  # type: (ModelProto) -> bool
    if model.graph.name not in test_coverage_safelist:
        return False
    for node in model.graph.node:
        if node.op_type in set(["RNN", "LSTM", "GRU"]):
            return False
    return True


backend_test = onnx.backend.test.BackendTest(DummyBackend, __name__)



backend_test.include(f"test_constant_cpu")
backend_test.include(f"test_conv_[a-z,_]*")
backend_test.include(f"test_Conv2d[a-z,_]*")
backend_test.include(f"test_abs_[a-z,_]*")
backend_test.include(f"test_acos_[a-z,_]*")
backend_test.include(f"test_atan_[a-z,_]*")
backend_test.include(f"test_ceil_[a-z,_]*")
backend_test.include(f"test_cos_[a-z,_]*")
backend_test.include(f"test_exp_[a-z,_]*")
backend_test.include(f"test_floor_[a-z,_]*")
backend_test.include(f"test_mul_bcast_[a-z,_]*")
backend_test.include(f"test_div_bcast_[a-z,_]*")
backend_test.include(f"test_add_bcast_[a-z,_]*")
backend_test.include(f"test_sub_bcast_[a-z,_]*")
backend_test.include(f"test_pow_bcast_[a-z,_]*")
backend_test.include(f"test_transpose[a-z,_]*")
backend_test.include(f"test_neg_[a-z,_]*")
backend_test.include(f"test_reciprocal_[a-z,_]*")
backend_test.include(f"test_shape_[a-z,_]*")
backend_test.include(f"test_size_[a-z,_]*")
backend_test.include(f"test_celu_[a-z,_]*")

# For these we only test the default version, as we don't support the bool type
backend_test.include(f"test_prelu_broadcast_cpu$")
backend_test.include(f"test_elu_cpu$")
backend_test.include(f"test_relu_cpu$")
backend_test.include(f"test_leakyrelu_default_cpu$")

# Don't support 'bool' type
# backend_test.include(f"test_and_bcast[a-z0-9,_]*")
# backend_test.include(f"test_or_bcast[a-z0-9,_]*")
# backend_test.include(f"test_equal_bcast_[a-z,_]*")
# backend_test.include(f"test_greater_bcast_[a-z,_]*")

# Disable tests for ReduceSum because ReduceSum accepts the 'axes' list as input instead of as an attribute, and the test
# case sets the 'axes' input dynamically, which we don't support (yet?).
# backend_test.include(f"test_reduce_sum_[a-z,_]*")
#backend_test.include(f"test_reduce_mean_[a-z,_]*")
#backend_test.include(f"test_reduce_l1_[a-z,_]*")
#backend_test.include(f"test_reduce_l2_[a-z,_]*")
#backend_test.include(f"test_reduce_min_[a-z,_]*")
#backend_test.include(f"test_reduce_prod_[a-z,_]*")
#backend_test.include(f"test_reduce_sum_square_[a-z,_]*")
#backend_test.include(f"test_reduce_max_[a-z,_]*")
#backend_test.include(f"test_reduce_log_sum_[a-z,_]*")
#backend_test.include(f"test_reduce_log_sum_exp_[a-z,_]*")

# Takes dynamic input, we don't support that yet
# backend_test.include(f"test_constantofshape_[a-z,_]*")

# Aggregation Test
backend_test.include(f"test_maxpool_2d_[a-z,_]*_[a-z,_]*")
backend_test.include(f"test_averagepool_2d_[a-z,_]*")
backend_test.include(f"test_globalaveragepool_[a-z,_]*")

# Pow: not supported are:
# - test_pow_bcast* (Pow with broadcast!=0 is not implemented)
# - test_pow_types (WGSL doesn't support pow(x,y) with non-f32 arguments)
backend_test.include(f"^test_pow_example")
backend_test.include(f"^test_pow_cpu$")

# Softmax
# For some reason, these test cases are expanded to "_expanded_cpu" (they appear to do Softmax followed by ReduceMax and
# some other operations) which currently appear to fail. Therefore only execute the test cases specific to Softmax for now
backend_test.include(f"test_softmax_axis_0_cpu$")
backend_test.include(f"test_softmax_axis_1_cpu$")
backend_test.include(f"test_softmax_axis_2_cpu$")
backend_test.include(f"test_softmax_large_number_cpu$")
backend_test.include(f"test_softmax_example_cpu$")
backend_test.include(f"test_softmax_negative_axis_cpu$")
backend_test.include(f"test_softmax_default_axis_cpu$")

# ConvTranspose
# We only have partial attribute support right now, so we hand select a few test cases limited to the supported ones
backend_test.include(f"test_convtranspose$")
backend_test.include(f"test_convtranspose_pads$")

globals().update(backend_test.enable_report().test_cases)

if __name__ == "__main__":
    unittest.main()
