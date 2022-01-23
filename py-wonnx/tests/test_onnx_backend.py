# SPDX-License-Identifier: Apache-2.0

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import itertools
import os
import platform
import unittest
import onnx.backend.base
import onnx.backend.test

from onnx.backend.base import BackendRep, Device, DeviceType, namedtupledict
from onnx.backend.test.runner import BackendIsNotSupposedToImplementIt
import onnx.shape_inference
import onnx.version_converter
from typing import NamedTuple, Optional, Text, Any, Tuple, Sequence
from onnx import NodeProto, ModelProto, TensorProto
import numpy  # type: ignore

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

import wonnx
import numpy as np


class DummyRep(BackendRep):
    def __init__(self, inputs, outputs, outputs_shape, model):
        self.inputs = inputs
        self.outputs = outputs
        self.outputs_shape = outputs_shape
        self.session = wonnx.PySession.from_bytes(onnx._serialize(model))
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
            print(self.outputs_shape[item[0]])
            tmp_v = np.reshape(tmp_v, self.outputs_shape[item[0]])
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
        **kwargs  # type: Any
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

        if do_enforce_test_coverage_safelist(model):
            for node in model.graph.node:
                for i, output in enumerate(node.output):
                    if node.op_type == "Dropout" and i != 0:
                        continue
                    assert output in value_infos
                    tt = value_infos[output].type.tensor_type
                    assert tt.elem_type != TensorProto.UNDEFINED
                    for dim in tt.shape.dim:
                        assert dim.WhichOneof("value") == "dim_value"

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
if os.getenv("APPVEYOR"):
    backend_test.exclude(r"(test_vgg19|test_zfnet)")
if platform.architecture()[0] == "32bit":
    backend_test.exclude(r"(test_vgg19|test_zfnet|test_bvlc_alexnet)")

# The test cases excluded below should be considered permanent restrictions
# based on the TensorFlow implementation. Unimplemented operators will raise
# a BackendIsNotSupposedToImplementIt exception so that their test cases
# will pass and show a verbose message stating it was effectively skipped.

# https://github.com/onnx/onnx/issues/349
backend_test.exclude(r"[a-z,_]*GLU[a-z,_]*")

# TF does not support dialation and strides at the same time:
# Will produce strides > 1 not supported in conjunction with dilation_rate > 1
backend_test.exclude(r"[a-z,_]*dilated_strided[a-z,_]*")
backend_test.exclude(r"[a-z,_]*Conv2d_dilated[a-z,_]*")

# TF does not have column major max_pool_with_argmax
backend_test.exclude(
    r"[a-z,_]*maxpool_with_argmax_2d_precomputed_strides[a-z,_]*"
)

# PRelu OnnxBackendPyTorchConvertedModelTest has wrong dim for broadcasting
backend_test.exclude(r"[a-z,_]*PReLU_[0-9]d_multiparam[a-z,_]*")

# TF does not support int8, int16, uint8, uint16, uint32, uint64 for
# tf.floormod and tf.truncatemod
backend_test.exclude(r"test_mod_[a-z,_]*uint[0-9]+")
backend_test.exclude(r"test_mod_[a-z,_]*int(8|(16))+")

# TF doesn't support most of the attributes in resize op
# test_node.py will cover the test
backend_test.exclude(r"test_resize_[a-z,_]*")

# range is using loop in the model test but all the outputs datatype are
# missing in the body attribute of the loop
backend_test.exclude(r"test_range_float_type_positive_delta_expanded[a-z,_]*")
backend_test.exclude(r"test_range_int32_type_negative_delta_expanded[a-z,_]*")

# skip all the cumsum testcases because all the axis in the testcases
# are created as a 1-D 1 element tensor, but the spec clearly state
# that axis should be a 0-D tensor(scalar)
backend_test.exclude(r"test_cumsum_[a-z,_]*")

# TF session run does not support sequence/RaggedTensor as model inputs
backend_test.exclude(r"test_loop13_seq[a-z,_]*")

# TF minimum/maximum do not support uint64 when auto-cast is False (default)
backend_test.exclude(r"test_min_uint64_[a-z,_]*")
backend_test.exclude(r"test_max_uint64_[a-z,_]*")

backend_test.exclude(r"[a-z,_]*Upsample[a-z,_]*")

if "TRAVIS" in os.environ:
    backend_test.exclude("test_vgg19")
    backend_test.exclude("zfnet512")

# These following tests fails by a tiny margin with onnx<1.2:
backend_test.exclude("test_operator_add_broadcast_cpu")
backend_test.exclude("test_operator_add_size1_broadcast_cpu")
backend_test.exclude("test_operator_add_size1_right_broadcast_cpu")
backend_test.exclude("test_operator_add_size1_singleton_broadcast_cpu")
backend_test.exclude("test_averagepool_3d_default_cpu")
# Do not support consumed flag:
backend_test.exclude("test_batch_normalization")
# Do not support RNN testing on onnx<1.2 due to incorrect tests:
backend_test.exclude(r"test_operator_rnn_cpu")
backend_test.exclude(r"test_operator_lstm_cpu")
backend_test.exclude(r"test_operator_rnn_single_layer_cpu")

# The onnx test for cast, float to string, does not work
backend_test.exclude(r"[a-z,_]*cast[a-z,_]*")


# Do not support dilations != 1 for ConvTranspose, test is added in opset 10
backend_test.exclude(r"[a-z,_]*convtranspose_dilations[a-z,_]*")

# Concat from sequence with new_axis=1 not supported
backend_test.exclude(r"test_sequence_model5_[a-z,_]*")

# Fails rounding tolerance
backend_test.exclude(r"test_gru_seq_length_[a-z,_]*")

# TF pow does not support uint64 when auto-cast is False (default)
backend_test.exclude(r"test_pow_types_float[0-9]+_uint64+_[a-z,_]*")

# TF session run does not support sequence/RaggedTensor as model inputs
backend_test.exclude(r"test_sequence_insert+_[a-z,_]*")

# Exclude tests for Dropout training that have randomness dependent on
# the different implementations
backend_test.exclude("test_training_dropout_default_[a-z,_]*")
backend_test.exclude("test_training_dropout_[a-z,_]*")
backend_test.exclude("test_training_dropout_default_mask_[a-z,_]*")
backend_test.exclude("test_training_dropout_mask_[a-z,_]*")

# TF module can't run gru, lstm, rnn in one session using custom variables
backend_test.exclude(r"test_gru_[a-z,_]*")
backend_test.exclude(r"test_lstm_[a-z,_]*")
backend_test.exclude(r"test_rnn_[a-z,_]*")
backend_test.exclude(r"test_simple_rnn_[a-z,_]*")

# TF doesn't support auto_pad=SAME_LOWER for Conv and ConvTranspose
backend_test.exclude(r"test_conv_with_autopad_same_[a-z,_]*")
backend_test.exclude(r"test_convtranspose_autopad_same_[a-z,_]*")

# Exclude non-deterministic tests
backend_test.exclude(r"test_bernoulli_[a-z,_]*")

# # onnx backend test support seq from 1.11 #3731
backend_test.exclude(r"test_optional_get_element[a-z,_]*")
backend_test.exclude(r"test_optional_has_element[a-z,_]*")

# Exclude BatchNormalization with training_mode=1 tests
backend_test.exclude(r"test_batchnorm_epsilon_training_mode[a-z,_]*")
backend_test.exclude(r"test_batchnorm_example_training_mode[a-z,_]*")

# ONNX 1.9.0 test case does not support sequence
backend_test.exclude(r"[a-z,_]*identity_sequence_[a-z,_]*")

# onnx backend test support seq from 1.11 #3731
backend_test.exclude(r"[a-z,_]*identity_opt_[a-z,_]*")

backend_test.exclude(r"test_if_seq[a-z,_]*")

backend_test.exclude(r"[a-z,_]*if_opt_[a-z,_]*")
backend_test.exclude(r"[a-z,_]*loop16_seq_none_[a-z,_]*")

backend_test.exclude(r"test_gridsample_[a-z,_]*")

# import all test cases at global scope to make them visible to python.unittest
globals().update(backend_test.test_cases)
globals().update(backend_test.enable_report().test_cases)

if __name__ == "__main__":
    unittest.main()
