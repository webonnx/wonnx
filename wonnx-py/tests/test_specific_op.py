# SPDX-License-Identifier: MIT OR Apache-2.0
import onnx.backend.test

pytest_plugins = ("onnx.backend.test.report",)

import os
import unittest
from test_onnx_backend import DummyBackend

OP_TESTED = os.environ["OP_TESTED"]

backend_test = onnx.backend.test.BackendTest(DummyBackend, __name__)
backend_test.include(f"test_{OP_TESTED}_[a-z,_]*")
globals().update(backend_test.enable_report().test_cases)

if __name__ == "__main__":
    unittest.main()
