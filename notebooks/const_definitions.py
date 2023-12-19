"""
Copies the definitions for solver logger codes for use in python notebooks.
"""

from enum import IntEnum, auto

class OptimizerTestCodes(IntEnum):
    CONVERGENCE_TEST_PASS = 1
    VALUE_TEST_FAIL = 2
    VALUE_TEST_PASS = 4
    GRADIENT_TEST_FAIL = 8
    GRADIENT_TEST_PASS = 16
    HESSIAN_NEGATIVE_DEFINITE = 32
    HESSIAN_MAYBE_INDEFINITE = 64
    HESSIAN_POSITIVE_DEFINITE = 128


class OptimizerEvents(IntEnum):
    OPTIMIZATION_BEGIN = 0
    OPTIMIZATION_COMPLETE = auto()
    TASK_BEGIN = auto()
    TASK_COMPLETE = auto()
    VALUE_TEST = auto()
    CONVERGENCE_TEST = auto()
    GRADIENT_TEST = auto()
    HESSIAN_TEST = auto()
    ALL_TESTS = auto()

class SolverEvents(IntEnum):
    SOLVER_BEGIN = 0
    SOLVER_COMPLETE = auto()
    TIME_STEP_NO_EVENT = auto()
    TIME_STEP_EVENT_CORRECTED = auto()
    OPTIMIZE = auto()
    OPTIMUM_CHANGE = auto()