"""
Default parameters, commandline arguments and common routines for the unit tests.
"""
import unittest
import os
import sys
import logging
import argparse

import pennylane
from pennylane import numpy as np

# Make sure pennylane_qiskit is always imported from the same source distribution
# where the tests reside, not e.g. from site-packages.
# See https://docs.python-guide.org/en/latest/writing/structure/#test-suite
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pennylane_qiskit


# defaults
if 'DEVICE' in os.environ and os.environ['DEVICE'] is not None:
    DEVICE = os.environ['DEVICE']
else:
    DEVICE = "all"
OPTIMIZER = "GradientDescentOptimizer"
if DEVICE == "all" or DEVICE == "ibm":
    TOLERANCE = 3e-2
else:
    TOLERANCE = 1e-3

IBMQX_TOKEN = ''
if 'IBMQX_TOKEN' in os.environ and os.environ['IBMQX_TOKEN'] is not None:
    IBMQX_TOKEN = os.environ['IBMQX_TOKEN']


# set up logging
if "LOGGING" in os.environ:
    logLevel = os.environ["LOGGING"]
    print('Logging:', logLevel)
    numeric_level = getattr(logging, logLevel.upper(), 10)
else:
    numeric_level = 100

logging.getLogger().setLevel(numeric_level)
logging.captureWarnings(True)


def get_commandline_args():
    """Parse the commandline arguments for the unit tests.
    If none are given (e.g. when the test is run as a module instead of a script),
    the defaults are used.

    Returns:
      argparse.Namespace: parsed arguments in a namespace container
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', type=str, default=DEVICE,
                        help='Device(s) to use for tests.', choices=['simulator', 'ibm', 'classical', 'all'])
    parser.add_argument('-t', '--tolerance', type=float, default=TOLERANCE,
                        help='Numerical tolerance for equality tests.')
    parser.add_argument("--ibmqx_token",
                        help="IBM Quantum Experience token")
    parser.add_argument("--optimizer", default=OPTIMIZER, choices=pennylane.optimize.__all__,
                        help="optimizer to use")

    # HACK: We only parse known args to enable unittest test discovery without parsing errors.
    args, _ = parser.parse_known_args()
    return args


# parse any possible commandline arguments
ARGS = get_commandline_args()


class BaseTest(unittest.TestCase):
    """ABC for tests.
    Encapsulates the user-given commandline parameters for the test run as class attributes.
    """
    num_subsystems = None  #: int: number of wires for the device, must be overridden by child classes

    def setUp(self):
        self.args = ARGS
        self.tol = self.args.tolerance

    def logTestName(self):
        logging.info('{}'.format(self.id()))

    def assertAllElementsAlmostEqual(self, l, delta, msg=None):
        l = list(l)
        first = l.pop()
        for value in l:
            self.assertAllAlmostEqual(first, value, delta, msg)

    def assertAllAlmostEqual(self, first, second, delta, msg=None):
        """
        Like assertAlmostEqual, but works with arrays. All the corresponding elements have to be almost equal.
        """
        if isinstance(first, tuple):
            # check each element of the tuple separately (needed for when the tuple elements are themselves batches)
            if np.all([np.all(first[idx] == second[idx]) for idx, _ in enumerate(first)]):
                return
            if np.all([np.all(np.abs(first[idx] - second[idx])) <= delta for idx, _ in enumerate(first)]):
                return
        else:
            if np.all(first == second):
                return
            if np.all(np.abs(first - second) <= delta):
                return
        standardMsg = '{} != {} within {} delta'.format(first, second, delta)
        msg = self._formatMessage(msg, standardMsg)
        raise self.failureException(msg)

    def assertAllEqual(self, first, second, msg=None):
        """
        Like assertEqual, but works with arrays. All the corresponding elements have to be equal.
        """
        return self.assertAllAlmostEqual(first, second, delta=0.0, msg=msg)

    def assertAllTrue(self, value, msg=None):
        """
        Like assertTrue, but works with arrays. All the corresponding elements have to be True.
        """
        return self.assertTrue(np.all(value))
