import pytest
import numpy as np
from scipy.linalg import block_diag

np.random.seed(42)

# defaults
TOLERANCE = 1e-5
SHOTS = 10000


# global variables and functions
I = np.identity(2)
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
H = np.array([[1, 1], [1, -1]])/np.sqrt(2)
S = np.diag([1, 1j])
T = np.diag([1, np.exp(1j*np.pi/4)])

U = np.array(
    [
        [0.83645892 - 0.40533293j, -0.20215326 + 0.30850569j],
        [-0.23889780 - 0.28101519j, -0.88031770 - 0.29832709j],
    ]
)

single_qubit = [
    ("PauliX", X),
    ("PauliY", Y),
    ("PauliZ", Z),
    ("Hadamard", H),
    ("S", S),
    ("Sdg", S.conj().T),
    ("T", T),
    ("Tdg", T.conj().T)
]



phase_shift = lambda phi: np.array([[1, 0], [0, np.exp(1j*phi)]])
rx = lambda theta: np.cos(theta/2) * I + 1j * np.sin(-theta/2) * X
ry = lambda theta: np.cos(theta/2) * I + 1j * np.sin(-theta/2) * Y
rz = lambda theta: np.cos(theta/2) * I + 1j * np.sin(-theta/2) * Z
rot = lambda a, b, c: rz(c) @ (ry(b) @ rz(a))


single_qubit_param = [
    ("PhaseShift", phase_shift),
    ("RX", rx),
    ("RY", ry),
    ("RZ", rz)
]



SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
CZ = np.diag([1, 1, 1, -1])
U2 = np.array([[0, 1, 1, 1], [1, 0, 1, -1], [1, -1, 0, 1], [1, 1, -1, 0]]) / np.sqrt(3)


two_qubit = [
    ("CNOT", CNOT),
    ("SWAP", SWAP),
    ("CZ", CZ)
]

crz = lambda theta: np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, np.exp(-1j*theta/2), 0], [0, 0, 0, np.exp(1j*theta/2)]])


two_qubit_param = [
    ("CRZ", crz)
]


toffoli = np.diag([1 for i in range(8)])
toffoli[6:8, 6:8] = np.array([[0, 1], [1, 0]])
CSWAP = block_diag(I, I, SWAP)

three_qubit = [
    ("Toffoli", toffoli),
    ("CSWAP", CSWAP)
]

A = np.array([[1.02789352, 1.61296440 - 0.3498192j], [1.61296440 + 0.3498192j, 1.23920938 + 0j]])



@pytest.fixture
def tol(scope="session"):
    return TOLERANCE


@pytest.fixture
def shots(scope="session"):
    """default shots"""
    return SHOTS


@pytest.fixture
def init_state(scope="session"):
    def _init_state(n):
        state = np.random.random([2**n]) + np.random.random([2**n])*1j
        state /= np.linalg.norm(state)
        return state
    return _init_state
