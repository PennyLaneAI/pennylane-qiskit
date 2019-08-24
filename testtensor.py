import pennylane as qml
from pennylane_qiskit import AerDevice

import numpy as np

dev = AerDevice(3, shots=8000, backend="qasm_simulator")

state = np.array(
    [
        0.185631 - 0.0835414 * 1j,
        -0.288625 - 0.165853 * 1j,
        0.463708 - 0.0219674 * 1j,
        0.099824 + 0.38887 * 1j,
        -0.180463 - 0.00272388 * 1j,
        -0.0987177 - 0.392217 * 1j,
        -0.44149 + 0.0779663 * 1j,
        -0.26467 - 0.0621438 * 1j,
    ]
)

state /= np.linalg.norm(state)

A = np.array([[1.15168 +0.*1j,  -0.637756-0.256156*1j,    0.899354 -0.223397*1j,    -0.894274+0.507884*1j],
[-0.637756+0.256156*1j,    -1.91632+0.*1j,   0.345997 -0.993101*1j,    -0.611875+1.24521*1j],
[0.899354 +0.223397*1j,    0.345997 +0.993101*1j,    0.33557 +0.*1j,   0.48734 +0.989581*1j],
[-0.894274-0.507884*1j,    -0.611875-1.24521*1j, 0.48734 -0.989581*1j, -0.409944+0.*1j]])

dev.apply("QubitStateVector", wires=[0, 1, 2], par=[state])

class Observable:
    return_type = "expectation"
    name = ["PauliY", "Hermitian"]
    wires = [[0], [1, 2]]
    parameters = [[], [A]]
    tensor = True

obs = Observable()

dev._obs_queue = [obs]
dev.pre_measure()
res = dev.expval(obs.name, wires=obs.wires, par=obs.parameters)
print(res)

