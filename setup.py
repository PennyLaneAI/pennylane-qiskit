# Copyright 2018 Carsten Blank

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python3
from setuptools import setup

with open("pennylane_qiskit/_version.py") as f:
    version = f.readlines()[-1].split()[-1].strip("\"'")

with open("README.rst", "r") as fh:
    long_description = fh.read()

requirements = [
    "qiskit>=0.32",
    "qiskit-aer",
    "qiskit-ibm-runtime<=0.29",
    "qiskit-ibm-provider",
    "pennylane>=0.38",
    "numpy",
    "sympy<1.13",
    "networkx>=2.2",
]

info = {
    'name': 'PennyLane-qiskit',
    'version': version,
    'maintainer': 'Xanadu',
    'maintainer_email': 'software@xanadu.ai',
    'url': 'https://github.com/XanaduAI/pennylane-qiskit',
    'license': 'Apache License 2.0',
    'packages': [
        'pennylane_qiskit'
    ],
    'entry_points': {
        'pennylane.plugins': [
            'qiskit.remote = pennylane_qiskit:RemoteDevice',
            'qiskit.aer = pennylane_qiskit:AerDevice',
            'qiskit.basicaer = pennylane_qiskit:BasicAerDevice',
            'qiskit.basicsim = pennylane_qiskit:BasicSimulatorDevice',
            ],
        'pennylane.io': [
            'qiskit = pennylane_qiskit:load',
            'qiskit_op = pennylane_qiskit:load_pauli_op',
            'qiskit_noise = pennylane_qiskit:load_noise_model',
            'qasm = pennylane_qiskit:load_qasm',
            'qasm_file = pennylane_qiskit:load_qasm_from_file',
            ],
        },
    'description': 'PennyLane plugin for Qiskit',
    'long_description': open('README.rst').read(),
    'provides': ["pennylane_qiskit"],
    'install_requires': requirements,
    # 'extras_require': extra_requirements,
    'command_options': {
        'build_sphinx': {
            'version': ('setup.py', version),
            'release': ('setup.py', version)}}
}

classifiers = [
    "Development Status :: 4 - Beta",
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Natural Language :: English",
    "Operating System :: POSIX",
    "Operating System :: MacOS :: MacOS X",
    "Operating System :: POSIX :: Linux",
    "Operating System :: Microsoft :: Windows",
    "Programming Language :: Python",
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11',
    'Programming Language :: Python :: 3 :: Only',
    "Topic :: Scientific/Engineering :: Physics"
]

setup(classifiers=classifiers, **(info))
