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
    "qiskit>=0.20",
    "pennylane>=0.11.0",
    "numpy",
    "networkx>=2.2;python_version>'3.5'",
    # Networkx 2.4 is the final version with python 3.5 support.
    "networkx>=2.2,<2.4;python_version=='3.5'",
    "pyscf<=1.7.2"
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
            'qiskit.aer = pennylane_qiskit:AerDevice',
            'qiskit.basicaer = pennylane_qiskit:BasicAerDevice',
            'qiskit.ibmq = pennylane_qiskit:IBMQDevice',
            ],
        'pennylane.io': [
            'qiskit = pennylane_qiskit:load',
            'qasm = pennylane_qiskit:load_qasm',
            'qasm_file = pennylane_qiskit:load_qasm_from_file',
            ],
        },
    'description': 'PennyLane plugin for qiskit-terra',
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
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3 :: Only',
    "Topic :: Scientific/Engineering :: Physics"
]

setup(classifiers=classifiers, **(info))
