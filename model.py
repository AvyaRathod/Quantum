import json
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
from qiskit import QuantumCircuit
from qiskit.algorithms.optimizers import COBYLA
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit.utils import algorithm_globals
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import EstimatorQNN
from sklearn.model_selection import train_test_split
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

algorithm_globals.random_seed = 12345

# We now define a two qubit unitary as defined in [3]
def conv_circuit(params):
    target = QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 1)
    target.cx(1, 0)
    target.rz(np.pi / 2, 0)
    return target

def conv_layer(num_qubits, param_prefix):
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc = qc.compose(conv_circuit(params[param_index : (param_index + 3)]), [q1, q2])
        qc.barrier()
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
        qc = qc.compose(conv_circuit(params[param_index : (param_index + 3)]), [q1, q2])
        qc.barrier()
        param_index += 3

    qc_inst = qc.to_instruction()

    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, qubits)
    return qc

def pool_circuit(params):
    target = QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 1)

    return target

def pool_layer(sources, sinks, param_prefix):
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for source, sink in zip(sources, sinks):
        qc = qc.compose(pool_circuit(params[param_index : (param_index + 3)]), [source, sink])
        qc.barrier()
        param_index += 3

    qc_inst = qc.to_instruction()

    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, range(num_qubits))
    return qc

def callback_graph(weights, obj_func_eval):
    clear_output(wait=True)
    objective_func_vals.append(obj_func_eval)
    plt.title("Objective function value against iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Objective function value")
    plt.plot(range(len(objective_func_vals)), objective_func_vals)
    plt.show()

# Feature map for 64 qubits suitable for the image size
feature_map = ZFeatureMap(64)

# Three alternating conv_layer and pool_layer functions
conv_layer_1 = conv_layer(64, "conv_params_1")
pool_layer_1 = pool_layer(range(0, 32), range(32, 64), "pool_params_1")

conv_layer_2 = conv_layer(64, "conv_params_2")
pool_layer_2 = pool_layer(range(0, 32), range(32, 64), "pool_params_2")

conv_layer_3 = conv_layer(64, "conv_params_3")
pool_layer_3 = pool_layer(range(0, 32), range(32, 64), "pool_params_3")

# Combine the layers to form the complete quantum circuit
ansatz = QuantumCircuit(64)
ansatz.compose(feature_map_64, range(64), inplace=True)
ansatz.compose(conv_layer_1, range(64), inplace=True)
ansatz.compose(pool_layer_1, range(64), inplace=True)
ansatz.compose(conv_layer_2, range(64), inplace=True)
ansatz.compose(pool_layer_2, range(64), inplace=True)
ansatz.compose(conv_layer_3, range(64), inplace=True)
ansatz.compose(pool_layer_3, range(64), inplace=True)

circuit = QuantumCircuit(64)
circuit.compose(feature_map, range(64), inplace=True)
circuit.compose(ansatz, range(64), inplace=True)

observable = [SparsePauliOp.from_list([('Z' + 'I' * i + 'Z' + 'I' * (63 - i), 1)]) for i in range(64)]

qnn = EstimatorQNN(
    circuit=circuit.decompose(),
    observables=observable,
    input_params=feature_map.parameters,
    weight_params=ansatz.parameters,
)

classifier = NeuralNetworkClassifier(
    qnn,
    optimizer=COBYLA(maxiter=200),  # Set max iterations here
    callback=callback_graph,
)