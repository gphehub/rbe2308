import copy
import numpy as np
from qiskit import(
  QuantumCircuit,
  execute,
  Aer)
from qiskit.visualization import plot_histogram
import csv
from sklearn.preprocessing import normalize
import math
from numpy import genfromtxt
from qiskit.quantum_info import Statevector



# Using Aer's qasm_simulator
simulator = Aer.get_backend('qasm_simulator')



def mapping(param, data):
    # Amplitude encoding:
    num_param = len(param)
    num_bits = num_qubits

    def feature_map(data):
        classical_state = normalize([data])
        desired_state = classical_state[0].tolist()
        for i in range(2 ** num_qubits - 784): desired_state.append(0.0)
        return desired_state



    # The RealAmplitudes ansatz:
    def real_amp(param):
        varform = QuantumCircuit(num_qubits, 0)
        for i in range(num_qubits): varform.ry(param[i], i)
        for j in range(repetitions):
            for m in range(num_qubits-1):
                for n in range(m+1, num_qubits): varform.cx(m, n) # full entanglement
                #varform.cx(m, m+1) # circular and linear entanglement
            #varform.cx(num_qubits-1, 0) # circular entanglement
            #varform.barrier(range(num_qubits))
            for i in range(num_qubits): varform.ry(param[(j + 1) * num_qubits + i], i)
        return varform



    # Building the quantum circuit:
    circuit_qc = QuantumCircuit(num_qubits, num_bits)
    circuit_qc.initialize(feature_map(data), range(num_qubits))
    circuit_qc = circuit_qc.compose(real_amp(param))



    #Computing:
    circuit_qc.save_statevector()
    result = simulator.run(circuit_qc).result()
    outputstate = result.get_statevector(circuit_qc)
    probs = Statevector(outputstate).probabilities()

    return probs



def vectorized_result(j, num_qubits):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((num_qubits, 1))
    e[j] = 1.0
    return e



# MAIN PROGRAM:
import pickle
import gzip
import time
import array



# Global adjustable parameters:
num_qubits = 10
repetitions = 1
dim = 784 # Dimension of the input data set. Must not exceed (2 ** num_qubits). For MNIST it is 784.



num_param = num_qubits * (repetitions + 1)



with open('data/results_of_old_method.csv','w', newline='') as resultsave:
    writer=csv.writer(resultsave)
    writer.writerow(("Number of computed data", "Time spent", "Cost function"))



# Loading the real input data:
f = gzip.open('data/mnist.pkl.gz', 'rb')
tr_d, va_d, te_d = pickle.load(f,encoding='bytes')
f.close()



# Loading the circuit parameters:
param0 = genfromtxt('data/parameters.csv', delimiter=',', skip_header = 0)
param_unshifted = param0[0 : num_param]



time0 = time.time()

localtime = time.localtime(time0)
strtime = time.strftime("%Y-%m-%d %H:%M:%S", localtime)
print('Program starts at', strtime)

time_quant = 0 # Time spent on the quantum circuit



cost_unnormalized = 0

# Prepare the keys for which the measurement result of the nth qubit is 1.
keys = []
for n in range(num_qubits):
    key = []
    for m in range(2 ** num_qubits):
        x = int(m / (2 ** n))
        if x % 2 == 1: key.append(m)
    keys.append(key)

# Computing the probability of the output states for each input data:
range_start = array.array('i', [0, 1, 100, 2000])
range_end = array.array('i', [1, 100, 2000, 50000])
batch_size = array.array('i', [1, 99, 100, 2000])

for i in range(4):
    for j in range(int((range_end[i] - range_start[i]) / batch_size[i])):
        for k in range(range_start[i] + j * batch_size[i], range_start[i] + (j + 1) * batch_size[i]):
            time_q0 = time.time()
            deduced_probs = mapping(param_unshifted, tr_d[0][k]) #mod_old_method_v1
            time_q1 = time.time()
            time_quant += time_q1 - time_q0

            # Cost function format v9 (following P9/Eq.(5) of ml94):
            probabilities = 0
            for m in keys[tr_d[1][k]]:
                probabilities += deduced_probs[m]
            cost_unnormalized -= np.log(probabilities) # Note that it is "-=".

        cost = cost_unnormalized / ((k + 1) * num_qubits)
        time_spent = time.time() - time0

        #Showing the results:
        print(k + 1, 'data computed. Time spent:', round(time_spent, 2), 'seconds. Cost function =', round(cost, 6))
        with open('data/results_of_old_method.csv','a', newline='') as resultsave:
            writer=csv.writer(resultsave)
            writer.writerow((k + 1, time_spent, time_quant, cost))

print('\nCalculation completed.')
