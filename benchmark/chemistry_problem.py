import random
import pennylane as qml
from pennylane import numpy as np
from math import pi
from Arguments import Arguments
args = Arguments()
import csv
import pickle
from ChemModel import translator, quantum_net


class ChemistryFunction():
    def __init__(self, dims, lb, ub):
        self.dims = dims
        self.lb = np.asarray(lb)
        self.ub = np.asarray(ub)

    def __call__(self, x):
        pass


class Chemistry(ChemistryFunction):
    def __init__(self, dims=args.n_qubits*2):
        assert dims == args.n_qubits * 2
        ChemistryFunction.__init__(
            self,
            dims,
            0 * np.ones(dims),
            1 * np.ones(dims),
        )


    def __call__(self, x):
        assert len(x) == self.dims
        assert x.ndim == 1
        np.random.seed(42)
        random.seed(42)
        lr = args.qlr

        # randomly initialize a net
        q = [random.randint(0, 1) for _ in range(12)]
        p = [random.randint(0, 11) for _ in range(12)]
        original_net = q + p
        print("original net: ", original_net)
        print("x: ", x)

        energy = []
        randomized_net = [0] * (args.n_qubits * 2)
        n_randnet = 0
        while n_randnet < 2:
            # randomize selected single-qubit gates
            for i in range(0, args.n_qubits):
                if x[i] == 1:
                    randomized_net[i] = random.randint(0, 1)
                else:
                    randomized_net[i] = original_net[i]

            # randomize selected two-qubit gates
            for i in range(args.n_qubits, args.n_qubits * 2):
                if x[i] == 1:
                    randomized_net[i] = random.randint(0, 11)
                else:
                    randomized_net[i] = original_net[i]

            print("randomized net: ", randomized_net)

            with open('data/chemistry_dataset', 'rb') as file:
                dataset = pickle.load(file)
            if str(randomized_net) in dataset:
                result = dataset.get(str(randomized_net))
                print("read result from dataset: ", result)
            else:
                design = translator(randomized_net)

                symbols = ["O", "H"]
                coordinates = np.array([[0.0, 0.0, 0.0], [0.45, -0.1525, -0.8454]])

                # Building the molecular hamiltonian for the trihydrogen cation
                hamiltonian, qubits = qml.qchem.molecular_hamiltonian(symbols, coordinates, charge=1, method='pyscf')

                dev = qml.device("lightning.qubit", wires=args.n_qubits)

                @qml.qnode(dev, diff_method="adjoint")
                def cost_fn(theta):
                    quantum_net(theta, design)
                    return qml.expval(hamiltonian)

                for i in range(10):
                    q_params = 2 * pi * np.random.rand(design['layer_repe'] * args.n_qubits * 2)
                    opt = qml.GradientDescentOptimizer(stepsize=lr)

                    for n in range(100):
                        q_params, prev_energy = opt.step_and_cost(cost_fn, q_params)
                        # print(f"--- Step: {n}, Energy: {cost_fn(q_params):.8f}")
                    energy.append(cost_fn(q_params))

            n_randnet += 1

        result = abs(np.mean(energy))
        print("average absolute energy: ", result)

        # with open('results.csv', 'a+', newline='') as res:
        #     writer = csv.writer(res)
        #     writer.writerow([original_net, x, result])

        return result


if __name__ == '__main__':
    net = [0, 1, 0, 1, 1, 0, 2, 5, 1, 2, 3, 3]
    # net = [1, 1, 0, 1, 1, 1, 5, 5, 4, 1, 2, 1]
    # design = translator(net)
    # report = chemistry(design)