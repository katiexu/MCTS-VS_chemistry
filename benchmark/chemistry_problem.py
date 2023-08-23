import random
import pennylane as qml
from pennylane import numpy as np
from math import pi
from benchmark.Arguments import Arguments
args = Arguments()
import csv


def translator(net):
    assert type(net) == type([])
    updated_design = {}

    # r = net[0]
    q = net[0:6]
    # c = net[8:15]
    p = net[6:12]

    # num of layer repetitions
    layer_repe = [1, 5, 7]
    updated_design['layer_repe'] = layer_repe[1]

    # categories of single-qubit parametric gates
    for i in range(args.n_qubits):
        if q[i] == 0:
            category = 'Rx'
        else:
            category = 'Ry'
        updated_design['rot' + str(i)] = category

    # categories and positions of entangled gates
    for j in range(args.n_qubits):
        # if c[j] == 0:
        #     category = 'IsingXX'
        # else:
        #     category = 'IsingZZ'
        updated_design['enta' + str(j)] = ([j, p[j]])

    updated_design['total_gates'] = len(q) + len(p)
    return updated_design


def quantum_net(q_params, design):
    current_design = design
    q_weights = q_params.reshape(current_design['layer_repe'], args.n_qubits, 2)
    # q_weights = q_params.reshape(args.n_qubits, 2)
    for layer in range(current_design['layer_repe']):
        for j in range(args.n_qubits):
            if current_design['rot' + str(j)] == 'Rx':
                qml.RX(q_weights[layer][j][0], wires=j)
            else:
                qml.RY(q_weights[layer][j][0], wires=j)

            qml.IsingZZ(q_weights[layer][j][1], wires=current_design['enta' + str(j)])


class ChemistryFunction():
    def __init__(self, dims, lb, ub):
        self.dims = dims
        # self.negate = negate
        self.lb = np.asarray(lb)
        self.ub = np.asarray(ub)
        # self.opt_val = opt_val
        # self.opt_point = np.array(opt_point)

    def __call__(self, x):
        pass


class Chemistry(ChemistryFunction):
    def __init__(self, dims=12):
        assert dims == 12
        ChemistryFunction.__init__(
            self,
            dims,
            # negate,
            0 * np.ones(dims),
            1 * np.ones(dims),
            # -3.32237,
            # np.array([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573])
        )


    def __call__(self, x):
        assert len(x) == self.dims
        assert x.ndim == 1
        # assert np.all(x <= self.ub) and np.all(x >= self.lb)
        np.random.seed(42)
        random.seed(42)

        # randomly initialize a net
        q = [random.randint(0, 1) for _ in range(6)]
        p = [random.choice([1, 2, 3, 4, 5]),
             random.choice([0, 2, 3, 4, 5]),
             random.choice([0, 1, 3, 4, 5]),
             random.choice([0, 1, 2, 4, 5]),
             random.choice([0, 1, 2, 3, 5]),
             random.choice([0, 1, 2, 3, 4])]
        original_net = q + p
        print("original net: ", original_net)
        print("x: ", x)

        energy = []
        randomized_net = [0] * (args.n_qubits * 2)
        n_randnet = 0
        while n_randnet < 10:
            # randomize selected single-qubit gates
            for i in range(0, args.n_qubits):
                if x[i] == 1:
                    randomized_net[i] = random.choice([0, 1])
                else:
                    randomized_net[i] = original_net[i]

            # randomize selected two-qubit gates
            for i in range(args.n_qubits, args.n_qubits * 2):
                if x[i] == 1:
                    randomized_net[i] = random.choice([0, 1, 2, 3, 4, 5])
                    while randomized_net[i] == i - args.n_qubits:
                        randomized_net[i] = random.choice([0, 1, 2, 3, 4, 5])
                else:
                    randomized_net[i] = original_net[i]

            print("randomized net: ", randomized_net)

            design = translator(randomized_net)

            symbols = ["H", "H", "H"]
            coordinates = np.array([0.028, 0.054, 0.0, 0.986, 1.610, 0.0, 1.855, 0.002, 0.0])

            # Building the molecular hamiltonian for the trihydrogen cation
            hamiltonian, qubits = qml.qchem.molecular_hamiltonian(symbols, coordinates, charge=1)

            dev = qml.device("lightning.qubit", wires=args.n_qubits)

            @qml.qnode(dev, diff_method="adjoint")
            def cost_fn(theta):
                quantum_net(theta, design)
                # print(qml.draw(quantum_net)(q_params, design))
                return qml.expval(hamiltonian)

            for i in range(5):
                q_params = 2 * pi * np.random.rand(design['layer_repe'] * args.n_qubits * 2)
                opt = qml.GradientDescentOptimizer(stepsize=0.4)

                for n in range(50):
                    q_params, prev_energy = opt.step_and_cost(cost_fn, q_params)
                    # print(f"--- Step: {n}, Energy: {cost_fn(q_params):.8f}")
                energy.append(cost_fn(q_params))

            n_randnet += 1

        result = np.mean(energy)
        result = abs(result)
        print("average absolute energy: ", result)

        with open('results.csv', 'a+', newline='') as res:
            writer = csv.writer(res)
            writer.writerow([original_net, x, result])

        return result


if __name__ == '__main__':
    net = [0, 1, 0, 1, 1, 0, 2, 5, 1, 2, 3, 3]
    # net = [1, 1, 0, 1, 1, 1, 5, 5, 4, 1, 2, 1]
    # design = translator(net)
    # report = chemistry(design)
