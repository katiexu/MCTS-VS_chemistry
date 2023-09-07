import torch
import botorch
import numpy as np
import argparse
import random
from benchmark import get_problem
from MCTSVS.MCTS_VS_chemistry import VSMCTS
from MCTS_chemisty import MCTS
import csv
import os
import pickle
from Arguments import Arguments
args = Arguments()


# set parameters for MCTS-VS
parser = argparse.ArgumentParser()
parser.add_argument('--func', default='chemistry', type=str,
                    choices=['hartmann6_300', 'hartmann6_500', 'levy10_100', 'levy10_300', 'nasbench', 'nasbench201', 'nasbench1shot1', 'nasbenchtrans', 'nasbenchasr', 'Hopper', 'Walker2d', 'chemistry'])
parser.add_argument('--max_samples', default=600, type=int)
parser.add_argument('--feature_batch_size', default=2, type=int)
parser.add_argument('--sample_batch_size', default=3, type=int)
parser.add_argument('--min_num_variables', default=3, type=int)
parser.add_argument('--select_right_threshold', default=5, type=int)
parser.add_argument('--turbo_max_evals', default=50, type=int)
parser.add_argument('--k', default=20, type=int)
parser.add_argument('--Cp', default=0.1, type=float)
parser.add_argument('--ipt_solver', default='bo', type=str)
parser.add_argument('--uipt_solver', default='bestk', type=str)
parser.add_argument('--root_dir', default='chemistry_logs', type=str)
parser.add_argument('--dir_name', default=None, type=str)
parser.add_argument('--postfix', default=None, type=str)
parser.add_argument('--seed', default=2021, type=int)
args = parser.parse_args()
print(args)

# set random seeds for MCTS-VS
random.seed(args.seed)
np.random.seed(args.seed)
botorch.manual_seed(args.seed)
torch.manual_seed(args.seed)

# get chemistry problem
f = get_problem(args.func, args.seed)

# save best variables from MCTS-VS
if os.path.isfile('best_variables.csv') == False:
    with open('best_variables.csv', 'w+', newline='') as res:
        writer = csv.writer(res)
        writer.writerow(['starting iteration #', 'num of iterations', 'best variables of current tree', 'best variables till now'])

# generate VSagent
VSagent = VSMCTS(
    func=f,
    dims=f.dims,
    lb=f.lb,
    ub=f.ub,
    feature_batch_size=args.feature_batch_size,
    sample_batch_size=args.sample_batch_size,
    Cp=args.Cp,
    min_num_variables=args.min_num_variables,
    select_right_threshold=args.select_right_threshold,
    k=args.k,
    split_type='mean',
    ipt_solver=args.ipt_solver,
    uipt_solver=args.uipt_solver,
    turbo_max_evals=args.turbo_max_evals,
)

# run VSsearch and get best variables
best_variables = VSagent.VSsearch(max_samples=args.max_samples, verbose=True)
print("best variables: ", best_variables)



best_variables = [0, 6, 8]


# set random seeds for MCTS
random.seed(42)
np.random.seed(42)
torch.random.manual_seed(42)


# load existing chemistry tabular dataset
with open('data/chemistry_dataset', 'rb') as file:
    dataset = pickle.load(file)
# sort top 100 best nets from tabular dataset
dataset_sorted = sorted(dataset.items(), key=lambda x: x[1])[:100]


# generate search space for MCTS based on best variables
new_search_space = []
i = 0
N = 100000
while i < N:
    randomized_best_variables = [0] * (args.n_qubits * 2)
    not_best_variables = [1] * (args.n_qubits * 2)
    for j in range(len(best_variables)):
        pos_best_variable = best_variables[j]
        not_best_variables[pos_best_variable] = 0
        if pos_best_variable <= args.n_qubits:
            randomized_best_variables[pos_best_variable] = random.randint(0, 1)
        else:
            randomized_best_variables[pos_best_variable] = random.randint(0, 5)

    for n in range(len(dataset_sorted)):
        arch = eval(dataset_sorted[n][0])
        arch = [x*y for x,y in zip(arch,not_best_variables)]
        arch = [x+y for x,y in zip(arch,randomized_best_variables)]
        if arch in new_search_space:
            continue
        new_search_space.append(arch)
        i += 1
        if i % 1000 == 0:
            print("Collected {} architectures".format(i))

with open('new_search_space', 'wb') as file:
    pickle.dump(new_search_space, file)


# load search space for MCTS
with open('search_space', 'rb') as file:
    search_space = pickle.load(file)
arch_code_len = len(search_space[0])
print("\nthe length of architecture codes:", arch_code_len)
print("total architectures:", len(search_space))


# save results for MCTS search
if os.path.isfile('results.csv') == False:
    with open('results.csv', 'w+', newline='') as res:
        writer = csv.writer(res)
        writer.writerow(['sample_id', 'arch_code', 'sample_node', 'Energy'])


# generate MCTS agent
agent = MCTS(search_space, dataset, 5, arch_code_len)


# run MCTS search
agent.search()