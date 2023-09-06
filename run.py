import torch
import botorch
import numpy as np
import argparse
import random
from benchmark import get_problem
from MCTSVS.MCTS_VS_chemistry import VSMCTS
import csv
import os


# MCTS-VS parameters
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

random.seed(args.seed)
np.random.seed(args.seed)
botorch.manual_seed(args.seed)
torch.manual_seed(args.seed)

f = get_problem(args.func, args.seed)

if os.path.isfile('best_variables.csv') == False:
    with open('best_variables.csv', 'w+', newline='') as res:
        writer = csv.writer(res)
        writer.writerow(['starting iteration #', 'num of iterations', 'best variables of current tree', 'best variables till now'])

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

best_variables = VSagent.VSsearch(max_samples=args.max_samples, verbose=True)
print("best variables: ", best_variables)