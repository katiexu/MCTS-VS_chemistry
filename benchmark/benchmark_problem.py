import numpy as np
from benchmark.tracker import Tracker
from benchmark.chemistry_problem import Chemistry
from Arguments import Arguments
args = Arguments()


class FunctionBenchmark:
    def __init__(self, func, dims, valid_idx):
        assert func.dims == len(valid_idx)
        self.func = func
        self.dims = dims
        self.valid_idx = valid_idx
        self.lb = func.lb[0] * np.ones(dims)
        self.ub = func.ub[0] * np.ones(dims)

        self.tracker = Tracker()

    def __call__(self, x):
        assert len(x) == self.dims
        result = self.func(x[self.valid_idx])
        self.tracker.track(result)
        return result


def get_problem(func_name, seed=2021):
    if func_name in ['nasbench', 'rover', 'HalfCheetah', 'Walker2d', 'Hopper', 'chemistry']:
        if func_name == 'chemistry':
            return FunctionBenchmark(Chemistry(), args.n_qubits*2, list(range(args.n_qubits*2)))
        else:
            assert 0