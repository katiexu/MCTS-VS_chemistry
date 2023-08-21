import numpy as np
# from baseline.vanilia_bo import get_gpr_model, optimize_acqf
from MCTSVS.Node import Node
from uipt_variable_strategy import UiptRandomStrategy, UiptBestKStrategy, UiptAverageBestKStrategy, UiptCopyStrategy, UiptMixStrategy
from utils import bernoulli, latin_hypercube, from_unit_cube, feature_complementary, ndarray2str, feature_dedup


# from inner_optimizer import Turbo1_VS_Component
# from inner_optimizer import run_saasbo_one_epoch


class MCTS:
    def __init__(self, func, dims, lb, ub, feature_batch_size=2,
                 sample_batch_size=3, Cp=5, min_num_variables=3,
                 select_right_threshold=5, k=20, split_type='mean',
                 ipt_solver='bo', uipt_solver='bestk', turbo_max_evals=50):
        # user defined parameters
        assert len(lb) == dims and len(ub) == dims
        self.func = func
        self.dims = dims
        self.lb = lb
        self.ub = ub
        self.feature_batch_size = feature_batch_size  # sample feature_batch_size features and feature_batch_size complementary features
        self.sample_batch_size = sample_batch_size  # sample sample_batch_size datas for each feature
        self.Cp = Cp
        self.min_num_variables = min_num_variables
        self.select_right_threshold = select_right_threshold
        self.turbo_max_evals = turbo_max_evals

        self.split_type = split_type
        self.ipt_solver = ipt_solver
        uipt_solver_dict = {
            'random': UiptRandomStrategy(self.dims),
            'bestk': UiptBestKStrategy(self.dims, k=k),
            'average_bestk': UiptAverageBestKStrategy(self.dims, k=k),
            'copy': UiptCopyStrategy(self.dims),
            'mix': UiptMixStrategy(self.dims),
        }
        self.uipt_solver = uipt_solver_dict[uipt_solver]

        # parameters to store datas
        self.features = []
        self.samples = []
        self.feature2sample_map = dict()
        self.curt_best_sample = None
        self.curt_best_value = float('-inf')
        self.best_value_trace = []
        self.value_trace = []
        self.sample_counter = 0

        # build the tree
        self.nodes = []
        root = Node(parent=None, dims=self.dims, active_dims_idx=list(range(self.dims)),
                    min_num_variables=self.min_num_variables, reset_id=True)
        self.nodes.append(root)
        self.ROOT = root
        self.CURT = self.ROOT
        self.num_select_right = float('inf')  # run 'dynamic_treeify' when iteration = 1

        self.init_train()

        self.selected_variables = []

    def init_train(self):
        assert len(self.features) == 0 and len(self.samples) == 0
        # init features
        features = bernoulli(self.feature_batch_size, self.dims, p=0.5)
        comp_features = [feature_complementary(features[idx]) for idx in range(self.feature_batch_size)]
        self.features.extend(feature_dedup(features + comp_features))

        # collect similar sample for each feature
        for feature in self.features:
            # points = latin_hypercube(self.sample_batch_size, self.dims)
            # points = from_unit_cube(points, self.lb, self.ub)
            points = bernoulli(self.sample_batch_size, self.dims, p=0.5)
            for i in range(self.sample_batch_size):
                y = self.func(points[i])
                self.samples.append((points[i], y))
                self.update_feature2sample_map(feature, points[i], y)

        assert len(self.samples) == len(self.features) * self.sample_batch_size

        # update current best sample information
        self.sample_counter += len(self.samples)
        X_sample, Y_sample = zip(*self.samples)
        best_sample_idx = np.argmax(Y_sample)
        self.curt_best_sample, self.curt_best_value = self.samples[best_sample_idx]
        self.best_value_trace.append((self.sample_counter, self.curt_best_value))

        # init
        self.uipt_solver.init_strategy(X_sample, Y_sample)

        # print mcts information
        print('=' * 10)
        print('feature_batch_size: {}'.format(self.feature_batch_size))
        print('sample_batch_size: {}'.format(self.sample_batch_size))
        print('collect {} samples for initializing MCTS'.format(len(self.samples)))
        print('collect {} features for initializing MCTS'.format(len(self.features)))
        print('dims: {}'.format(self.dims))
        print('min_num_variables: {}'.format(self.min_num_variables))
        print('=' * 10)


    def update_feature2sample_map(self, feature, sample, y):
        feature_str = ndarray2str(feature)
        if self.feature2sample_map.get(feature_str, None) is None:
            self.feature2sample_map[feature_str] = [ (sample, y) ]
        else:
            self.feature2sample_map[feature_str].append( (sample, y) )