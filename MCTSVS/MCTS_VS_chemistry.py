import numpy as np
from baseline.vanilia_bo import get_gpr_model, optimize_acqf
from MCTSVS.VSNode import VSNode
from uipt_variable_strategy import UiptRandomStrategy, UiptBestKStrategy, UiptAverageBestKStrategy, UiptCopyStrategy, UiptMixStrategy
from utils import bernoulli, feature_complementary, ndarray2str, feature_dedup
import csv
from Arguments import Arguments
args = Arguments()


class VSMCTS:
    def __init__(self, func, dims, lb, ub, feature_batch_size=2,
                 sample_batch_size=3, Cp=5, min_num_variables=3,
                 select_right_threshold=5, k=20, split_type='mean',
                 ipt_solver='bo', uipt_solver='bestk', turbo_max_evals=50):
        # user defined parameters
        assert len(lb) == dims and len(ub) == dims
        self.func = func
        self.dims = args.n_qubits * 2
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
        root = VSNode(parent=None, dims=self.dims, active_dims_idx=list(range(self.dims)),
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

    def collect_samples(self, feature):
        train_x, train_y = zip(*self.samples)
        np_train_x = np.vstack(train_x)
        np_train_y = np.array(train_y)
        feature_idx = [idx for idx, i in enumerate(feature) if i == 1]
        ipt_x = np_train_x[:, feature_idx]
        ipt_lb = np.array([i for idx, i in enumerate(self.lb) if idx in feature_idx])
        ipt_ub = np.array([i for idx, i in enumerate(self.ub) if idx in feature_idx])
        # print('select feature: {}'.format(feature))

        if self.ipt_solver == 'bo':
            # get important variables
            gpr = get_gpr_model()
            gpr.fit(ipt_x, train_y)
            new_ipt_x, _ = optimize_acqf(len(feature_idx), gpr, ipt_x, train_y, self.sample_batch_size, ipt_lb, ipt_ub)
            # get unimportant variables
            X_sample, Y_sample = [], []
            for i in range(len(new_ipt_x)):
                fixed_variables = {idx: float(v) for idx, v in zip(feature_idx, new_ipt_x[i])}
                new_x = self.uipt_solver.get_full_variable(
                    fixed_variables,
                    self.lb,
                    self.ub
                )
                # turn new_x (decimals) into 0 or 1
                for j in range(0, len(new_x)):
                    if new_x[j] < 0.5:
                        new_x[j] = 0
                    else:
                        new_x[j] = 1

                value = self.func(new_x)
                self.uipt_solver.update(new_x, value)
                self.samples.append((new_x, value))
                self.update_feature2sample_map(feature, new_x, value)

                X_sample.append(new_x)
                Y_sample.append(value)
        else:
            assert 0

        for idx, y in enumerate(Y_sample):
            self.sample_counter += 1
            if y > self.curt_best_value:
                self.curt_best_sample = X_sample[idx]
                self.curt_best_value = y
                self.best_value_trace.append((self.sample_counter, self.curt_best_value))

            self.value_trace.append((self.sample_counter, self.curt_best_value))

        return X_sample, Y_sample

    def populate_training_data(self):
        self.nodes.clear()
        self.ROOT = VSNode(parent=None, dims=self.dims, active_dims_idx=list(range(self.dims)),
                         min_num_variables=self.min_num_variables, reset_id=True)
        self.nodes.append(self.ROOT)
        self.CURT = self.ROOT
        self.ROOT.init_bag(self.features, self.samples, self.feature2sample_map)
        assert VSNode.obj_counter == 1

    def get_split_idx(self):
        split_by_samples = np.argwhere(self.get_leaf_status() == True).reshape(-1)
        return split_by_samples

    def get_leaf_status(self):
        status = []
        for node in self.nodes:
            Y_sample = [y for _, y in node.samples]
            Y_sample = np.array(Y_sample)
            if node.is_leaf() and Y_sample.std() > 0.1:
                status.append(True)
            else:
                status.append(False)
        return np.array(status)

    def is_splitable(self):
        status = self.get_leaf_status()
        if True in status:
            return True
        else:
            return False

    def dynamic_treeify(self):
        # print('rebuild the tree')
        self.num_select_right = 0
        self.populate_training_data()

    def greedy_select(self):
        pass

    def select(self, verbose=True):
        self.CURT = self.ROOT
        curt_node = self.ROOT
        path = []

        while curt_node.is_leaf() == False:
            UCT = []
            for i in curt_node.kids:
                UCT.append(i.get_uct(self.Cp))
            choice = np.random.choice(np.argwhere(UCT == np.amax(UCT)).reshape(-1), 1)[0]
            path.append((curt_node, choice))
            curt_node = curt_node.kids[choice]
            self.num_select_right += choice  # 0: left, 1: right
            if verbose:
                print('=>', curt_node.get_name(), end=' ')
        if verbose:
            print('')

        return curt_node, path

    def backpropogate(self, leaf, feature, X_sample, Y_sample):
        samples = [(x, y) for x, y in zip(X_sample, Y_sample)]
        curt_node = leaf
        while curt_node is not None:
            curt_node.n += 1
            curt_node.update_bag(feature, samples)
            curt_node = curt_node.parent


    def VSsearch(self, max_samples, verbose=True):
        idx = 0
        max_num_of_iterations = 0
        while True:
            if verbose:
                print('')
                print('=' * 10)
                print('iteration: {}'.format(idx))
                print('=' * 10)

            if self.num_select_right >= self.select_right_threshold:
                self.dynamic_treeify()
                starting_idx = idx
                # print('rebuild')
            leaf, path = self.select(verbose)
            self.selected_variables.append((self.sample_counter, leaf.active_dims_idx))

            for i in range(1):
                new_feature, new_comp_features = leaf.sample_features(self.feature_batch_size)
                all_features = feature_dedup(new_feature + new_comp_features)

                for feature in all_features:
                    if ndarray2str(feature) not in self.feature2sample_map.keys():
                        self.features.append(feature)
                    X_sample, Y_sample = self.collect_samples(feature)
                    self.backpropogate(leaf, feature, X_sample, Y_sample)

            left_kid, right_kid = leaf.split(self.split_type)
            if left_kid is not None and right_kid is not None:
                self.nodes.append(left_kid)
                self.nodes.append(right_kid)

            if verbose:
                self.print_tree()
                print('axis_score argsort:', np.argsort(self.ROOT.axis_score)[:: -1])
                print('total samples: {}'.format(len(self.samples)))
                print('current best f(x): {}'.format(self.curt_best_value))
                # print('current best x: {}'.format(self.curt_best_sample))
                node = self.ROOT
                while len(node.kids) > 0:
                    node = node.kids[0]
                print(node.active_dims_idx)
                print(node.active_axis_score)

            idx += 1

            # save best variables of the best tree
            if self.num_select_right >= self.select_right_threshold:
                num_of_iterations = idx - starting_idx
                if num_of_iterations >= max_num_of_iterations:
                    max_num_of_iterations = num_of_iterations
                    best_variables = node.active_dims_idx
                with open('best_variables.csv', 'a+', newline='') as res:
                    writer = csv.writer(res)
                    writer.writerow([starting_idx, num_of_iterations, node.active_dims_idx, best_variables])

            if self.sample_counter >= max_samples:
                break

        return best_variables


    def update_feature2sample_map(self, feature, sample, y):
        feature_str = ndarray2str(feature)
        if self.feature2sample_map.get(feature_str, None) is None:
            self.feature2sample_map[feature_str] = [ (sample, y) ]
        else:
            self.feature2sample_map[feature_str].append( (sample, y) )


    def print_tree(self):
        print('='*10)
        for node in self.nodes:
            print('-'*10)
            print(node)
            print('-'*10)
        print('='*10)
