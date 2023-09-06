import pandas as pd
import os
import time


class Tracker:
    def __init__(self):
        self.counter = 0
        self.best_value_trace = []
        self.curt_best = float('-inf')
        self.start_time = time.time()

    def track(self, result):
        self.counter += 1
        if result > self.curt_best:
            self.curt_best = result
        self.best_value_trace.append((
            self.counter,
            self.curt_best,
            time.time() - self.start_time
        ))



def save_results(root_dir, algo, func, seed, df_data):
    os.makedirs(root_dir, exist_ok=True)
    save_dir = os.path.join(root_dir, func)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, '%s-%d.csv' % (algo, seed))
    df_data.to_csv(save_path)
    print('save %s result into: %s' % (algo, save_path))