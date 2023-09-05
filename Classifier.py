import json
from math import log2, ceil
import torch
import numpy as np
import torch.nn as nn
from torch import optim

"""
class LinearModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearModel, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        y = self.fc(x)
        return y
"""
torch.cuda.is_available = lambda : False

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Encoder, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, output_dim)
            )
        
    def forward(self, x):
        y = self.network(x)
        y = torch.sigmoid(y)
        return y


class Classifier:
    def __init__(self, samples, input_dim, node_id):
        assert type(samples) == type({})
        assert input_dim     >= 1

        self.samples          = samples
        self.input_dim        = input_dim
        self.training_counter = 0
        self.node_layer       = ceil(log2(node_id + 2) - 1)
        self.hidden_dims      = [5, 6, 7, 8, 9, 10]
        self.model            = Encoder(input_dim, self.hidden_dims[self.node_layer], input_dim+1)
        if torch.cuda.is_available():
            self.model.cuda()
        self.loss_fn          = nn.MSELoss()
        self.l_rate           = 0.001
        self.optimizer        = optim.Adam(self.model.parameters(), lr=self.l_rate, betas=(0.9, 0.999), eps=1e-08)
        self.epochs           = []
        self.boundary         = -1
        self.nets             = None
        self.maeinv           = None
        self.labels           = None

    def get_label(self, energy):
        label = torch.zeros_like(energy)
        for i in range(energy.shape[0]): 
            label[i] = energy[i] > energy.mean()
        return label
    
    def update_samples(self, latest_samples):
        assert type(latest_samples) == type(self.samples)
        sampled_nets = []        
        nets_maeinv  = []
        for k, v in latest_samples.items():
            net = json.loads(k)
            sampled_nets.append(net)            
            nets_maeinv.append(v)
        self.nets = torch.from_numpy(np.asarray(sampled_nets, dtype=np.float32).reshape(-1, self.input_dim))
        self.maeinv = torch.from_numpy(np.asarray(nets_maeinv, dtype=np.float32).reshape(-1, 1))
        self.labels = self.get_label(self.maeinv)
        self.samples = latest_samples
        if torch.cuda.is_available():
            self.nets = self.nets.cuda()
            self.labels = self.labels.cuda()


    def train(self):
        if self.training_counter == 0:
            self.epochs = 200
        else:
            self.epochs = 100
        self.training_counter += 1
        # in a rare case, one branch has no networks
        if len(self.nets) == 0:
            return
        for epoch in range(self.epochs):
            nets = self.nets
            maeinv = self.maeinv
            # clear grads
            self.optimizer.zero_grad()
            # forward to get predicted values
            outputs = self.model(nets)
            loss_s = self.loss_fn(outputs[:, :12], nets)
            loss_t = self.loss_fn(outputs[:, -1].reshape(-1, 1), maeinv)
            loss = loss_s + loss_t
            loss.backward()  # back props
            nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optimizer.step()  # update the parameters


    def predict(self, remaining):
        assert type(remaining) == type({})
        remaining_archs = []
        for k, v in remaining.items():
            net = json.loads(k)
            remaining_archs.append(net)
        remaining_archs = torch.from_numpy(np.asarray(remaining_archs, dtype=np.float32).reshape(-1, self.input_dim))
        if torch.cuda.is_available():
            remaining_archs = remaining_archs.cuda()
        outputs = self.model(remaining_archs)[:, -1].reshape(-1, 1)
        if torch.cuda.is_available():
            remaining_archs = remaining_archs.cpu()
            outputs         = outputs.cpu()
        result = {}
        for k in range(0, len(remaining_archs)):
            arch = remaining_archs[k].detach().numpy().astype(np.int32)
            arch_str = json.dumps(arch.tolist())
            result[arch_str] = outputs[k].detach().numpy().tolist()[0]
        assert len(result) == len(remaining)
        return result


    def split_predictions(self, remaining):
        assert type(remaining) == type({})
        samples_badness = {}
        samples_goodies = {}
        if len(remaining) == 0:
            return samples_goodies, samples_badness
        predictions = self.predict(remaining)  # arch_str -> pred_test_mae
        avg_maeinv  = self.sample_mean()
        self.boundary = avg_maeinv
        for k, v in predictions.items():
            if v < avg_maeinv:
                samples_badness[k] = (0.0, v)
            else:
                samples_goodies[k] = (0.0, v)
        assert len(samples_badness) + len(samples_goodies) == len(remaining)
        return samples_goodies, samples_badness

    """
    def predict_mean(self):
        if len(self.nets) == 0:
            return 0
        # can we use the actual maeinv?
        outputs = self.model(self.nets)
        pred_np = None
        if torch.cuda.is_available():
            pred_np = outputs.detach().cpu().numpy()
        else:
            pred_np = outputs.detach().numpy()
        return np.mean(pred_np)
    """

    def sample_mean(self):
        if len(self.nets) == 0:
            return 0
        outputs = self.maeinv
        true_np = None
        if torch.cuda.is_available():
            true_np = outputs.cpu().numpy()
        else:
            true_np = outputs.numpy()
        return np.mean(true_np)


    def split_data(self):
        samples_badness = {}
        samples_goodies = {}
        if len(self.nets) == 0:
            return samples_goodies, samples_badness
        self.train()
        outputs = self.model(self.nets)[:, -1].reshape(-1, 1)
        if torch.cuda.is_available():
            self.nets = self.nets.cpu()
            outputs   = outputs.cpu()
        predictions = {}
        for k in range(0, len(self.nets)):
            arch = self.nets[k].detach().numpy().astype(np.int32)
            arch_str = json.dumps(arch.tolist())
            predictions[arch_str] = outputs[k].detach().numpy().tolist()[0]  # arch_str -> pred_test_mae
        assert len(predictions) == len(self.nets)
        avg_maeinv = self.sample_mean()
        self.boundary = avg_maeinv
        for k, v in predictions.items():
            if v < avg_maeinv:
                samples_badness[k] = self.samples[k]  # (val_loss, test_mae)
            else:
                samples_goodies[k] = self.samples[k]  # (val_loss, test_mae)
        assert len(samples_badness) + len(samples_goodies) == len(self.samples)
        return samples_goodies, samples_badness
