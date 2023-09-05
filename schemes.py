import pickle
import copy
from pennylane import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from sklearn.metrics import accuracy_score, f1_score
from ChemModel import translator, quantum_net
from Arguments import Arguments


def get_param_num(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total:', total_num, 'trainable:', trainable_num)


def display(metrics):
    print("\nTest mae: {}".format(metrics['mae']))
    print("Test correlation: {}".format(metrics['corr']))
    print("Test multi-class accuracy: {}".format(metrics['multi_acc']))
    print("Test binary accuracy: {}".format(metrics['bi_acc']))
    print("Test f1 score: {}".format(metrics['f1']))


def train(model, data_loader, optimizer, criterion, args):
    model.train()
    for data_a, data_v, data_t, target in data_loader:
        data_a, data_v, data_t = data_a.to(args.device), data_v.to(args.device), data_t.to(args.device)
        target = target.to(args.device)
        optimizer.zero_grad()
        output = model(data_a, data_v, data_t)
        loss = criterion(output, target)
        # loss = output[1]
        loss.backward()
        optimizer.step()


def test(model, data_loader, criterion, args):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data_a, data_v, data_t, target in data_loader:
            data_a, data_v, data_t = data_a.to(args.device), data_v.to(args.device), data_t.to(args.device)
            target = target.to(args.device)
            output = model(data_a, data_v, data_t)
            instant_loss = criterion(output, target).item()
            total_loss += instant_loss
    total_loss /= len(data_loader.dataset)
    return total_loss


def evaluate(model, data_loader, args):
    model.eval()
    metrics = {}
    with torch.no_grad():
        data_a, data_v, data_t, target = next(iter(data_loader))
        data_a, data_v, data_t = data_a.to(args.device), data_v.to(args.device), data_t.to(args.device)
        output = model(data_a, data_v, data_t)
    output = output.cpu().numpy()
    target = target.numpy()
    metrics['mae'] = np.mean(np.absolute(output - target)).item()
    metrics['corr'] = np.corrcoef(output, target)[0][1].item()
    metrics['multi_acc'] = round(sum(np.round(output) == np.round(target)) / float(len(target)), 5).item()
    true_label = (target >= 0)
    pred_label = (output >= 0)
    metrics['bi_acc'] = accuracy_score(true_label, pred_label).item()
    metrics['f1'] = f1_score(true_label, pred_label, average='weighted').item()
    return metrics


def Scheme(design):
    args = Arguments()
    if torch.cuda.is_available() and args.device == 'cuda':
        print("using cuda device")
    else:
        print("using cpu device")
    train_loader, val_loader, test_loader = MOSIDataLoaders(args)
    model = QNet(args, design).to(args.device)
    criterion = nn.L1Loss(reduction='sum')
    optimizer = optim.Adam([
        {'params': model.ClassicalLayer_a.parameters()},
        {'params': model.ClassicalLayer_v.parameters()},
        {'params': model.ClassicalLayer_t.parameters()},
        {'params': model.ProjLayer_a.parameters()},
        {'params': model.ProjLayer_v.parameters()},
        {'params': model.ProjLayer_t.parameters()},
        {'params': model.QuantumLayer.parameters(), 'lr': args.qlr},
        {'params': model.Regressor.parameters()}
        ], lr=args.clr)
    train_loss_list, val_loss_list = [], []
    best_val_loss = 10000

    start = time.time()
    for epoch in range(args.epochs):
        train(model, train_loader, optimizer, criterion, args)
        train_loss = test(model, train_loader, criterion, args)
        train_loss_list.append(train_loss)
        val_loss = test(model, val_loader, criterion, args)
        val_loss_list.append(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(epoch, train_loss, val_loss, 'saving model')
            best_model = copy.deepcopy(model)
        else:
            print(epoch, train_loss, val_loss)
    end = time.time()
    print("Running time: %s seconds" % (end - start))
    
    metrics = evaluate(best_model, test_loader, args)
    display(metrics)
    report = {'train_loss_list': train_loss_list, 'val_loss_list': val_loss_list,
              'best_val_loss': best_val_loss, 'metrics': metrics}
    return best_model, report

def chemistry(design):
    import pennylane as qml
    from math import pi

    np.random.seed(42)
    args = Arguments()
    symbols = ["H", "H", "H"]
    coordinates = np.array([0.028, 0.054, 0.0, 0.986, 1.610, 0.0, 1.855, 0.002, 0.0])

# Building the molecular hamiltonian for the trihydrogen cation
    hamiltonian, qubits = qml.qchem.molecular_hamiltonian(symbols, coordinates, charge=1)

    dev = qml.device("lightning.qubit", wires=args.n_qubits)
    @qml.qnode(dev, diff_method="adjoint")

    def cost_fn(theta):
        quantum_net(theta, design)
        return qml.expval(hamiltonian)
   
    energy = []
    for i in range(10):
        q_params = 2 * pi * np.random.rand(design['layer_repe'] * args.n_qubits * 2)
        opt = qml.GradientDescentOptimizer(stepsize=0.4)

        for n in range(50):
            q_params, prev_energy = opt.step_and_cost(cost_fn, q_params)
            # print(f"--- Step: {n}, Energy: {cost_fn(q_params):.8f}")
        energy.append(cost_fn(q_params))
    
    metrics = np.mean(energy)
    report = {'energy': metrics}
    print(metrics)
    return report


if __name__ == '__main__':
    with open('data/chemistry_dataset', 'rb') as json_data:
        data = pickle.load(json_data)
    net = '[1, 1, 0, 0, 1, 1, 1, 4, 3, 4, 3, 1]'
    design = translator(net)
    report = chemistry(design)
