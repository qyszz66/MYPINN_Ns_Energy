import math
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from functools import reduce
import operator
import wandb
from torch import optim
from torch.optim import LBFGS
from loss import pde, boundary_upper, boundary_BLR, data_p, reconstruction
from sample import data_sampling, dataT_sampling
# Solve steady 2D N.S. and Energy
# %%
class MLP(nn.Module):
    def __init__(self, in_features, out_features, num_layers, num_neurons, activation=torch.tanh):
        super(MLP, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_layers = num_layers
        self.num_neurons = num_neurons

        self.act_func = activation

        self.layers = nn.ModuleList()

        self.layer_input = nn.Linear(self.in_features, self.num_neurons)

        for ii in range(self.num_layers - 1):
            self.layers.append(nn.Linear(self.num_neurons, self.num_neurons))
        self.layer_output = nn.Linear(self.num_neurons, self.out_features)

    def forward(self, x):
        x_temp = self.act_func(self.layer_input(x))
        for dense in self.layers:
            x_temp = self.act_func(dense(x_temp))
        x_temp = self.layer_output(x_temp)
        return x_temp

    def init_weights(self, init_type="xavier"):
        if init_type == "xavier":
            nn.init.xavier_normal_(self.layer_input.weight)
            nn.init.zeros_(self.layer_input.bias)
            for layer in self.layers:
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
            nn.init.xavier_normal_(self.layer_output.weight)
            nn.init.zeros_(self.layer_output.bias)

    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))
        return c
# %%
def paraing():

    Mu = torch.ones(1 + 1, 1, device=device) # BC and PDE constraints
    mu_max = Mu * 1.
    Lambda = Mu * 1.
    Bar_v  = Lambda * 0.
    return Lambda, Mu, mu_max, Bar_v

class ParaAdapt:
    def __init__(self, zeta, omega, eta, epsilon):
        self.zeta = zeta
        self.omega = omega
        self.eta = eta
        self.epsilon = epsilon

# %%
def My_PINN(X_top_torch, X_BLR_torch, X_f_torch, XY_star_torch, T_star_torch, Gr, Re, Pr,
    para_adapt, Lambda, Mu, mu_max, Bar_v, iteration_num):


    # Training Loop
    last_loss = torch.tensor(torch.inf).to(device)
    start_time = time.time()

    for iter in range(iteration_num):

        def _closure():
            net.eval()
            loss_cont1, loss_mom1, loss_mom2, loss_enery1 = pde(net, X_f_torch, Gr, Re, Pr)
            pde_loss = loss_cont1 + loss_mom1 + loss_mom2 + loss_enery1

            boundary_loss = boundary_upper(net, X_top_torch) + boundary_BLR(net, X_BLR_torch)

            p_loss = data_p(net, X_f_torch)
            dataT_loss = reconstruction(net, XY_star_torch, T_star_torch)
            data_loss =  p_loss + dataT_loss

            objective = pde_loss

            constr = torch.cat((boundary_loss, data_loss), dim=0)

            loss = objective + Lambda.T @ constr + 0.5 * Mu.T @ constr.pow(2)
            return objective, constr, loss

        def closure():
            if torch.is_grad_enabled():
                net.train()
                optimizer.zero_grad()

            objective, constr, loss = _closure()
            if loss.requires_grad:
                loss.backward()
            return loss

        optimizer.step(closure)
        if configuration['Adam']:
            scheduler.step()

        objective, constr, loss = _closure()
        with torch.no_grad():
            Bar_v = para_adapt.zeta * Bar_v + (1 - para_adapt.zeta) * constr.pow(2)
            if loss >= para_adapt.omega * last_loss or iter == iteration_num - 1:
                Lambda += Mu * constr
                new_mu_max = para_adapt.eta / (torch.sqrt(Bar_v) + para_adapt.epsilon)
                mu_max = torch.max(new_mu_max, mu_max)
                Mu = torch.min(torch.max(constr / (torch.sqrt(Bar_v) + para_adapt.epsilon), torch.tensor(1.)) * Mu,
                               mu_max)
        last_loss = loss.detach()

        if (iter + 1) % 50 == 0:
            print('n = %d, loss = %.3e, objective = %.3e, constr_loss = %.3e, %.3e, Lambda = %.3e,  %.3e, Mu = %.3e,  %.3e'  % (
            iter + 1, loss, objective, constr[0], constr[1], Lambda[0], Lambda[1], Mu[0], Mu[1] ))
            torch.save({
                'epoch': iter,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            }, file_path)

            wandb.log({'pde Loss': objective,
                       'Boundary Loss': constr[0],
                       'data Loss': constr[1],
                       'Total Loss ': loss,
                       'LambdaB': Lambda[0],
                       'LambdaT': Lambda[1],
                       'MuB': Mu[0],
                       'MuT': Mu[1],
                       })
        # collect_metrics(constr, objective, loss, Lambda, Mu, q, outer_s, mu_s, lambda_s, constr_s, object_s, q_s,
        #                 loss_s)
        # return model, q, Lambda, Mu, mu_max, Bar_v, optimizer, optim_q

    train_time = time.time() - start_time
    print('Training Time: %d seconds' % (train_time))
    wandb.run.summary['Training Time'] = train_time
    # wandb.run.summary['l2 Error'] = l2_error

# %%
# ############################################################################
# Main code:
device = "cuda" if torch.cuda.is_available() else "cpu"
# Set default dtype to float32
torch.set_default_dtype(torch.float)
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
def torch_tensor_grad(x, device):
    if device == 'cuda':
        x = torch.cuda.FloatTensor(x)
    else:
        x = torch.FloatTensor(x)
    x.requires_grad = True
    return x
def torch_tensor_nograd(x, device):
    if device == 'cuda':
        x = torch.cuda.FloatTensor(x)
    else:
        x = torch.FloatTensor(x)
    x.requires_grad = False
    return x
# %%
configuration = {
    "Type": "MLP",
    "Layers": 10,
    "Neurons": 150,
    "Activation": "Tanh",
    "Flag_pretrain": False,
    "Flag_initialization": True,
    "iteration_num": 30000,
    "warmup_steps": 15000,
    "N_boundary": 400,  # Each Boundary
    "N_domain": 10000,
    "Domain": "x -> [0,1], y -> [0,1]",
    "Pr_Gr_Re": [0.71, 0.0, 400],
    "Sparseness": None, 
    "Adam": 0,
    "Learning Rate": 5e-4,
    "Learning Rate Scheme": "LambdaLR",
    "LBFGS": 1,
    "Note": "_Gr=0_Re=400.pth",
}

run = wandb.init(project='NS_Energy',
                 notes='Only Sparse Recon Error',
                 config=configuration)

run_id =  wandb.run.id
wandb.save('NS_Energy.py')

path = os.getcwd()
results_dir = os.path.join(path, 'models')
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

name = configuration['Note']
file_path = os.path.join(results_dir, run_id + name)
# %%
Pr = configuration['Pr_Gr_Re'][0]#普朗特数
Gr = configuration['Pr_Gr_Re'][1]#格拉晓夫数
Re = configuration['Pr_Gr_Re'][2]#雷诺数

# Samples taken from each region for optimisation purposes.
N_b = configuration['N_boundary']
# N_bl = configuration['N_block']
N_f = configuration['N_domain']

X_top, X_BLR, X_f = data_sampling(N_b, N_f)
XY_star, T_star, theta_txt_grid = dataT_sampling()
# Converting to tensors
X_top_torch = torch_tensor_grad(X_top, device)
X_BLR_torch = torch_tensor_grad(X_BLR, device)
X_f_torch = torch_tensor_grad(X_f, device)
XY_star_torch = torch_tensor_grad(XY_star, device)
T_star_torch = torch_tensor_grad(T_star, device)

# Loading some sparse Simulation Data
if configuration['Sparseness'] != None:
    sparseness = configuration['Sparseness'] / 100
    data_size = len(theta_txt_grid.flatten())
    sparse_indices = np.random.randint(data_size, size=int(sparseness * data_size))

    XY_sparse = XY_star[sparse_indices]
    T_sparse = T_star[sparse_indices]
    # X_sparse_torch = torch_tensor_grad(X_sparse, device)
    # Y_sparse_torch = torch_tensor_grad(Y_sparse, device)
# %%
# 初始化网络和优化器
net = MLP(2, 4, configuration['Layers'], configuration['Neurons'])
net = net.to(device)
# use the modules apply function to recursively apply the initialization
if configuration['Flag_initialization']:
    net.init_weights(init_type="xavier")

iteration_num = configuration['iteration_num']
if configuration['LBFGS']:
    optimizer = LBFGS(net.parameters(), line_search_fn="strong_wolfe")
else:
    # 定义学习率调度函数
    warmup_steps = configuration['warmup_steps']
    def get_lr_lambda(current_step):
        # Warmup阶段：线性增加
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))

        # 余弦衰减阶段：平滑衰减
        progress = float(current_step - warmup_steps) / float(
            max(1, iteration_num - warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    # Optimizer
    optimizer = optim.Adam(net.parameters(), lr=configuration['Learning Rate'])
    # 使用LambdaLR来设置学习率调度器
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, get_lr_lambda)

if configuration['Flag_pretrain']:
    print('Reading previous results')
    checkpoint = torch.load("modelsj7xo73od_Gr=0_Re=100.path")
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

wandb.watch(net, log='all')
wandb.run.summary['Params'] = net.count_params()
# %%
Lambda, Mu, mu_max, Bar_v = paraing()
para_adapt = ParaAdapt(
    zeta=0.99, omega=0.999, eta=torch.tensor([[1.0], [1.0]]).to(device), epsilon=1e-16
)

My_PINN(X_top_torch, X_BLR_torch, X_f_torch, XY_star_torch, T_star_torch, Gr, Re, Pr, para_adapt, Lambda, Mu, mu_max, Bar_v, iteration_num)
# %%
# # Saving the trained models
# wandb.save(path + '/models/' + run_id + '_NS_Block.pth')
wandb.run.finish()
sys.exit()
# %%
