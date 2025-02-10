import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import wandb
from functools import reduce
import operator
from loss import pde, boundary_upper, boundary_BLR, data_p, reconstruction
from sample import data_sampling, dataT_sampling
from model import MLP
from torch.optim import Adam, LBFGS
from adam_lbfgs import Adam_LBFGS
# Solve steady 2D N.S. and Energy
# %%
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.0)
# 定义一个函数来计算参数数量（适用于 PyTorch）
def count_params(model):
    return sum(p.numel() for p in model.parameters())

# %%
def get_opt(opt_name, opt_params, model_params):
    if opt_name == 'adam':
        return Adam(model_params, lr=0.001, betas=(0.9, 0.999))
    elif opt_name == 'lbfgs':
        if "history_size" in opt_params:
            opt_params["history_size"] = int(opt_params["history_size"])
        return LBFGS(model_params, max_iter=10, line_search_fn = 'strong_wolfe')
    elif opt_name == 'adam_lbfgs':
        if "switch_epochs" not in opt_params:
            raise KeyError("switch_epochs is not specified for Adam_LBFGS optimizer.")
        switch_epochs = opt_params["switch_epochs"]

        # Ensure switch_epochs is a list of integers
        if not isinstance(switch_epochs, list):
            switch_epochs = [switch_epochs]
        switch_epochs = [int(epoch) for epoch in switch_epochs]

        # Get parameters for Adam and LBFGS, remove the prefix "adam_" and "lbfgs_" from the keys
        adam_params = {k[5:]: v for k, v in opt_params.items() if k.startswith("adam_")}
        lbfgs_params = {k[6:]: v for k, v in opt_params.items() if k.startswith("lbfgs_")}
        lbfgs_params["line_search_fn"] = "strong_wolfe"

        # If max_iter or history_size is specified, convert them to integers
        if "max_iter" in lbfgs_params:
            lbfgs_params["max_iter"] = int(lbfgs_params["max_iter"])
        if "history_size" in lbfgs_params:
            lbfgs_params["history_size"] = int(lbfgs_params["history_size"])

        return Adam_LBFGS(model_params, switch_epochs, adam_params, lbfgs_params)
# %%
def paraing():

    Mu = torch.ones(1 + 1 +1, 1, device=device) # BC and PDE constraints
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
            pde_loss = loss_cont1 +  loss_enery1

            boundary_loss = boundary_upper(net, X_top_torch) + boundary_BLR(net, X_BLR_torch)

            p_loss = data_p(net, X_f_torch)
            dataT_loss = reconstruction(net, XY_star_torch, T_star_torch)
            data_loss =  p_loss + dataT_loss

            objective = loss_mom1 + loss_mom2

            constr = torch.cat((boundary_loss, pde_loss, data_loss), dim=0)

            loss = objective + Lambda.T @ constr + 0.5 * Mu.T @ constr.pow(2)
            return objective, constr, loss

        def closure():
            if torch.is_grad_enabled():
                net.train()
                opt.zero_grad()

            objective, constr, loss = ( _closure())
            if loss.requires_grad:
                loss.backward()
            return loss

        opt.step(closure)

        objective, constr, loss = ( _closure())
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
            print(
                "n = %d, loss = %.3e, objective = %.3e, constr_loss = %.3e, %.3e, %.3e, Lambda = %.3e,  %.3e,  %.3e, Mu = %.3e,  %.3e，%.3e"
                % (
                    iter + 1,
                    loss,
                    objective,
                    constr[0],
                    constr[1],
                    constr[2],
                    Lambda[0],
                    Lambda[1],
                    Lambda[2],
                    Mu[0],
                    Mu[1],
                    Mu[2],
                )
            )
            torch.save({
                'epoch': iter,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'loss': loss
            }, file_path)

            wandb.log(
                {
                    "Momentum Loss": objective,
                    "Boundary Loss": constr[0],
                    "Con_EN Loss": constr[1],
                    "data Loss": constr[2],
                    "Total Loss ": loss,
                    "LambdaB": Lambda[0],
                    "LambdaP": Lambda[1],
                    "LambdaT": Lambda[2],
                    "MuB": Mu[0],
                    "MuP": Mu[1],
                    "MuT": Mu[2],
                }
            )
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
    "Flag_pretrain":True,
    "Flag_initialization": False,
    "iteration_num": 20000,
    "N_boundary": 400,  # Each Boundary
    "N_domain": 10000,
    "Domain": "x -> [0,1], y -> [0,1]",
    "Pr_Gr_Re": [0.71,1E6, 100],
    "Sparseness": None, 
    "opt_name": ['adam', 'lbfgs', 'adam_lbfgs'],
    "opt_params": {"adam_lr": 5e-4, "adam_betas": (0.9, 0.999),
                   "lbfgs_max_iter": 10, "lbfgs_line_search_fn":None,
                   "switch_epochs": 10000},
    "Note": "_Gr=1E6_Re=100.pth",
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
    net.apply(init_weights)

iteration_num = configuration['iteration_num']

opt_params = configuration['opt_params']
opt = get_opt(configuration['opt_name'][1], opt_params, net.parameters())

if configuration['Flag_pretrain']:
    print('Reading previous results')
    checkpoint = torch.load("fbvqr5lq_Gr=1E6_Re=1000.pth")
    net.load_state_dict(checkpoint['model_state_dict'])


wandb.watch(net, log='all')
wandb.run.summary["Params"] = count_params(net)
# %%
Lambda, Mu, mu_max, Bar_v = paraing()
para_adapt = ParaAdapt(
    zeta=0.99, omega=0.999, eta=torch.tensor([[1.0], [0.01], [1.0]]).to(device), epsilon=1e-16
)

My_PINN(X_top_torch, X_BLR_torch, X_f_torch, XY_star_torch, T_star_torch, Gr, Re, Pr, para_adapt, Lambda, Mu, mu_max, Bar_v, iteration_num)
# %%
# # Saving the trained models
# wandb.save(path + '/models/' + run_id + '_NS_Block.pth')
wandb.run.finish()
sys.exit()
# %%
