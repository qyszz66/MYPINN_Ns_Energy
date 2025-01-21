import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"
# Setting up a derivative function that goes through the graph and calculates via chain rule the derivative of u wrt x
deriv = lambda u, x: torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, allow_unused=True)[0]
# # 创建损失函数
criterion = nn.MSELoss(reduction='mean')

def pde(net, X, Gr, Re, Pr):
    x = X[:, 0:1]
    y = X[:, 1:2]
    out = net(torch.cat([x, y], 1))

    u = out[:, 0:1]
    v = out[:, 1:2]
    p = out[:, 2:3]
    T = out[:, 3:4]

    u_x = deriv(u, x)
    v_x = deriv(v, x)
    p_x = deriv(p, x)
    T_x = deriv(T, x)

    u_xx = deriv(u_x, x)
    v_xx = deriv(v_x, x)
    T_xx = deriv(T_x, x)

    u_y = deriv(u, y)
    v_y = deriv(v, y)
    p_y = deriv(p, y)
    T_y = deriv(T, y)

    u_yy = deriv(u_y, y)
    v_yy = deriv(v_y, y)
    T_yy = deriv(T_y, y)

    cont1 = u_x + v_y
    mom1 = u * u_x + v * u_y + p_x - (1/Re) * (u_xx + u_yy)
    mom2 = u * v_x + v * v_y + p_y - (1/Re) * (v_xx + v_yy) - (Gr / Re ** 2) * T
    enery1 = u * T_x + v * T_y - 1 / (Re * Pr) * (T_xx + T_yy)

    loss_cont1 = criterion(cont1, torch.zeros_like(cont1))
    loss_mom1 = criterion(mom1, torch.zeros_like(mom1))
    loss_mom2 = criterion(mom2, torch.zeros_like(mom2))
    loss_enery1 = criterion(enery1, torch.zeros_like(enery1))

    return  (loss_cont1.reshape(-1, 1),
             loss_mom1.reshape(-1, 1),
             loss_mom2.reshape(-1, 1),
            loss_enery1.reshape(-1, 1),)

def boundary_upper(net, X):
    x = X[:, 0:1]
    y = X[:, 1:2]
    out = net(torch.cat([x, y], 1))

    u = out[:, 0:1]
    v = out[:, 1:2]

    u_top_torch = torch.ones_like(u).to(device)

    loss_top1 = criterion(u, u_top_torch)
    loss_top2 = criterion(v, 0 * v)

    bc_loss = loss_top1 + loss_top2

    return bc_loss.reshape(-1, 1)

def boundary_BLR(net, X):
    x = X[:, 0:1]
    y = X[:, 1:2]
    out = net(torch.cat([x, y], 1))

    u = out[:, 0:1]
    v = out[:, 1:2]

    loss_BLR1 = criterion(u, 0 * u)
    loss_BLR2 = criterion(v, 0 * v)

    bc_loss = loss_BLR1 + loss_BLR2

    return bc_loss.reshape(-1, 1)

def data_p(net, X):
    x = X[:, 0:1]
    y = X[:, 1:2]
    out = net(torch.cat([x, y], 1))

    p = out[:, 2:3]

    p_loss = criterion(p[0], 0*p[0])

    return p_loss.reshape(-1, 1)

def reconstruction(net, X, Tdata):
    T = net(X)[:, 3:4]

    dataT_loss = criterion(T, Tdata)

    return dataT_loss.reshape(-1, 1)

