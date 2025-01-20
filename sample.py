from scipy.interpolate import griddata
import numpy as np
from pyDOE import lhs

# Specifying the Domain of Interest.
x_range = [0.0, 1.0]
y_range = [0.0, 1.0]
lb = np.asarray([x_range[0], y_range[0]])
ub = np.asarray([x_range[1], y_range[1]])
def LHS_Sampling(N, lb, ub):
    return lb + (ub - lb) * lhs(2, N)

def data_sampling(N_b, N_f):

    # Data for Boundary Input
    X_left = LHS_Sampling(N_b, lb, ub)
    X_left[:, 0:1] = x_range[0]

    X_right = LHS_Sampling(N_b, lb, ub)
    X_right[:, 0:1] = x_range[1]

    X_bottom = LHS_Sampling(N_b, lb, ub)
    X_bottom[:, 1:2] = y_range[0]

    X_top = LHS_Sampling(N_b, lb, ub)
    X_top[:, 1:2] = y_range[1]

    X_BLR = np.vstack((X_bottom, X_left, X_right))
    # np.random.shuffle(X_BLR)

    # Data for Domain Input
    X_f = LHS_Sampling(N_f, lb, ub)

    return X_top, X_BLR, X_f

def dataT_sampling():
    # 添加温度数据点
    theta_txt_data = np.loadtxt("theta_Gr=0_Re=400.txt", comments="%")
    x_txt_theta = theta_txt_data[:, 0]
    y_txt_theta = theta_txt_data[:, 1]
    theta_txt = theta_txt_data[:, 2]
    x_min = np.min(x_txt_theta)
    x_max = np.max(x_txt_theta)
    y_min = np.min(y_txt_theta)
    y_max = np.max(y_txt_theta)
    x = np.linspace(x_min, x_max, 101)
    y = np.linspace(y_min, y_max, 101)

    X, Y = np.meshgrid(x, y)
    XY_star = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))
    theta_txt_grid = griddata((x_txt_theta, y_txt_theta), theta_txt, (X, Y), method="cubic")
    T_star = np.hstack((np.expand_dims(theta_txt_grid.flatten(), 1),))

    return XY_star, T_star, theta_txt_grid


