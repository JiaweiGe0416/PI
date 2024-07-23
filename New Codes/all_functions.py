import torch
import torch.nn as nn # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim # For all Optimization algorithms, SGD, Adam, etc.
import torch.nn.functional as F # All functions that don't have any parameters
from torch.autograd import Variable

from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from quantile_forest import RandomForestQuantileRegressor
from scipy.interpolate import UnivariateSpline

import cvxpy as cp
import numpy as np
from numpy import linalg
import pandas as pd

from scipy.linalg import sqrtm


from matplotlib import pyplot as plt

import seaborn as sns
import matplotlib.patches as patches
import matplotlib.lines as lines

random_seed = 42

class NN1(nn.Module):
    def __init__(self, input_size, output_size):
        super(NN1, self).__init__()
        self.fc1 = nn.Linear(input_size, 10)
        self.fc2 = nn.Linear(10, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
# 2-layer NN
class NN2(nn.Module):
    def __init__(self, input_size, output_size):
        super(NN2, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x 
    
# Random Forest
# model setup: rf = RandomForestRegressor(n_estimators = 500, random_state = 18)
# training RF: rf.fit(X_pre, Y_pre)
# predict with RF: prediction = rf.predict(X_t)

# XGBoost
# model setup: xgb = XGBRegressor(n_estimators=300,random_state=0)
# training XGBoost: xgb.fit(X_pre,Y_pre)
#prediction with XGBoost: xgb.predict(X_t)
def mean_RKHS(K, Y):
    n = K.shape[0]
    a = cp.Variable(n)
    sqrt_K = sqrtm(K)
    constraints = []
    prob = cp.Problem(cp.Minimize(cp.sum_squares(Y-K@a)+cp.sum_squares(sqrt_K@a)), constraints)
    prob.solve()
    return a.value


def mean_est(est_type,X_pre,Y_pre,X_opt,X_adj,X_t, degree = 1, k=3):
    
    """
        est_type: 
        "NN1": 1-layer NN;    "NN2": 2-layer NN;    "rf": random forest; 
        "gb": gradient boosting;      "lin": linear regression;    "poly2": polynomial with degree 2;
        "rkhs_poly2": RKHS with polynomial kernel (degree=2);      "rkhs_rbf": RKHS with RBF kernel
    
        (X_pre,Y_pre): training data
        X_opt,X_adj,X_t: data used to predict
        output: mean estimator m and the predictions m(X)
    """
    input_dim = X_pre.shape[1]
    if est_type == "NN1":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42) 
        model = NN1(input_size=input_dim, output_size=1).to(device)
        criterion=nn.MSELoss()
        learning_rate = 0.001
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        for epoch in range(1000):
            #convert numpy array to torch Variable
            inputs=Variable(torch.from_numpy(X_pre))
            labels=Variable(torch.from_numpy(Y_pre))
            
            #clear gradients wrt parameters
            optimizer.zero_grad()
    
            #Forward to get outputs
            outputs=model(inputs.float())
    
            #calculate loss
            loss=criterion(outputs.float(), labels.float())
    
            #getting gradients wrt parameters
            loss.backward()
    
            #updating parameters
            optimizer.step()
        M_pre = model(torch.from_numpy(X_pre).float())
        M_pre = M_pre.detach().cpu().numpy().reshape(-1,1)
        M_opt = model(torch.from_numpy(X_opt).float())
        M_opt = M_opt.detach().cpu().numpy().reshape(-1,1)
        M_adj = model(torch.from_numpy(X_adj).float())
        M_adj = M_adj.detach().cpu().numpy().reshape(-1,1)
        M_t = model(torch.from_numpy(X_t).float())
        M_t = M_t.detach().cpu().numpy().reshape(-1,1)
        return M_pre, M_opt, M_adj, M_t
    if est_type == "NN2":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42) 
        model = NN2(input_size=input_dim, output_size=1).to(device)
        criterion=nn.MSELoss()
        learning_rate = 0.001
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        for epoch in range(1000):
            #convert numpy array to torch Variable
            inputs=Variable(torch.from_numpy(X_pre))
            labels=Variable(torch.from_numpy(Y_pre))
    
            #clear gradients wrt parameters
            optimizer.zero_grad()
    
            #Forward to get outputs
            outputs=model(inputs.float())
    
            #calculate loss
            loss=criterion(outputs.float(), labels.float())
    
            #getting gradients wrt parameters
            loss.backward()
    
            #updating parameters
            optimizer.step()
        M_pre = model(torch.from_numpy(X_pre).float())
        M_pre = M_pre.detach().cpu().numpy().reshape(-1,1)
        M_opt = model(torch.from_numpy(X_opt).float())
        M_opt = M_opt.detach().cpu().numpy().reshape(-1,1)
        M_adj = model(torch.from_numpy(X_adj).float())
        M_adj = M_adj.detach().cpu().numpy().reshape(-1,1)
        M_t = model(torch.from_numpy(X_t).float())
        M_t = M_t.detach().cpu().numpy().reshape(-1,1)
        return M_pre, M_opt, M_adj, M_t
    
    if est_type == "rf":
        model = RandomForestRegressor(n_estimators = 500, random_state=random_seed,criterion='squared_error')
        model.fit(X_pre, Y_pre.reshape(-1))
        M_pre = model.predict(X_pre).reshape(-1,1)
        M_opt = model.predict(X_opt).reshape(-1,1)
        M_adj = model.predict(X_adj).reshape(-1,1)
        M_t = model.predict(X_t).reshape(-1,1)
        return M_pre, M_opt, M_adj, M_t
    
    if est_type == "gb":
        model = GradientBoostingRegressor(n_estimators=300,random_state=random_seed,loss = "squared_error")
        model.fit(X_pre, Y_pre.reshape(-1))
        M_pre = model.predict(X_pre).reshape(-1,1)
        M_opt = model.predict(X_opt).reshape(-1,1)
        M_adj = model.predict(X_adj).reshape(-1,1)
        M_t = model.predict(X_t).reshape(-1,1)
        return M_pre, M_opt, M_adj, M_t
    
    if est_type == "lin":
        model = LinearRegression()
        model.fit(X_pre,Y_pre)
        M_pre = model.predict(X_pre).reshape(-1,1)
        M_opt = model.predict(X_opt).reshape(-1,1)
        M_adj = model.predict(X_adj).reshape(-1,1)
        M_t = model.predict(X_t).reshape(-1,1)
        return M_pre, M_opt, M_adj, M_t
    
    if est_type == "poly":
        model = LinearRegression()
        poly_regr = PolynomialFeatures(degree=degree, include_bias=False)
        X_pre_poly = poly_regr.fit_transform(X_pre) 
        model.fit(X_pre_poly, Y_pre)
        X_opt_poly = poly_regr.fit_transform(X_opt)
        X_adj_poly = poly_regr.fit_transform(X_adj) 
        X_t_poly = poly_regr.fit_transform(X_t)
        M_pre = model.predict(X_pre_poly).reshape(-1,1)
        M_opt = model.predict(X_opt_poly).reshape(-1,1)
        M_adj = model.predict(X_adj_poly).reshape(-1,1)
        M_t = model.predict(X_t_poly).reshape(-1,1)
        return M_pre, M_opt, M_adj, M_t

    
    if est_type == "rkhs_poly":
        X_inner_prod = X_pre @ X_pre.T
        K = np.power(1 + X_inner_prod, degree)
        a = mean_RKHS(K,Y_pre[:,0])
        
        M_pre = K@a
        M_pre = M_pre.reshape(-1,1)
        
        X_inner_prod_opt = X_opt @ X_pre.T
        K_opt = np.power(1 + X_inner_prod_opt, degree)
        M_opt = K_opt@a
        M_opt = M_opt.reshape(-1,1)
        
        X_inner_prod_adj = X_adj @ X_pre.T
        K_adj = np.power(1 + X_inner_prod_adj, degree)
        M_adj = K_adj@a
        M_adj = M_adj.reshape(-1,1)
        
        X_inner_prod_t = X_t @ X_pre.T
        K_t = np.power(1 + X_inner_prod_t, degree)
        M_t = K_t@a
        M_t = M_t.reshape(-1,1)
        return M_pre, M_opt, M_adj, M_t
    
    if est_type == "rkhs_rbf":
        sigma = 1
        K = rbf_kernel(X_pre, gamma = 1/(2*sigma**2))
        a = mean_RKHS(K,Y_pre[:,0])
        
        M_pre = K@a
        M_pre = M_pre.reshape(-1,1)
        
        K_opt = rbf_kernel(X_opt, X_pre, gamma = 1/(2*sigma**2))
        M_opt = K_opt@a
        M_opt = M_opt.reshape(-1,1)
        
        K_adj = rbf_kernel(X_adj, X_pre, gamma = 1/(2*sigma**2))
        M_adj = K_adj@a
        M_adj = M_adj.reshape(-1,1)
        
        K_t = rbf_kernel(X_t, X_pre, gamma = 1/(2*sigma**2))
        M_t = K_t@a
        M_t = M_t.reshape(-1,1)
        return M_pre, M_opt, M_adj, M_t
    
    if est_type == "spline":
        sorted_indices = np.argsort(X_pre[:,0])
        X_sorted = X_pre[sorted_indices]
        Y_sorted = Y_pre[sorted_indices]
        spline = UnivariateSpline(X_sorted, Y_sorted, k=k) # Adjust 'k' for the desired spline order
        M_pre = spline(X_pre).reshape(-1,1)
        M_opt = spline(X_opt).reshape(-1,1)
        M_adj = spline(X_adj).reshape(-1,1)
        M_t = spline(X_t).reshape(-1,1)
        return M_pre, M_opt, M_adj, M_t
        
	
def var_est(X_pre,Y_pre,M_pre, X_opt,X_adj,X_t, max_depth, est_type ="NN1"):
    X = X_pre
    Y = (Y_pre-M_pre)**2
    input_dim = X.shape[1]
    if est_type == "NN1":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42) 
        model = NN1(input_size=input_dim, output_size=1).to(device)
        criterion=nn.MSELoss()
        learning_rate = 0.001
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        for epoch in range(1000):
            #convert numpy array to torch Variable
            inputs=Variable(torch.from_numpy(X))
            labels=Variable(torch.from_numpy(Y))
            
            #clear gradients wrt parameters
            optimizer.zero_grad()
    
            #Forward to get outputs
            outputs=model(inputs.float())
    
            #calculate loss
            loss=criterion(outputs.float(), labels.float())
    
            #getting gradients wrt parameters
            loss.backward()
    
            #updating parameters
            optimizer.step()
        var_opt = model(torch.from_numpy(X_opt).float())
        var_opt = var_opt.detach().cpu().numpy().reshape(-1,1)
        var_opt = np.maximum(var_opt,0)
        var_adj = model(torch.from_numpy(X_adj).float())
        var_adj = var_adj.detach().cpu().numpy().reshape(-1,1)
        var_adj = np.maximum(var_adj,0)
        var_t = model(torch.from_numpy(X_t).float())
        var_t = var_t.detach().cpu().numpy().reshape(-1,1)
        var_t = np.maximum(var_t,0)
    
        
    if est_type == "RF":
        var_model = RandomForestRegressor(n_estimators = 1000, max_depth = max_depth, random_state = 42 )
        var_model.fit(X, Y.reshape(-1,1))
    
        var_opt = var_model.predict(X_opt)
        var_adj = var_model.predict(X_adj)
        var_t = var_model.predict(X_t)
        
        
    return var_opt, var_adj, var_t 
	
	
def quantile_loss(preds, target, quantile):
    assert not target.requires_grad
    assert preds.size(0) == target.size(0)
    errors = target - preds
    q = quantile
    losses = torch.max((q - 1) * errors, q * errors)
    loss = torch.sum(losses)
    return loss

def est_quantile(est_type,quantile,X_pre,Y_pre,X_opt,X_adj,X_t):
    """
        est_type: 
        "NN1": 1-layer NN;              "NN2": 2-layer NN; 
        "qrf": quantile regression forest;    "gb": gradient boosting
        
        quantile: the quantile we are estimating
        (X_pre,Y_pre): training data
        X_opt,X_adj,X_t: data used to predict
        output: quantile estimator Q and the prediction Q(X) 
    """
    input_dim = X_pre.shape[1]
    if est_type == "NN1":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42) 
        model = NN1(input_size=input_dim, output_size=1).to(device)
        learning_rate = 0.001
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        for epoch in range(1000):
            #convert numpy array to torch Variable
            inputs=Variable(torch.from_numpy(X_pre))
            labels=Variable(torch.from_numpy(Y_pre))
        
            #clear gradients wrt parameters
            optimizer.zero_grad()
    
            #Forward to get outputs
            outputs=model(inputs.float())
    
            #calculate loss
            loss=quantile_loss(outputs, labels, quantile)
    
            #getting gradients wrt parameters
            loss.backward()
    
            #updating parameters
            optimizer.step()
        Q_opt = model(torch.from_numpy(X_opt).float())
        Q_opt = Q_opt.detach().cpu().numpy().reshape(-1,1)
        Q_adj = model(torch.from_numpy(X_adj).float())
        Q_adj = Q_adj.detach().cpu().numpy().reshape(-1,1)
        Q_t = model(torch.from_numpy(X_t).float())
        Q_t = Q_t.detach().cpu().numpy().reshape(-1,1)
        return model, Q_opt, Q_adj, Q_t
    
    if est_type == "NN2":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42) 
        model = NN2(input_size=input_dim, output_size=1).to(device)
        learning_rate = 0.001
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        for epoch in range(1000):
            #convert numpy array to torch Variable
            inputs=Variable(torch.from_numpy(X_pre))
            labels=Variable(torch.from_numpy(Y_pre))
        
            #clear gradients wrt parameters
            optimizer.zero_grad()
    
            #Forward to get outputs
            outputs=model(inputs.float())
    
            #calculate loss
            loss=quantile_loss(outputs, labels, quantile)
    
            #getting gradients wrt parameters
            loss.backward()
    
            #updating parameters
            optimizer.step()
        Q_opt = model(torch.from_numpy(X_opt).float())
        Q_opt = Q_opt.detach().cpu().numpy().reshape(-1,1)
        Q_adj = model(torch.from_numpy(X_adj).float())
        Q_adj = Q_adj.detach().cpu().numpy().reshape(-1,1)
        Q_t = model(torch.from_numpy(X_t).float())
        Q_t = Q_t.detach().cpu().numpy().reshape(-1,1)
        return model, Q_opt, Q_adj, Q_t
    
    if est_type == "qrf":
        model = RandomForestQuantileRegressor(n_estimators = 500, random_state=random_seed)
        model.fit(X_pre, Y_pre.reshape(-1))
        Q_opt = model.predict(X_opt,quantiles = [quantile]).reshape(-1,1)
        Q_adj = model.predict(X_adj,quantiles = [quantile]).reshape(-1,1)
        Q_t = model.predict(X_t,quantiles = [quantile]).reshape(-1,1)
        return model, Q_opt, Q_adj, Q_t
    
    
    if est_type == "gb":
        model = GradientBoostingRegressor(n_estimators=300,random_state=random_seed,loss = "quantile", alpha = quantile)
        model.fit(X_pre, Y_pre.reshape(-1))
        Q_opt = model.predict(X_opt).reshape(-1,1)
        Q_adj = model.predict(X_adj).reshape(-1,1)
        Q_t = model.predict(X_t).reshape(-1,1)
        return model, Q_opt, Q_adj, Q_t
	
def RKHS_opt(K, Y):
    n = K.shape[0]
    hB = cp.Variable((n, n), symmetric=True)
    constraints = [hB >> 0]
    constraints += [K[i, :] @ hB @ K[i, :] >= cp.square(Y[i]) for i in range(n)]
    prob = cp.Problem(cp.Minimize(cp.trace(K @ hB @ K.T)), constraints)
    prob.solve()
    return hB.value


def solve_opt(X_opt,Y_opt, M_opt, M_adj, M_t, X_adj, X_t,
              function_class, E_opt=None, E_adj=None, E_t=None, degree = None,sigma = None):
    """
        M_opt: mean estimator m(X_opt)
        
        function_class: 
        "aug": Augmentation, "rkhs_poly": RKHS with polynomial kernel, "rkhs_rbf": RKHS with RBF kernel
        
        E_opt: if function_class = "aug", then E is the estimator matrix, i.e. (E_opt)_{ij} = f_i((X_opt)_j)
        degree: if function_class = "rkhs_poly", then set the degree of the polynomial
        sigma: if function_class = "rkhs_rbf", then set parameter sigma
        
        (X_opt,Y_opt): data used to solve the optimization problem
        output: estimator for 100% coverage V_adj and V_t, optimal solution
    """
    n_opt = X_opt.shape[0]
    n_adj = X_adj.shape[0]
    n_t = X_t.shape[0]
    Y = (Y_opt-M_opt)[:,0]
        
    if function_class == "aug":
        cons_opt = np.ones(n_opt).reshape(1,-1)
        A_opt = np.vstack((E_opt,cons_opt))
        weight = cp.Variable(A_opt.shape[0])
        constraints = [weight>=0]+[weight @ A_opt >= cp.square(Y)]
        prob = cp.Problem(cp.Minimize(cp.sum(weight @ A_opt)), constraints)
        prob.solve()
        optimal_weight = weight.value
        
        cons_adj = np.ones(n_adj).reshape(1,-1)
        A_adj = np.vstack((E_adj,cons_adj))
        V_adj = optimal_weight @ A_adj
        V_adj = V_adj.reshape(-1,1)
        
        cons_t = np.ones(n_t).reshape(1,-1)
        A_t = np.vstack((E_t,cons_t))
        V_t = optimal_weight @ A_t
        V_t = V_t.reshape(-1,1)
        return optimal_weight, V_adj, V_t
    
    if function_class == "rkhs_poly":
        X_inner_prod = X_opt @ X_opt.T
        K = np.power(1 + X_inner_prod, degree)
        hB = RKHS_opt(K,Y)
        
        X_inner_prod_adj = X_adj @ X_opt.T
        K_adj = np.power(1 + X_inner_prod_adj, degree)
        V_adj = np.diag(K_adj @ hB @ K_adj.T)
        V_adj = V_adj.reshape(-1,1)
        
        X_inner_prod_t = X_t @ X_opt.T
        K_t = np.power(1 + X_inner_prod_t, degree)
        V_t = np.diag(K_t @ hB @ K_t.T)
        V_t = V_t.reshape(-1,1)
        return hB, V_adj, V_t
    
    if function_class == "rkhs_rbf":
        K_opt = rbf_kernel(X_opt, gamma = 1/(2*sigma**2))
        hB = RKHS_opt(K_opt,Y)
        
        K_adj = rbf_kernel(X_adj, X_opt, gamma = 1/(2*sigma**2))
        V_adj = np.diag(K_adj @ hB @ K_adj.T)
        V_adj = V_adj.reshape(-1,1)
        
        K_t = rbf_kernel(X_t, X_opt, gamma = 1/(2*sigma**2))
        V_t = np.diag(K_t @ hB @ K_t.T)
        V_t = V_t.reshape(-1,1)
        return hB, V_adj, V_t
    
	
	
def interval_adj(X_adj,Y_adj,M_adj,V_adj,alpha, stepsize=0.001,eps=0):
    """
        (X_adj,Y_adj): data used to adjust the interval
         M_adj: mean estimator m(X_adj)
         V_adj: variance estimator for 100% coverage f(X_adj)
         alpha: get 1-alpha coverage
         eps: extend the interval a bit 1/sqrt{log(n_opt)}
         output: adjustment level delta
         the prediction level is: [M-sqrt{delta V}, M+sqrt{delta V}]
    """
    I = np.where(V_adj+eps-np.square(Y_adj-M_adj)>=0)[0]
    I.tolist()
    Y_adj = Y_adj[I].reshape(-1,1)
    M_adj = M_adj[I].reshape(-1,1)
    V_adj = (V_adj[I]+eps).reshape(-1,1)
    delta = 1
    prop_outside = (np.power(Y_adj[:,0]-M_adj[:,0], 2) > delta * V_adj[:,0]).mean()
    while prop_outside <= alpha:
        delta = delta-stepsize
        prop_outside = (np.power(Y_adj[:,0]-M_adj[:,0], 2) > delta * V_adj[:,0]).mean()
    return delta


def interval_adj_2(X_adj,Y_adj,M_adj,V_adj,alpha, stepsize=0.001,eps=0):
    scores_temp = (np.power(Y_adj[:,0]-M_adj[:,0], 2))/( V_adj[:,0] + 1e-10)
    level = (1 - alpha) * 100
    level = int(level)
    threshold = np.percentile(scores_temp, level)
    return threshold


def my_plot(X_t,Y_t,M_t,V_t,X_axis="X", Y_axis="Y",ylim = [-3,3]):
    """
        (X_t,Y_t): test data
         M_t: mean estimator m(X_t)
         V_t: variance estimator f(X_t)
         The prediction interval is [M_t-sqrt{V_t},M+sqrt{V_t}]
         
         X_axis: name for the X axis
         Y_axis: name for the Y axis
    
    """
    
    X_sort = np.sort(X_t, axis=0)
    X_sort_indices = np.argsort(X_t, axis=0)
    Y_sort = Y_t[X_sort_indices[:, 0]]
    lower_CI = M_t-np.sqrt(V_t)
    lower_CI_sort = lower_CI[X_sort_indices[:, 0]]
    upper_CI = M_t+np.sqrt(V_t)
    upper_CI_sort = upper_CI[X_sort_indices[:, 0]]
    mean = M_t[X_sort_indices[:, 0]]
    
    sns.set()
    sns.set_style("darkgrid")
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
    palette = sns.color_palette("Blues_r", 4)
    sns.scatterplot(x=X_sort[:,0], y=Y_sort[:,0], color=palette[0], edgecolor='w', linewidth=0.5)
    plt.fill_between(X_sort[:,0], lower_CI_sort[:,0], upper_CI_sort[:,0], color=palette[1], alpha=0.4)
    plt.plot(X_sort, lower_CI_sort, color=palette[2], lw=2,alpha=0.6)
    plt.plot(X_sort, upper_CI_sort, color=palette[2], lw=2,alpha=0.6)
    plt.plot(X_sort, mean, '-', color='orange', linewidth=2,label="Mean")
#    plt.plot(X_sort, mean, color=palette[3], linewidth=2, label="Mean")
    plt.xlabel(X_axis)
    plt.ylabel(Y_axis)
    plt.ylim(ylim)
    legend_elements = [
    patches.Rectangle((0, 0), 1, 1, lw=0, color=palette[1], alpha=0.4, label="PI"),
    lines.Line2D([0], [0], color='orange', lw=2, label="Mean")]
    plt.legend(handles=legend_elements, loc='upper right')
#     plt.legend(loc='upper right')
    plt.show()
    # plt.savefig("plot.png", dpi=300)
    coverage = (np.power(Y_t[:,0]-M_t[:,0], 2) <= V_t[:,0]).mean()
    bandwidth = np.mean(V_t[:,0])
    print("The overall coverage is", coverage)
    print("The mean bandwidth for testing data is", bandwidth)
	
def my_plot2(X_t,Y_t,M_t,V_t,X_axis="X",Y_axis="Y"):
    """
        (X_t,Y_t): test data
         M_t: mean estimator m(X_t)
         V_t: variance estimator f(X_t)
         The prediction interval is [M_t-sqrt{V_t},M+sqrt{V_t}]
    """
    X_plot = M_t
    X_sort = np.sort(X_plot, axis=0)
    X_sort_indices = np.argsort(X_plot, axis=0)
    Y_sort = Y_t[X_sort_indices[:, 0]]
    lower_CI = M_t-np.sqrt(V_t)
    lower_CI_sort = lower_CI[X_sort_indices[:, 0]]
    upper_CI = M_t+np.sqrt(V_t)
    upper_CI_sort = upper_CI[X_sort_indices[:, 0]]
    mean = M_t[X_sort_indices[:, 0]]
        
    sns.set()
    sns.set_style("darkgrid")
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
    palette = sns.color_palette("Blues_r", 4)
    sns.scatterplot(x=X_sort[:,0], y=Y_sort[:,0], color=palette[0], edgecolor='w', linewidth=0.5)
    plt.fill_between(X_sort[:,0], lower_CI_sort[:,0], upper_CI_sort[:,0], color=palette[1], alpha=0.4)
    plt.plot(X_sort, lower_CI_sort, color=palette[2], lw=2,alpha=0.6)
    plt.plot(X_sort, upper_CI_sort, color=palette[2], lw=2,alpha=0.6)
    plt.plot(X_sort, mean, '-', color='orange', linewidth=2,label="Mean")
#    plt.plot(X_sort, mean, color=palette[3], linewidth=2, label="Mean")
    plt.xlabel(X_axis)
    plt.ylabel(Y_axis)
    legend_elements = [
    patches.Rectangle((0, 0), 1, 1, lw=0, color=palette[1], alpha=0.4, label="PI"),
    lines.Line2D([0], [0], color='orange', lw=2, label="Mean")]
    plt.legend(handles=legend_elements, loc='upper right')
#     plt.legend(loc='upper right')
    plt.show()
    # plt.savefig("plot.png", dpi=300)
    coverage = (np.power(Y_t[:,0]-M_t[:,0], 2) <= V_t[:,0]).mean()
    bandwidth = np.mean(V_t[:,0])
    print("The overall coverage is", coverage)
    print("The mean bandwidth for testing data is", bandwidth)
	
	
def my_plot3(X_t,Y_t,M_t,V_t,Y_axis="Y"):
    """
        (X_t,Y_t): test data
         M_t: mean estimator m(X_t)
         V_t: variance estimator f(X_t)
         The prediction interval is [M_t-sqrt{V_t},M+sqrt{V_t}]
    """
    lower_CI = M_t-np.sqrt(V_t)
    upper_CI = M_t+np.sqrt(V_t)
    mean = M_t
    x_axis = range(Y_t.shape[0])
    
    sns.set()
    sns.set_style("darkgrid")
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
    palette = sns.color_palette("Blues_r", 4)
    sns.scatterplot(x_axis,y=Y_t[:,0], color=palette[0], edgecolor='w', linewidth=0.5)
    plt.fill_between(x_axis, lower_CI[:,0], upper_CI[:,0], color=palette[1], alpha=0.4)
    plt.plot(x_axis, lower_CI, color=palette[2], lw=2,alpha=0.6)
    plt.plot(x_axis, upper_CI, color=palette[2], lw=2,alpha=0.6)
    plt.plot(x_axis, mean, '-', color='orange', linewidth=2,label="Mean")
#    plt.plot(X_sort, mean, color=palette[3], linewidth=2, label="Mean")
#    plt.xlabel("X")
    plt.ylabel(Y_axis)
    legend_elements = [
    patches.Rectangle((0, 0), 1, 1, lw=0, color=palette[1], alpha=0.4, label="PI"),
    lines.Line2D([0], [0], color='orange', lw=2, label="Mean")]
    plt.legend(handles=legend_elements, loc='upper right')
#     plt.legend(loc='upper right')
    plt.show()
    # plt.savefig("plot.png", dpi=300)
#     coverage = (np.power(Y_t[:,0]-M_t[:,0], 2) <= V_t[:,0]).mean()
#     bandwidth = np.mean(V_t[:,0])
#     print("The overall coverage is", coverage)
#     print("The mean bandwidth for testing data is", bandwidth)



def mean_est_others(est_type,X_lin,Y_lin,X_quantile,X_test):
    input_dim = X_lin.shape[1]
    if est_type == "NN1":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42) 
        model = NN1(input_size=input_dim, output_size=1).to(device)
        criterion=nn.MSELoss()
        learning_rate = 0.001
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        for epoch in range(1000):
            #convert numpy array to torch Variable
            inputs=Variable(torch.from_numpy(X_lin))
            labels=Variable(torch.from_numpy(Y_lin))
            
            #clear gradients wrt parameters
            optimizer.zero_grad()
    
            #Forward to get outputs
            outputs=model(inputs.float())
    
            #calculate loss
            loss=criterion(outputs.float(), labels.float())
    
            #getting gradients wrt parameters
            loss.backward()
    
            #updating parameters
            optimizer.step()
        M_quantile = model(torch.from_numpy(X_quantile).float())
        M_quantile = M_quantile.detach().cpu().numpy().reshape(-1,1)
        M_test = model(torch.from_numpy(X_test).float())
        M_test = M_test.detach().cpu().numpy().reshape(-1,1)
        return M_quantile, M_test
    if est_type == "NN2":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42) 
        model = NN2(input_size=input_dim, output_size=1).to(device)
        criterion=nn.MSELoss()
        learning_rate = 0.001
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        for epoch in range(1000):
            #convert numpy array to torch Variable
            inputs=Variable(torch.from_numpy(X_lin))
            labels=Variable(torch.from_numpy(Y_lin))
    
            #clear gradients wrt parameters
            optimizer.zero_grad()
    
            #Forward to get outputs
            outputs=model(inputs.float())
    
            #calculate loss
            loss=criterion(outputs.float(), labels.float())
    
            #getting gradients wrt parameters
            loss.backward()
    
            #updating parameters
            optimizer.step()
        M_quantile = model(torch.from_numpy(X_quantile).float())
        M_quantile = M_quantile.detach().cpu().numpy().reshape(-1,1)
        M_test = model(torch.from_numpy(X_test).float())
        M_test = M_test.detach().cpu().numpy().reshape(-1,1)
        return M_quantile, M_test
    if est_type == "rf":
        model = RandomForestRegressor(n_estimators = 500, random_state=random_seed,criterion='squared_error')
        model.fit(X_lin, Y_lin.reshape(-1))
        M_quantile = model.predict(X_quantile).reshape(-1,1)
        M_test = model.predict(X_test).reshape(-1,1)
        return M_quantile, M_test
    if est_type == "gb":
        model = GradientBoostingRegressor(n_estimators=300,random_state=random_seed,loss = "squared_error")
        model.fit(X_lin, Y_lin.reshape(-1))
        M_quantile = model.predict(X_quantile).reshape(-1,1)
        M_test = model.predict(X_test).reshape(-1,1)
        return M_quantile, M_test