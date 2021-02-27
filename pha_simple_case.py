import gurobipy as grb
import numpy as np 

d_list = np.array([5,2])
d_prob = np.array([0.6,0.4])

def solve_sub_problem(d):
    opt_model = grb.Model()
    opt_model.setParam('OutputFlag', 0)
    x_var = opt_model.addVar(lb= 3, ub= 6, vtype=grb.GRB.CONTINUOUS,name="x")
    objective = pow(x_var-d,2)
    opt_model.ModelSense = grb.GRB.MINIMIZE
    opt_model.setObjective(objective)
    opt_model.optimize()
    return x_var.x

def solve_lagrange_sub_problem(d,w,r,last_x):
    opt_model = grb.Model()
    opt_model.setParam('OutputFlag', 0)
    x_var = opt_model.addVar(lb= 3, ub= 6, vtype=grb.GRB.CONTINUOUS,name="x")
    objective = pow(x_var-d,2)+w*x_var+r/2*(x_var - last_x)**2
    opt_model.ModelSense = grb.GRB.MINIMIZE
    opt_model.setObjective(objective)
    opt_model.optimize()
    return x_var.x

w = [0,0]
r = 1
opt_x_for_each_d = [solve_sub_problem(d) for d in d_list]

implementable_x = sum(opt_x_for_each_d * d_prob)

for k in range(1000):
    opt_x_for_each_d = [solve_lagrange_sub_problem(d,w[index],r,implementable_x) for (index, d) in enumerate(d_list)]
    if abs(implementable_x - sum(opt_x_for_each_d * d_prob)) < 1e-4:
        break
    implementable_x = sum(opt_x_for_each_d * d_prob)
    w = w + r*(opt_x_for_each_d - implementable_x)
    print(implementable_x)
