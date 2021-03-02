import gurobipy as grb
import itertools
import numpy as np
import matplotlib.pyplot as plt

def solve_sub_problem(epsilons, A, x_0):
        num_t = len(epsilons)
        num_x = num_t + 1
        set_T = range(0,num_t)
        set_X = range(0,num_x)
        # initial optimize model
        opt_model = grb.Model()
        opt_model.setParam('OutputFlag', 0)
        # add variables
        mu_vars = {i: opt_model.addVar(lb=mu_lb, ub=mu_ub, vtype=grb.GRB.CONTINUOUS,name="mu_{0}".format(i)) for i in set_T}
        x_vars = {i: opt_model.addVar(lb=-float('inf'), ub=float('inf'), vtype=grb.GRB.CONTINUOUS,name="x_{0}".format(i)) for i in set_X}
        # add constraints
        constraints = {t : 
        opt_model.addLConstr(
                lhs=x_vars[t+1],
                sense=grb.GRB.EQUAL,
                rhs=A*x_vars[t]+ mu_vars[t]+epsilons[t], 
                name="constraint_{0}".format(t))
        for t in set_T}

        constraints_x0 = {
        opt_model.addLConstr(
                lhs=x_vars[0],
                sense=grb.GRB.EQUAL,
                rhs=x_0, 
                name="constraint_x0")
        }
        # add objective function
        objective = grb.quicksum(x_vars[t]**2 + mu_vars[t]**2 for t in set_T) + x_vars[set_X[-1]]**2
        opt_model.ModelSense = grb.GRB.MINIMIZE
        opt_model.setObjective(objective)

        opt_model.optimize()
        # print(opt_model.display())
        return [[mu_vars[i].x for i in set_T], [x_vars[i].x for i in set_X]]

def solve_lagrange_sub_problem(epsilons, A, x_0, w_s, r, last_policy):
        num_t = len(epsilons)
        num_x = num_t + 1
        set_T = range(0,num_t)
        set_X = range(0,num_x)
        # initial optimize model
        opt_model = grb.Model()
        opt_model.setParam('OutputFlag', 0)
        # add variables
        mu_vars = {i: opt_model.addVar(lb=mu_lb, ub=mu_ub, vtype=grb.GRB.CONTINUOUS,name="mu_{0}".format(i)) for i in set_T}
        x_vars = {i: opt_model.addVar(lb=-float('inf'), ub=float('inf'), vtype=grb.GRB.CONTINUOUS,name="x_{0}".format(i)) for i in set_X}
        # add constraints
        constraints = {t : 
        opt_model.addLConstr(
                lhs=x_vars[t+1],
                sense=grb.GRB.EQUAL,
                rhs=A*x_vars[t]+ mu_vars[t]+epsilons[t], 
                name="constraint_{0}".format(t))
        for t in set_T}

        constraints_x0 = {
        opt_model.addLConstr(
                lhs=x_vars[0],
                sense=grb.GRB.EQUAL,
                rhs=x_0, 
                name="constraint_x0")
        }
        # add objective function
        objective = grb.quicksum(x_vars[t]**2 + mu_vars[t]**2 for t in set_T) + x_vars[set_X[-1]]**2 + grb.quicksum(w_s[t]*mu_vars[t] for t in set_T) + r/2 * grb.quicksum((mu_vars[t]-last_policy[t])**2 for t in set_T)
        opt_model.ModelSense = grb.GRB.MINIMIZE
        opt_model.setObjective(objective)

        opt_model.optimize()
        # print(opt_model.display())
        return [[mu_vars[i].x for i in set_T], [x_vars[i].x for i in set_X]]

# set hyperparameters
num_t = 2 # time periods 
# x_{t+1} = A*x_{t} + \mu_{t} + \epsilon_{t}
A = 5
x_0 = 1 # x_{0}
mu_lb = -5 # lower bound of \mu
mu_ub = 5 # upper bound of \mu
# mu_lb = -float('inf') # lower bound of \mu
# mu_ub = float('inf') # upper bound of \mu
r = 100 # penalty factor
max_iter_num = 10000 # max number of iterations
tol = 1e-4 # terminate threshold

# generate all scenarios
epsilon_element = [1,-1]
scenarios_list = []
for scenario in itertools.product(epsilon_element, repeat=num_t):
        scenarios_list.append(scenario)
# initialize information price W 
w = np.zeros((len(scenarios_list), num_t))
opt_sol_for_each_scenario = []
policy_list = []
# get initial solutions 
for scenario in scenarios_list:
        opt_sol, opt_state = solve_sub_problem(epsilons=scenario,A=A,x_0=x_0)
        opt_sol_for_each_scenario.append(opt_sol)
opt_sol_for_each_scenario = np.array(opt_sol_for_each_scenario)
implementable_policy = opt_sol_for_each_scenario.mean(axis=0)
policy_list.append(implementable_policy)
w = r * (opt_sol_for_each_scenario - implementable_policy)
# iteration
for k in range(max_iter_num):
    opt_sol_for_each_scenario = []
    for index, scenario in enumerate(scenarios_list):
        opt_sol, opt_state = solve_lagrange_sub_problem(epsilons=scenario, A=A, x_0=x_0,w_s=w[index],r=r,last_policy=implementable_policy) 
        opt_sol_for_each_scenario.append(opt_sol)
    opt_sol_for_each_scenario = np.array(opt_sol_for_each_scenario)
    # early stop
    if np.linalg.norm(implementable_policy - opt_sol_for_each_scenario.mean(axis=0)) < tol:
                break
    implementable_policy = opt_sol_for_each_scenario.mean(axis=0)
    policy_list.append(implementable_policy)
    # update w
    w = w + r * (opt_sol_for_each_scenario - implementable_policy)
print(policy_list[-1])
# get optimal solution
opt_sol,opt_state = solve_sub_problem(epsilons=[0]*num_t, A=A, x_0=x_0)
print(opt_sol)
# plot iteration figure
policy_list = np.array(policy_list)
index_list = range(len(policy_list))
for i in range(policy_list.shape[1]):
        plt.plot(index_list, policy_list[:,i], label = 'mu_'+str(i))
        plt.plot(index_list, [opt_sol[i]]*len(index_list), linestyle="--",label = 'opt_mu_'+str(i))
plt.legend()
plt.rcParams["figure.figsize"] = 10,10
plt.show()

# def solve_entire_problem(A, x_0,mu_lb,mu_ub):
#         set_T = range(0,2)
#         opt_model = grb.Model()
#         opt_model.setParam('OutputFlag', 0)
#         mu_vars = {i: opt_model.addVar(lb=mu_lb, ub=mu_ub, vtype=grb.GRB.CONTINUOUS,name="mu_{0}".format(i)) for i in set_T}
#         objective = pow(mu_vars[0],2)+pow(mu_vars[1],2) + 1/2*pow(A*x_0+mu_vars[0]+1,2)+ 1/2*pow(A*x_0+mu_vars[0]-1,2)+1/4*pow(A*A*x_0+A*mu_vars[0]+mu_vars[1]+A+1,2)+1/4*pow(A*A*x_0+A*mu_vars[0]+mu_vars[1]+A-1,2)+1/4*pow(A*A*x_0+A*mu_vars[0]+mu_vars[1]-A+1,2)+1/4*pow(A*A*x_0+A*mu_vars[0]+mu_vars[1]-A-1,2)
#         opt_model.ModelSense = grb.GRB.MINIMIZE
#         opt_model.setObjective(objective)
#         opt_model.optimize()
#         return [[mu_vars[i].x for i in set_T]]

# solve_entire_problem(A=A, x_0=x_0, mu_lb=mu_lb, mu_ub=mu_ub)
