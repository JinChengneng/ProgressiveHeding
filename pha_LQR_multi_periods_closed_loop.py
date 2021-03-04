import gurobipy as grb
import itertools
import numpy as np
import matplotlib.pyplot as plt

try:
    opt_model = grb.Model()
except:
    print("Please check if Gurobi is available in your devices")


def solve_sub_problem(epsilons, A, x_0):
    num_t = len(epsilons)
    num_x = num_t + 1
    set_T = range(0, num_t)
    set_X = range(0, num_x)
    # initial optimize model
    opt_model = grb.Model()
    opt_model.setParam('OutputFlag', 0)
    # add variables
    mu_vars = {i: opt_model.addVar(
        lb=mu_lb, ub=mu_ub, vtype=grb.GRB.CONTINUOUS, name="mu_{0}".format(i)) for i in set_T}
    x_vars = {i: opt_model.addVar(lb=-float('inf'), ub=float('inf'),
                                  vtype=grb.GRB.CONTINUOUS, name="x_{0}".format(i)) for i in set_X}
    # add constraints
    constraints = {t:
                   opt_model.addLConstr(
                       lhs=x_vars[t+1],
                       sense=grb.GRB.EQUAL,
                       rhs=A*x_vars[t] + mu_vars[t]+epsilons[t],
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
    objective = grb.quicksum(
        x_vars[t]**2 + mu_vars[t]**2 for t in set_T) + x_vars[set_X[-1]]**2
    opt_model.ModelSense = grb.GRB.MINIMIZE
    opt_model.setObjective(objective)

    opt_model.optimize()
    # print(opt_model.display())
    return [[mu_vars[i].x for i in set_T], [x_vars[i].x for i in set_X]]


def solve_lagrange_sub_problem(epsilons, A, x_0, w_s, r, last_policy):
    num_t = len(epsilons)
    num_x = num_t + 1
    set_T = range(0, num_t)
    set_X = range(0, num_x)
    # initial optimize model
    opt_model = grb.Model()
    opt_model.setParam('OutputFlag', 0)
    # add variables
    mu_vars = {i: opt_model.addVar(
        lb=mu_lb, ub=mu_ub, vtype=grb.GRB.CONTINUOUS, name="mu_{0}".format(i)) for i in set_T}
    x_vars = {i: opt_model.addVar(lb=-float('inf'), ub=float('inf'),
                                  vtype=grb.GRB.CONTINUOUS, name="x_{0}".format(i)) for i in set_X}
    # add constraints
    constraints = {t:
                   opt_model.addLConstr(
                       lhs=x_vars[t+1],
                       sense=grb.GRB.EQUAL,
                       rhs=A*x_vars[t] + mu_vars[t]+epsilons[t],
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
    objective = grb.quicksum(x_vars[t]**2 + mu_vars[t]**2 for t in set_T) + x_vars[set_X[-1]]**2 + grb.quicksum(
        w_s[t]*mu_vars[t] for t in set_T) + r/2 * grb.quicksum((mu_vars[t]-last_policy[t])**2 for t in set_T)
    opt_model.ModelSense = grb.GRB.MINIMIZE
    opt_model.setObjective(objective)

    opt_model.optimize()
    # print(opt_model.display())
    return [[mu_vars[i].x for i in set_T], [x_vars[i].x for i in set_X]]


def ProgressiveHedging(x_t, future_t):
    """
    x_t: the current value of x at time t.
    future_t: the remaining periods at time t.
    """
    # generate all scenarios
    scenarios_list = []
    for scenario in itertools.product(epsilon_element, repeat=future_t):
        scenarios_list.append(scenario)
    # initialize information price W
    w = np.zeros((len(scenarios_list), future_t))
    opt_sol_for_each_scenario = []
    policy_list = []
    # get initial solutions
    for scenario in scenarios_list:
        opt_sol, opt_state = solve_sub_problem(epsilons=scenario, A=A, x_0=x_t)
        opt_sol_for_each_scenario.append(opt_sol)
    opt_sol_for_each_scenario = np.array(opt_sol_for_each_scenario)
    implementable_policy = opt_sol_for_each_scenario.mean(axis=0)
    policy_list.append(implementable_policy)
    w = r * (opt_sol_for_each_scenario - implementable_policy)
    # iteration
    for k in range(max_iter_num):
        opt_sol_for_each_scenario = []
        for index, scenario in enumerate(scenarios_list):
            opt_sol, opt_state = solve_lagrange_sub_problem(
                epsilons=scenario, A=A, x_0=x_t, w_s=w[index], r=r, last_policy=implementable_policy)
            opt_sol_for_each_scenario.append(opt_sol)
        opt_sol_for_each_scenario = np.array(opt_sol_for_each_scenario)
        # early stop
        if np.linalg.norm(implementable_policy - opt_sol_for_each_scenario.mean(axis=0)) < tol:
            break
        implementable_policy = opt_sol_for_each_scenario.mean(axis=0)
        policy_list.append(implementable_policy)
        # update w
        w = w + r * (opt_sol_for_each_scenario - implementable_policy)
    if k == max_iter_num - 1:
        print("reached max iteration numbers.")
    print("The PHA optimal solution with x_t =", round(x_t, 4), "and future_t=",
          future_t, "is:", implementable_policy, "The iteration stopped at iteration", k)
    return implementable_policy, policy_list

# get closed_loop solution for each scenario


def get_closed_loop_solution(x_0, num_t, A):
    """
    implemented_action : optimal policy of each scenario.
    x_dict : optimal value of x of each scenario.
    """
    x_dict = {}
    for t in range(0, num_t):
        if t == 0:
            x_dict[t] = [x_0]
            implemented_action = {}
            implementable_policy, policy_list = ProgressiveHedging(x_0, num_t)
            implemented_action[t] = implementable_policy[0]
            x_dict[t+1] = []
            for epsilon in epsilon_element:
                x_dict[t+1].append(x_0 * A + implemented_action[0] + epsilon)
        else:
            implemented_action[t] = []
            x_dict[t+1] = []
            for x_t in x_dict[t]:
                implementable_policy, policy_list = ProgressiveHedging(
                    x_t, num_t-t)
                implemented_action[t].append(implementable_policy[0])
                for epsilon in epsilon_element:
                    x_dict[t+1].append(x_t * A +
                                       implementable_policy[0] + epsilon)
    return implemented_action, x_dict


def lqr_dp(A, B, Q, Q_f, R, N):
    P_list = [np.nan]*(N+1)
    P_list[N] = Q_f
    K_list = [np.nan]*N
    for t in range(N, 0, -1):
        P_list[t-1] = Q + A*P_list[t]*A - A*P_list[t] * \
            B*1/(R+B*P_list[t]*B)*B*P_list[t]*A
    for t in range(0, N, 1):
        K_list[t] = -1/(R+B*P_list[t+1]*B)*B*P_list[t+1]*A
    return K_list


# set hyperparameters
num_t = 3  # time periods
# x_{t+1} = A*x_{t} + \mu_{t} + \epsilon_{t}
A = 3
x_0 = 1  # x_{0}
mu_lb = -5  # lower bound of \mu
mu_ub = 5  # upper bound of \mu
# mu_lb = -float('inf') # lower bound of \mu
# mu_ub = float('inf') # upper bound of \mu
r = 10  # penalty factor
max_iter_num = 10000  # max number of iterations
tol = 1e-5  # terminate threshold
epsilon_element = [1, -1]

# get closed loop solution via PHA
optimal_policy, optimal_x = get_closed_loop_solution(x_0=x_0, num_t=num_t, A=A)
# calculate K_t
for t in range(0, num_t):
    print("K_"+str(t), ":", [round(x, 4)
                             for x in np.array(optimal_policy[t]) / np.array(optimal_x[t])])
# calculate the analytical optimal K_t
K_list = lqr_dp(A=A, B=1, Q=1, Q_f=1, R=1, N=num_t)
print("The analytical K is ", [round(x, 4) for x in K_list])
