import gurobipy as grb
import itertools
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def solve_sub_problem(riskless_return, risk_returns, expected_risk_return, x_0):
    num_t = len(risk_returns)
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
    expected_x_vars = {i: opt_model.addVar(lb=-float('inf'), ub=float(
        'inf'), vtype=grb.GRB.CONTINUOUS, name="expected_x_{0}".format(i)) for i in set_X}
    # add constraints
    # recursive formula of x
    constraints_x = {t:
                     opt_model.addLConstr(
                         lhs=x_vars[t+1],
                         sense=grb.GRB.EQUAL,
                         rhs=riskless_return *
                         x_vars[t] + mu_vars[t] *
                         (risk_returns[t]-riskless_return),
                         name="constraint_{0}".format(t))
                     for t in set_T}
    # recursive formula of expected x
    constraints_expected_x = {t:
                              opt_model.addLConstr(
                                  lhs=expected_x_vars[t+1],
                                  sense=grb.GRB.EQUAL,
                                  rhs=riskless_return *
                                  expected_x_vars[t] + mu_vars[t] *
                                  (expected_risk_return - riskless_return),
                                  name="constraint_{0}".format(t))
                              for t in set_T}
    # initial x
    constraints_x0 = {
        opt_model.addLConstr(
            lhs=x_vars[0],
            sense=grb.GRB.EQUAL,
            rhs=x_0,
            name="constraint_x0")
    }
    # initial expected x
    constraints_expected_x0 = {
        opt_model.addLConstr(
            lhs=expected_x_vars[0],
            sense=grb.GRB.EQUAL,
            rhs=x_0,
            name="constraint_x0")
    }
    # no-shorting constraint of riskless asset
    constraints_mu = {t:
                      opt_model.addLConstr(
                          lhs=mu_vars[t],
                          sense=grb.GRB.LESS_EQUAL,
                          rhs=x_vars[t],
                          name="constraint_mu_{0}".format(t))
                      for t in set_T
                      }
    # no-shorting constraint of risk asset
    constraints_mu_0 = {t:
                        opt_model.addLConstr(
                            lhs=x_vars[t] - mu_vars[t],
                            sense=grb.GRB.LESS_EQUAL,
                            rhs=x_vars[t],
                            name="constraint_mu_0")
                        for t in set_T
                        }
    # constraint of variance of x_T
    constraints_variance = {
        opt_model.addConstr(
            lhs=(x_vars[set_X[-1]] - expected_x_vars[set_X[-1]])**2,
            sense=grb.GRB.LESS_EQUAL,
            rhs=max_variance,
            name="constraint_max_variance"
        )
    }

    # add objective function
    objective = -x_vars[num_t]
    opt_model.ModelSense = grb.GRB.MINIMIZE
    opt_model.setObjective(objective)
    # display the model
    # print(opt_model.display())
    opt_model.optimize()
    # print([[mu_vars[i].x for i in set_T], [x_vars[i].x for i in set_X]])
    # print([expected_x_vars[i].x for i in set_X])
    [mu_vars, x_vars, expected_x_vars] = [[mu_vars[i].x for i in set_T], [
        x_vars[i].x for i in set_X], [expected_x_vars[i].x for i in set_X]]
    # if (x_vars[set_X[-1]] - expected_x_vars[set_X[-1]])**2 > max_variance:
    #     print((x_vars[set_X[-1]] - expected_x_vars[set_X[-1]])**2)
    return [mu_vars, x_vars]


def solve_lagrange_sub_problem(riskless_return, risk_returns, expected_risk_return, x_0, w_s, r, last_policy):
    num_t = len(risk_returns)
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
    expected_x_vars = {i: opt_model.addVar(lb=-float('inf'), ub=float(
        'inf'), vtype=grb.GRB.CONTINUOUS, name="expected_x_{0}".format(i)) for i in set_X}

    # add constraints
    # recursive formula of x
    constraints_x = {t: opt_model.addLConstr(
        lhs=x_vars[t+1],
        sense=grb.GRB.EQUAL,
        rhs=riskless_return *
        x_vars[t] + mu_vars[t] *
        (risk_returns[t]-riskless_return),
        name="constraint_{0}".format(t))
        for t in set_T}
    # recursive formula of expected x
    constraints_expected_x = {t: opt_model.addLConstr(
        lhs=expected_x_vars[t+1],
        sense=grb.GRB.EQUAL,
        rhs=riskless_return *
        expected_x_vars[t] + mu_vars[t] *
        (expected_risk_return - riskless_return),
        name="constraint_{0}".format(t))
        for t in set_T}
    # initial x
    constraints_x0 = {
        opt_model.addLConstr(
            lhs=x_vars[0],
            sense=grb.GRB.EQUAL,
            rhs=x_0,
            name="constraint_x0")
    }
    # initial expected x
    constraints_expected_x0 = {
        opt_model.addLConstr(
            lhs=expected_x_vars[0],
            sense=grb.GRB.EQUAL,
            rhs=x_0,
            name="constraint_x0")
    }
    # no-shorting constraint of riskless asset
    constraints_mu = {t:
                      opt_model.addLConstr(
                          lhs=mu_vars[t],
                          sense=grb.GRB.LESS_EQUAL,
                          rhs=x_vars[t],
                          name="constraint_mu_{0}".format(t))
                      for t in set_T
                      }
    # no-shorting constraint of risk asset
    constraints_mu_0 = {t:
                        opt_model.addLConstr(
                            lhs=x_vars[t] - mu_vars[t],
                            sense=grb.GRB.LESS_EQUAL,
                            rhs=x_vars[t],
                            name="constraint_mu_0")
                        for t in set_T
                        }
    # constraint of variance of x_T
    constraints_variance = {
        opt_model.addConstr(
            lhs=(x_vars[set_X[-1]] - expected_x_vars[set_X[-1]])**2,
            sense=grb.GRB.LESS_EQUAL,
            rhs=max_variance,
            name="constraint_max_variance"
        )
    }

    # add objective function
    objective = -x_vars[num_t] + grb.quicksum(w_s[t]*mu_vars[t] for t in set_T) + \
        r/2 * grb.quicksum((mu_vars[t]-last_policy[t])**2 for t in set_T)
    opt_model.ModelSense = grb.GRB.MINIMIZE

    # Objective Q not PSD (negative diagonal entry). Set NonConvex parameter to 2 to solve model.
    opt_model.setParam("NonConvex", 2)

    opt_model.setObjective(objective)
    # display the model
    # print(opt_model.display())
    opt_model.optimize()
    # print([[mu_vars[i].x for i in set_T], [x_vars[i].x for i in set_X]])
    [mu_vars, x_vars, expected_x_vars] = [[mu_vars[i].x for i in set_T], [
        x_vars[i].x for i in set_X], [expected_x_vars[i].x for i in set_X]]
    # if (x_vars[set_X[-1]] - expected_x_vars[set_X[-1]])**2 > max_variance:
    #     print((x_vars[set_X[-1]] - expected_x_vars[set_X[-1]])**2 )
    return [mu_vars, x_vars]


def ProgressiveHedging(x_t, future_t, verbose=0):
    """
    x_t: the current value of x at time t.
    future_t: the remaining periods at time t.
    """
    # generate all scenarios
    scenarios_list = []
    for scenario in itertools.product(risk_return_elements, repeat=future_t):
        scenarios_list.append(scenario)
    # initialize information price W
    w = np.zeros((len(scenarios_list), future_t))
    opt_sol_for_each_scenario = []
    policy_list = []
    # get initial solutions
    for scenario in scenarios_list:
        opt_sol, opt_state = solve_sub_problem(
            risk_returns=scenario, riskless_return=riskless_return, x_0=x_t, expected_risk_return=expected_risk_return)
        opt_sol_for_each_scenario.append(opt_sol)
    opt_sol_for_each_scenario = np.array(opt_sol_for_each_scenario)
    implementable_policy = opt_sol_for_each_scenario.mean(axis=0)
    policy_list.append(implementable_policy)
    w = r * (opt_sol_for_each_scenario - implementable_policy)
    # iteration
    for k in range(max_iter_num):
        opt_sol_for_each_scenario = []
        for index, scenario in enumerate(scenarios_list):
            opt_sol, opt_state = solve_lagrange_sub_problem(risk_returns=scenario, riskless_return=riskless_return,
                                                            expected_risk_return=expected_risk_return, x_0=x_t, w_s=w[index], r=r, last_policy=implementable_policy)
            opt_sol_for_each_scenario.append(opt_sol)
        opt_sol_for_each_scenario = np.array(opt_sol_for_each_scenario)
        # early stop
        if np.linalg.norm(implementable_policy - opt_sol_for_each_scenario.mean(axis=0)) < tol:
            break
        implementable_policy = opt_sol_for_each_scenario.mean(axis=0)
        policy_list.append(implementable_policy)
        # update w
        w = w + r * (opt_sol_for_each_scenario - implementable_policy)
    if k >= max_iter_num - 1:
        print("reached max iteration numbers.")
    if verbose == 1:
        print("The PHA optimal solution with x_t =", round(x_t, 4), "and future_t=",
              future_t, "is:", implementable_policy, "The iteration stopped at iteration", k)
    return implementable_policy, policy_list

def get_closed_loop_solution(x_0, num_t):
    """ get closed_loop solution for each scenario.
    
    Returns:
        implemented_action : optimal policy of each scenario.
        x_dict : optimal value of x of each scenario.
    """
    x_dict = {}
    action_history = {}
    for t in range(0, num_t):
        if t == 0:
            x_dict[t] = [x_0]
            implemented_action = {}
            # calculate the optimal policy by PHA
            implementable_policy, policy_list = ProgressiveHedging(
                x_0, num_t, verbose=1)
            implemented_action[t] = implementable_policy[0]
            action_history[t] = np.array([x[0] for x in policy_list])
            x_dict[t+1] = []
            for risk_return in risk_return_elements:
                x_dict[t+1].append((x_0-implementable_policy[0]) *
                                   riskless_return + implementable_policy[0]*risk_return)
        else:
            implemented_action[t] = []
            x_dict[t+1] = []
            action_history[t] = []
            for x_t in x_dict[t]:
                # calculate the optimal policy by PHA
                implementable_policy, policy_list = ProgressiveHedging(
                    x_t, num_t-t, verbose=1)
                implemented_action[t].append(implementable_policy[0])
                action_history[t].append(np.array([x[0] for x in policy_list]))
                for risk_return in risk_return_elements:
                    x_dict[t+1].append((x_t-implementable_policy[0]) *
                                       riskless_return + implementable_policy[0]*risk_return)
    return implemented_action, x_dict, action_history


def simulate_closed_loop(riskless_return, risk_return_elements, expected_risk_return, num_t, x_0):
    simulation_count = 1000
    x_T = []
    for i in tqdm(range(simulation_count)):
        x_t = x_0
        x_t_list = [x_t]
        for t in range(num_t):
            implementable_policy, policy_list = ProgressiveHedging(
                x_t=x_t, future_t=num_t-t)
            implemented_action = implementable_policy[0]
            random = np.random.randint(len(risk_return_elements))
            risk_return = risk_return_elements[random]
            x_t = riskless_return*x_t + \
                (risk_return-riskless_return) * implemented_action
            x_t_list.append(x_t)
        x_T.append(x_t)
        mean_x_T = np.average(x_T)
        variance_x_T = np.var(x_T)
    return mean_x_T, variance_x_T


def simulate_open_loop(riskless_return, risk_return_elements, expected_risk_return, num_t, x_0):
    simulation_count = 1000
    x_T = []
    for i in tqdm(range(simulation_count)):
        x_t = x_0
        x_t_list = [x_t]
        implementable_policy, policy_list = ProgressiveHedging(
            x_t=x_t, future_t=num_t)
        for t in range(num_t):
            implemented_action = implementable_policy[t]
            random = np.random.randint(len(risk_return_elements))
            risk_return = risk_return_elements[random]
            x_t = riskless_return*x_t + \
                (risk_return-riskless_return) * implemented_action
            x_t_list.append(x_t)
            x_T.append(x_t)
        mean_x_T = np.average(x_T)
        variance_x_T = np.var(x_T)
    return mean_x_T, variance_x_T


# set hyperparameters
mu_lb = -float('inf')  # lower bound of \mu
mu_ub = float('inf')  # upper bound of \mu
risk_return_elements = [1.4, 0.8]
expected_risk_return = np.mean(risk_return_elements)
riskless_return = 1.05
x_0 = 1
num_t = 2  # time periods
r = 1  # penalty factor
max_iter_num = 10000  # max number of iterations
tol = 1e-4  # terminate threshold
max_variance = 0.10

# get closed loop solution via PHA
optimal_policy, optimal_x, action_history = get_closed_loop_solution(
    x_0=x_0, num_t=num_t)

# closed_loop simulation
mean_x_T, variance_x_T = simulate_closed_loop(
    riskless_return, risk_return_elements, expected_risk_return, num_t, x_0)
print("\n mean_x_T", mean_x_T, "variance_x_T", variance_x_T)

# open-loop simulation
mean_x_T, variance_x_T = simulate_open_loop(
    riskless_return, risk_return_elements, expected_risk_return, num_t, x_0)
print("\n mean_x_T", mean_x_T, "variance_x_T", variance_x_T)
