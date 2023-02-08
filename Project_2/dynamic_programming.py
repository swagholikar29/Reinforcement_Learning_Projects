from matplotlib import pyplot as plt
import time
import numpy as np

# load the dat file
def load_file(file_path):
    map = []

    ### How to read dat file in python
    ### Reference: https://stackoverflow.com/questions/29328420/read-specific-column-from-dat-file-in-python
    data = open('grid.dat')
    text = data.readlines()

    for index, line in enumerate(text):
        map.append([])
        for char in line:
            if char == "\n":
                continue
            map[index].append(int(char))                        # generate the map
    return np.array(map)


#How to add arrows in plotting
#Reference: https://pythonforundergradengineers.com/quiver-plot-with-matplotlib-and-jupyter-notebooks.html 

# function to plot the Value function
def plot1(suptitle, title1, Y, X, v, u, mapy, mapx, title2, param, ifModel):
    fig = plt.figure()
    fig.suptitle(suptitle)
    plt.xlabel(ifModel)
    plt.title(title1)
    plt.quiver(Y, X, v, u, color = 'b', scale=8)
    plt.axis([0, mapy-1, 0, mapx-1])
    plt.imshow(map*-1,cmap="bone")
    plt.show()


# function to plot the Control Policy function
def plot2(suptitle, title1, Y, X, v, u, mapy, mapx, title2, param, ifModel):
    fig = plt.figure()
    fig.suptitle(suptitle)
    plt.xlabel(ifModel) 
    plt.title(title2)
    plt.imshow(param,cmap="summer")
    plt.show()


# initialize an array of rewards with the given conditions
# Goal = 100.0
# obstacle = -50.0
# any movement = -1.0
def reward_array(map,goal):
    rewards = np.ones(map.shape)*-1
    for i in range(xmax):
        for j in range(ymax):
            if map[i][j] == 1:
                rewards[i][j]=-50

    rewards[goal[0]-1,goal[1]-1] = 100
    plt.imshow(rewards)
    plt.show()
    return rewards


# returns a list of possible actions for the agent to choose from
def possible_actions(x, y):
    global map, ifModel
    
    ifModel = deterministic                                     # check for the model
    if(index == 2):
        ifModel = stochastic
    xmax,ymax = map.shape

    if abs(map[x,y]) == 1:                                      # check if the cell in the map is an obstacle or not
        return [],0,[]                                          # return no possible action if an obstacle is found
    
    no_of_actions = 0
    action_list = []
    action_counter = []
    k = 0

    # return a set of possible actions if the cell lies within the boundaries of the map
    for ii in ifModel:
        action_var = []                                         # stores the possible actions as per the model
        for ([dir1, dir2], Prob_var) in ii:
            action_var.append(([dir1, dir2], Prob_var))
            if x + dir1 < 0 or x + dir1 >= xmax or y + dir2 < 0 or y + dir2 >= ymax:
                action_var.remove(([dir1, dir2], Prob_var))
                               
        if(len(action_var)>0):
            action_list.append(action_var)
            action_counter.append(k)                            # action counter is used to compare the index of the policy and that of value function. Has been used later in this code.
        k += 1
    no_of_actions = len(action_list)
    return action_list, no_of_actions, action_counter

def value_iteration(no_of_actions, rewards, theta = 1e-3, gamma = 0.95):
    T0 = time.time()                                            # begin time
    global map, plot_array_vi
    mapx, mapy = map.shape

    # initialize a policy matrix and value function
    policy = np.zeros((mapx, mapy, no_of_actions))
    value_function = np.zeros((mapx, mapy))
    delta = theta
    xmax, ymax = value_function.shape

    while delta >= theta:
        optimal_actions = np.zeros((mapx, mapy), dtype=np.uint8)
        delta = 0
        for x in range(xmax):
            for y in range(ymax):
                v = 0
                actions,no_of_actions, action_counter = possible_actions(x, y)              # function call
                if no_of_actions > 0:
                    for a in range(no_of_actions):
                        new_V = 0

                        # check for every action from the possible actions list
                        for ([dir1, dir2], Prob_var) in actions[a]:
                            # Bellman equation
                            new_V +=  Prob_var*(rewards[x + dir1, y + dir2] + gamma * value_function[x + dir1, y + dir2])    

                        # update the value function to the maximum
                        if(new_V > v):
                            v = new_V
                            optimal_actions[x,y] = action_counter[a]
                    
                    delta = max(delta, abs(v - value_function[x, y]))
                    value_function[x, y] = v

    # caluclate the total time
    T_final = round((time.time() - T0), 3)
    print(f"Value Iteration Convergence Time = {T_final} secs")


    # Initialize the plotting environment
    x_axis = np.arange(0,mapx,1).astype(float)
    y_axis = np.arange(0,mapy,1).astype(float)
    X, Y = np.meshgrid(x_axis,y_axis)
    u = X.copy()
    v = Y.copy()

    # traversing the mesh
    for ix in range(X.shape[0]):
        for iy in range(X.shape[1]):
            i = int(X[ix,iy])
            j = int(Y[ix,iy])
            
            # detecting obstacle
            if map[i][j]==1:
                u[ix,iy]=0
                v[ix,iy]=0
                continue

            u[ix,iy] = deterministic[optimal_actions[i,j]][0][0][0]*0.1
            v[ix,iy] = deterministic[optimal_actions[i,j]][0][0][1]*0.1

    for x in range(xmax):
        for y in range(ymax):
            policy[x,y,max(0,optimal_actions[x,y]-1)]=1


    # storing parameters in array, later used for plotting
    plot_array_vi = ["Value Iteration", "Control Policy", Y, X, v, u, mapy, mapx, "Value Function", value_function]

    return policy, value_function


def policy_evaluation(policy,value_function,rewards,theta = 1e-3,gamma=0.95):
    delta = theta
    xmax,ymax = value_function.shape
    while delta>=theta:
        delta = 0
        for x in range(xmax):
            for y in range(ymax):
                v = 0
                actions,no_of_actions,_ = possible_actions(x,y)
                if no_of_actions>0:

                    # iterating through all actions
                    for a in range(no_of_actions):
                        new_V = 0
                        
                        # check for every action from the possible actions list
                        for ([dir1, dir2], Prob_var) in actions[a]:
                            # Bellman equation
                            new_V +=  Prob_var*(rewards[x + dir1, y + dir2] + gamma * value_function[x + dir1, y + dir2])
                        
                        v += policy[x, y, a]*new_V

                    # update the value function to the maximum
                    delta = max(delta, abs(v - value_function[x, y]))
                    value_function[x, y] = v
        
        # in case of a negative threshold
        if(theta < 0):
            break
    return value_function


def policy_improvement(policy, value_function,rewards,gamma=0.95):
    xmax,ymax = value_function.shape
    policy_stable = True

    # traversing through the value function
    for x in range(xmax):
        for y in range(ymax):
            act = 0
            act_value = -1 * np.inf
            actions, no_of_actions, action_counter = possible_actions(x,y)
            if no_of_actions > 0:

                # iterating through all actions
                for a in range(no_of_actions):
                    new_V = 0

                    # check for every action from the possible actions list
                    for ([dir1, dir2], Prob_var) in actions[a]:

                        # Bellman Equation
                        new_V +=  Prob_var * (rewards[x + dir1, y + dir2] + gamma * value_function[x + dir1, y + dir2])
                    
                    # update the value function to the maximum
                    if new_V > act_value:
                        act_value = new_V
                        act = action_counter[a]  

                previous_action = np.argmax(policy[x, y, :])
                if(previous_action != act):
                    policy_stable = False
                policy[x, y,:] = policy[x, y,:]*0
                policy[x, y, act] = 1
                value_function[x, y] = act_value
    return policy, value_function, policy_stable


def policy_iteration(no_of_actions,rewards,theta = 1e-3,gamma=0.95):
    #starting the timer
    T0 = time.time()
    global map, plot_array_pi
    mapx,mapy = map.shape

    #initializing control policy and value functions
    policy = np.ones((mapx,mapy,no_of_actions))/no_of_actions
    value_function = np.zeros((mapx,mapy))
     
    policy_stable = False
    while not policy_stable:
        value_function = policy_evaluation(policy,value_function,rewards,theta,gamma)                    #calling function
        policy, opt_value ,policy_stable = policy_improvement(policy,value_function,rewards,gamma)       #calling function
    
    #calculating total time
    T_final = round((time.time() - T0), 3)                                                               
    print(f"Policy Iteration Convergence Time = {T_final} secs")

    #setting up the plotting environment
    x_axis = np.arange(0, mapx, 1).astype(float)
    y_axis = np.arange(0, mapy, 1).astype(float)
    X, Y = np.meshgrid(x_axis, y_axis)
    u = X.copy()
    v = Y.copy()

    #iterating and updating action counter
    for ix in range(X.shape[0]):
        for iy in range(X.shape[1]):
            i = int(X[ix, iy])
            j = int(Y[ix, iy])
            if map[i][j]==1:
                u[ix, iy]=0
                v[ix, iy]=0
                continue
            action_counter = np.argmax(policy[i, j, :])
            u[ix, iy] = deterministic[action_counter][0][0][0] * 0.1
            v[ix, iy] = deterministic[action_counter][0][0][1] * 0.1

    # storing parameters in array, later used for plotting
    plot_array_pi = ["Policy Iteration", "Control Policy", Y, X, v, u, mapy, mapx, "Value Function", opt_value]
    

    return policy, opt_value


def GPI(no_of_actions,rewards,theta = 0.001,gamma=0.95):
    #starting with the timer
    T0 = time.time()
    global map, plot_array_gpi
    mapx,mapy = map.shape

     #initializing control policy and value functions
    policy = np.ones((mapx,mapy,no_of_actions))/no_of_actions
    value_function = np.zeros((mapx, mapy))
    policy_stable = False
    while not policy_stable:
        value_function = policy_evaluation(policy, value_function, rewards, -1, gamma)                      #calling function
        policy, opt_value ,policy_stable = policy_improvement(policy, value_function, rewards, gamma)       #calling function
    
    #calculating total time
    T_final = round((time.time() - T0), 3)
    
    print(f"General Policy Iteration Convergence Time = {T_final} secs")

    #setting up the plotting environment
    x_axis = np.arange(0, mapx, 1).astype(float)
    y_axis = np.arange(0, mapy, 1).astype(float)
    X, Y = np.meshgrid(x_axis, y_axis)
    u = X.copy()
    v = Y.copy()

    #iterating and updating action counter
    for ix in range(X.shape[0]):
        for iy in range(X.shape[1]):
            i = int(X[ix, iy])
            j = int(Y[ix, iy])
            if map[i][j]==1:
                u[ix, iy]=0
                v[ix, iy]=0
                continue
            action_counter = np.argmax(policy[i, j, :])
            u[ix, iy] = deterministic[action_counter][0][0][0] * 0.1
            v[ix, iy] = deterministic[action_counter][0][0][1] * 0.1


    # storing parameters in array, later used for plotting
    plot_array_gpi = ["General Policy Iteration", "Control Policy", Y, X, v, u, mapy, mapx, "Value Function", opt_value]

    return policy, opt_value



if __name__ == "__main__":

    # all potential actions from any given cell
    N = [-1,0]
    S = [1,0]
    E = [0,1]
    W = [0,-1]
    NE = [-1,1]
    NW = [-1,-1]
    SE = [1,1]
    SW = [1,-1]

    index = 1
    #probability is one for all actions in deterministic model
    deterministic = [[(N,1)],[(NE,1)],[(E,1)],[(SE,1)],[(S,1)],[(SW,1)],[(W,1)],[(NW,1)]]

    #probability is 0.8 for all taken actions in stochastic model and it goes to near diagonals for 0.2 probability
    stochastic = [[(NW,0.1),(N,0.8),(NE,0.1)], [(N,0.1),(NE,0.8),(E,0.1)], [(NE,0.1),(E,0.8),(SE,0.1)], [(E,0.1),(SE,0.8),(S,0.1)],
    [(SE,0.1),(S,0.8),(SW,0.1)], [(S,0.1),(SW,0.8),(W,0.1)], [(SW,0.1),(W,0.8),(NW,0.1)], [(W,0.1),(NW,0.8),(N,0.1)]]

    #loading the map of dat file
    map = load_file('grid.dat')

    xmax = len(map)
    ymax = len(map[0])
    #print(xmax, ymax) - (15,51)

    goal = [8,11]
    rewards = reward_array(map,goal)

    print("For Deterministic Model:")
    no_of_actions = 8
    policy,value = policy_iteration(no_of_actions,rewards)
    policy,value = value_iteration(no_of_actions,rewards)
    policy,value = GPI(no_of_actions,rewards)


    plot1(plot_array_pi[0], plot_array_pi[1], plot_array_pi[2], plot_array_pi[3], plot_array_pi[4], plot_array_pi[5], plot_array_pi[6], plot_array_pi[7], plot_array_pi[8], plot_array_pi[9], "Deterministic")
    plot2(plot_array_pi[0], plot_array_pi[1], plot_array_pi[2], plot_array_pi[3], plot_array_pi[4], plot_array_pi[5], plot_array_pi[6], plot_array_pi[7], plot_array_pi[8], plot_array_pi[9], "Deterministic")

    plot1(plot_array_vi[0], plot_array_vi[1], plot_array_vi[2], plot_array_vi[3], plot_array_vi[4], plot_array_vi[5], plot_array_vi[6], plot_array_vi[7], plot_array_vi[8], plot_array_vi[9], "Deterministic")
    plot2(plot_array_vi[0], plot_array_vi[1], plot_array_vi[2], plot_array_vi[3], plot_array_vi[4], plot_array_vi[5], plot_array_vi[6], plot_array_vi[7], plot_array_vi[8], plot_array_vi[9], "Deterministic")

    plot1(plot_array_gpi[0], plot_array_gpi[1], plot_array_gpi[2], plot_array_gpi[3], plot_array_gpi[4], plot_array_gpi[5], plot_array_gpi[6], plot_array_gpi[7], plot_array_gpi[8], plot_array_gpi[9], "Deterministic")
    plot2(plot_array_gpi[0], plot_array_gpi[1], plot_array_gpi[2], plot_array_gpi[3], plot_array_gpi[4], plot_array_gpi[5], plot_array_gpi[6], plot_array_gpi[7], plot_array_gpi[8], plot_array_gpi[9], "Deterministic")

    

    print("For Stochastic Model:")
    index = 2
    policy,value = policy_iteration(no_of_actions,rewards)
    policy,value = value_iteration(no_of_actions,rewards)
    policy,value = GPI(no_of_actions,rewards)


    plot1(plot_array_pi[0], plot_array_pi[1], plot_array_pi[2], plot_array_pi[3], plot_array_pi[4], plot_array_pi[5], plot_array_pi[6], plot_array_pi[7], plot_array_pi[8], plot_array_pi[9], "Stochastic")
    plot2(plot_array_pi[0], plot_array_pi[1], plot_array_pi[2], plot_array_pi[3], plot_array_pi[4], plot_array_pi[5], plot_array_pi[6], plot_array_pi[7], plot_array_pi[8], plot_array_pi[9], "Stochastic")

    plot1(plot_array_vi[0], plot_array_vi[1], plot_array_vi[2], plot_array_vi[3], plot_array_vi[4], plot_array_vi[5], plot_array_vi[6], plot_array_vi[7], plot_array_vi[8], plot_array_vi[9], "Stochastic")
    plot2(plot_array_vi[0], plot_array_vi[1], plot_array_vi[2], plot_array_vi[3], plot_array_vi[4], plot_array_vi[5], plot_array_vi[6], plot_array_vi[7], plot_array_vi[8], plot_array_vi[9], "Stochastic")

    plot1(plot_array_gpi[0], plot_array_gpi[1], plot_array_gpi[2], plot_array_gpi[3], plot_array_gpi[4], plot_array_gpi[5], plot_array_gpi[6], plot_array_gpi[7], plot_array_gpi[8], plot_array_gpi[9], "Stochastic")
    plot2(plot_array_gpi[0], plot_array_gpi[1], plot_array_gpi[2], plot_array_gpi[3], plot_array_gpi[4], plot_array_gpi[5], plot_array_gpi[6], plot_array_gpi[7], plot_array_gpi[8], plot_array_gpi[9], "Stochastic")
