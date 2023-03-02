import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt


# load the dat file
def load_file(file_path):
    map = []
    data = open('grid.dat')
    text = data.readlines()

    for index, line in enumerate(text):
        map.append([])
        for char in line:
            if char == "\n":
                continue
            map[index].append(int(char))   

    return np.array(map)

    
#current state of the robot
def return_current_pos(map):
    global done_flag
    global x, y
    return tuple((x, y)), done_flag

#next state and reward estimation after each action taken
def state_update(action):
    global list_of_actions
    global x, y
    done_flag = False
    x += list_of_actions[action][0]
    y += list_of_actions[action][1]

    # check for boundaries
    x = max(x, 0)
    x = min(x, xmax - 1)
    y = max(y, 0)
    y = min(y, ymax - 1)


    if 1 <= y <= 10 and x == 0:         # fall into a cliff
        reward = -100
        x = 0
        y = 0
        done_flag = False

    elif x == 0 and y == ymax - 1:      # the final goal
        reward = -1
        done_flag = True

    # any other state other than cliff and goal
    else:   
        reward = -1
    return tuple((x, y)), reward, done_flag

#probability estimation of all possible actions under epsilon-greedy policy
def epsilon_greedy_policy(Q, state, no_of_actions, epsilon):
    EGP = np.ones(no_of_actions, dtype = np.float32) * epsilon * (1 / no_of_actions)
    action = np.argmax(Q[state])
    EGP[action] += 1 - epsilon
    return EGP

#Qlearning policy implementation
def Qlearning(alpha = 0.1, no_of_epi = 500, discount = 0.95, epsilon = 0.1):
    global x, y
    global map, no_of_actions
    
    Q = defaultdict(lambda: np.zeros(no_of_actions))                                #dictionary to store state-action values
    cum_rewards = []                                                                #to store sum of all rewards at the end of each episode
    for i in range(no_of_epi):
        total_reward = 0.0
        done_flag = False
        x = 0
        y = 0
        cur_state, done_flag = return_current_pos(map)
        while done_flag == False:
            EGD_prob = epsilon_greedy_policy(Q, cur_state, no_of_actions, epsilon)
            action = np.random.choice(np.arange(no_of_actions), p=EGD_prob)         # re-select action per round
            next_state, reward, done_flag = state_update(action)                    # apply action to find next state

            # if goal is reached, there is no next state, hence 0.0
            if done_flag:
                Q[cur_state][action] = Q[cur_state][action] + alpha * (reward + discount * 0.0 - Q[cur_state][action])
                break

            # calculate and store Q for exploration. 
            else:
                next_action = np.argmax(Q[next_state]) #argmax of state-action value is taken for predicting the next_action
                Q[cur_state][action] = Q[cur_state][action] + alpha * (reward + discount * Q[next_state][next_action] - Q[cur_state][action])

                cur_state = next_state
            total_reward += reward
        cum_rewards.append(total_reward)                           #sum of all rewards collected at end of each episode
    return Q, cum_rewards



#SARSA policy implementation
def sarsa(map, no_of_epis = 500, discount = 0.95, alpha = 0.5, epsilon = 0.1):
    global x, y

    global no_of_actions
    Q = defaultdict(lambda: np.zeros(no_of_actions))                #dictionary to store state-action values
    cum_rewards = []                                                #to store sum of all rewards at the end of each episode

    for episode in range(no_of_epis):
        total_reward = 0.0
        done_flag = False
        x = 0
        y = 0
        state, done_flag = return_current_pos(map)
        EGD_probs = epsilon_greedy_policy(Q, state, no_of_actions, epsilon)                            # action policy
        action = np.random.choice(np.arange(no_of_actions), p = EGD_probs)                             # action from action policy
        while done_flag == False:
            next_state, reward, done_flag = state_update(action)                                       # new state and reward for every action taken
            if done_flag: #if goal reached
                Q[state][action] = Q[state][action] + alpha * (reward + discount * 0.0 - Q[state][action])
                break
            else:
                EGD_probs = epsilon_greedy_policy(Q, next_state, no_of_actions, epsilon)               # get action probability distribution for next state
                next_action = np.random.choice(np.arange(no_of_actions), p = EGD_probs)                # get next action, use [next_state][next_action]  to update Q[state][action] through epsilon greedy policy
                Q[state][action] = Q[state][action] + alpha * (reward + discount * Q[next_state][next_action] - Q[state][action])
                state = next_state
                action = next_action
            total_reward += reward
        cum_rewards.append(total_reward)            #sum of all rewards collected at end of each episode
    return Q, cum_rewards


#function for plotting the greedy-action-policy with respect to the map
def Policy_display(Q):
    global map
    result = ""
    for i in range(len(map)):
        line = ""
        for j in range(len(map[0])):
            if i == 0 and j >= 1 and j <= 10:
                line += "● "
            else:
                action = np.argmax(Q[(i, j)])               # find the action to max Q value (greedy policy)
                # print((i, j), Q[(i, j)], "Action:", action)
                if action == 0:
                    line += "→ "
                elif action == 1:
                    line += "← "
                elif action == 2:
                    line += "↓ "
                else:
                    line += "↑ "
        result = line + "\n" + result
    print(result)


if __name__ == "__main__":

    #start state at (0,0)
    x = 0
    y = 0

    # all potential actions from any given cell
    W = [0, 1]
    E = [0, -1]
    S = [-1, 0]
    N = [1, 0]
    no_of_actions = 4
    list_of_actions = [W, E, S, N]

    #loading the map as dat file
    map = load_file('grid.dat')

    #finding map boundary conditions
    xmax = len(map)
    ymax = len(map[0])
    done_flag = False

    epsilon_list=[0.25, 0.1, 0.01, 0.0001]
    
    print_matrix = []
    ctr = 0
    for epsilon in epsilon_list:
        
        #SARSA Policy
        print('SARSA: epsilon='+str(epsilon))

        Q_s, rewards_s = sarsa(map, no_of_epis=500, epsilon = epsilon)
        if ctr == 0:
            print_matrix = rewards_s[:]
        ctr += 1

        plt.plot(range(500), rewards_s, label ='SARSA: epsilon ='+str(epsilon))
        Policy_display(Q_s)

    #Plotting the rewards as a function of episodes
    plt.xlabel('No. of episodes')
    plt.ylabel('Cumulative Rewards')
    plt.ylim(-500, 0)
    plt.legend()
    plt.show()


    ctr = 0
    print_matrixQ = []
    for epsilon in epsilon_list:
        
        #Qlearning Policy
        print('Qlearning: epsilon='+str(epsilon))

        Q, rewards = Qlearning(no_of_epi=500, epsilon = epsilon)
        if ctr == 0:
            print_matrixQ = rewards[:]
        ctr += 1
        
        plt.plot(range(500), rewards, label ='Qlearning: epsilon ='+str(epsilon))
        Policy_display(Q)

    #Plotting the rewards as a function of episodes
    plt.xlabel('No. of episodes')
    plt.ylabel('Cumulative Rewards')
    plt.ylim(-500, 0)
    plt.legend()
    plt.show()



    # graphically comparing the rewards of Qlearning and SARSA for a particular epsilon as a function of episodes
    plt.plot(range(500), print_matrixQ, label ='Qlearning: epsilon=0.25')
    plt.plot(range(500), print_matrix, label ='SARSA: epsilon=0.25')
    plt.xlabel('No. of episodes')
    plt.ylabel('Cumulative Rewards')
    plt.ylim(-500, 0)
    plt.legend()
    plt.show()
