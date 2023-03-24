import numpy as np
import matplotlib.pyplot as plt

#plotting for all planning step choices in a single figure
def plotting(no_of_planning_steps, no_of_episodes):    
    plt.figure(figsize=(10,8))
    Steps_vs_episode_Plot = plt.subplot()
    for i,n in enumerate(no_of_planning_steps):
        Steps_vs_episode_Plot.plot(np.arange(no_of_episodes),experiment_results[i,:],label = '{}'.format(n))
    Steps_vs_episode_Plot.set_xlabel('no_of_episodes')
    Steps_vs_episode_Plot.set_ylabel('Steps per episode')
    Steps_vs_episode_Plot.set_ylim((0,800))
    Steps_vs_episode_Plot.set_yticks([14,200,400,600,800])
    Steps_vs_episode_Plot.set_xticks([2,10,20,30,40,50])
    Steps_vs_episode_Plot.legend(title = 'Planning steps')
    plt.show()

# choosing action under epsilon-greedy policy
def select_action(state):
    choice_Qvalues = Q_values[state[0],state[1],:]
    
    #chosing as per epsilon-greedy policy
    e = np.random.random()

    #exploring  if e is less than epsilon
    if e<epsilon:
        action = np.random.choice(action_array)

    #exploiting otherwise, but taking only flatnonzero values
    else:
        action = np.random.choice(np.flatnonzero(choice_Qvalues == np.max(choice_Qvalues)))
    return action

def transition_values(state,action):
    terminated = False
    reward = 0
    new_state = [0, 0]
    new_state[0] = state[0] + action_direction_array[action][0]
    new_state[1] = state[1] + action_direction_array[action][1]
    
    #if goal is reached, update reward and termination status
    if new_state[0] == goal[0] and new_state[1] == goal[1]:
        reward = 1
        terminated = True

    #do not update the state if obstacle is present
    for j in obstacles_array:
        if new_state[0] == j[0] and new_state[1] == j[1]:
            new_state = state
    
    #do not update the state if out of bounds
    if new_state[0] == -1 or new_state[0] == grid_limits[0]+1:
        new_state = state
    elif new_state[1] == -1 or new_state[1] == grid_limits[1]+1:
        new_state = state

    return new_state, reward, terminated

def update_visited_states(state_list, state, action):
    #append state-action pair directly if state_list is empty
    if not state_list:
        state_list.append([state,[action]])
    
    #otherwise, check and see if the list has no repeating elements
    else:
        state_flag = False

        #if state exists in list, but action not exists
        for index, jj in enumerate(state_list):
            if jj[0][0] == state[0] and jj[0][1] == state[1]:
                state_flag = True
                if action not in jj[1]:
                    state_list[index][1].append(action)

        #if state not exists
        if not state_flag:
            state_list.append([state, [action]])

    return state_list

def get_output():
    no_of_experiment_steps = np.zeros(no_of_episodes)
    
    #loop for 30 experiments (each for 50 episodes)
    for experiment in range(no_of_experiments):
        if experiment in np.arange(0, no_of_experiments, no_of_experiments/5):
            print('Running experiment {}'.format(experiment))
        steps_episode_list = np.zeros(no_of_episodes) #no. of steps per episode
        
        global Q_values

        #zeros grid for all possible action values(tensor of size 6*9*4)
        Q_values = np.zeros((6, 9, 4))
        visited_states_list = []
        for i in range(no_of_episodes):
            current_state = np.array([3, 0]) #start position
            terminated = False
            step = 0
            while not terminated:
                action = select_action(current_state) #choose action based on current position on grid

                new_state,reward,terminated = transition_values(current_state, action) #change of state and corresponding reward

                visited_states_list = update_visited_states(visited_states_list, current_state, action) #list of all visited states

                #update Qvalues
                choice_Qvalues = Q_values[new_state[0], new_state[1], :] 
                max_action = np.random.choice(np.flatnonzero(choice_Qvalues == np.max(choice_Qvalues)))
                Q_values[current_state[0], current_state[1], action] += alpha*(reward + gamma * Q_values[new_state[0], new_state[1], max_action] - Q_values[current_state[0], current_state[1], action]) 

                current_state = new_state #update the current position of agent

                for i in range(planning_step_index):
                    random_choice = np.random.choice(np.arange(len(visited_states_list)))
                    random_state = visited_states_list[random_choice][0]
                    random_action = np.random.choice(visited_states_list[random_choice][1])
                    new_random_state, new_random_reward, _ = transition_values(random_state, random_action)
                    
                    #update Qvalues
                    choice_Qvalues = Q_values[new_random_state[0], new_random_state[1], :]
                    max_action = np.random.choice(np.flatnonzero(choice_Qvalues == np.max(choice_Qvalues)))
                    Q_values[random_state[0], random_state[1], random_action] += alpha * (new_random_reward + gamma * Q_values[new_random_state[0], new_random_state[1], max_action] - Q_values[random_state[0], random_state[1], random_action])
                step += 1


            steps_episode_list[i] = step 

        no_of_experiment_steps += steps_episode_list

    no_of_experiment_steps /= no_of_experiments
    # print(step)
    # print(steps_episode_list)
    # print(no_of_experiment_steps)
    return no_of_experiment_steps


if __name__ == "__main__":
    
    # given parameters
    epsilon = 0.1
    alpha = 0.1
    gamma = 0.95
    
    # no_of_planning_steps = [0,5,50]
    no_of_planning_steps = [0,5,50]
    
    # all potential actions from any given cell
    W = [0, 1]
    S = [-1, 0]
    E = [0, -1]
    N = [1, 0]
    
    action_array = np.array([0,1,2,3])
    action_direction_array = [W, S, E, N]


    obstacles_array = [[2,2],[3,2],[4,2],[1,5],[3,7],[4,7],[5,7]]
    goal = [5, 8]
    grid_limits = [5,8]

    no_of_experiments = 30
    no_of_episodes = 50

    experiment_results = np.zeros((3, no_of_episodes))

    for i,planning_step_index in enumerate(no_of_planning_steps):
        print('Running planning steps = {}'.format(planning_step_index))
        experiment_results[i,:] = get_output()

    
    plotting(no_of_planning_steps, no_of_episodes)


