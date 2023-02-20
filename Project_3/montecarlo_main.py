import numpy as np
import matplotlib.pyplot as plt
from random import randint, choice, random
import random

# initialization
Policy_Pi = [[0, 'T'], [1, 'L'], [2, 'L'], [3, 'L'], [4, 'R'], [5, 'T']]
S_A = [[0, 'T'], [1, 'L'], [1, 'R'], [2, 'L'], [2, 'R'], [3, 'L'], [3, 'R'], [4, 'L'], [4, 'R'], [5, 'T']]
Q_s = {(0, 'T'): 0, (1, 'L'): 0, (1, 'R'): 0, (2, 'L'): 0, (2, 'R'): 0, (3, 'L'): 0, (3, 'R'): 0, (4, 'L'): 0, (4, 'R'): 0, (5, 'T'): 0}
Returns = {(0, 'T'): [], (1, 'L'): [], (1, 'R'): [], (2, 'L'): [], (2, 'R'): [], (3, 'L'): [], (3, 'R'): [], (4, 'L'): [], (4, 'R'): [], (5, 'T'): []}

Episode = []
State = []
New_state = []
Reward = 0
threshold = 100             # used for preventing looping
No_ep = 500                 # number of episodes

# returns the optimal policy for a given state-action pair in the first-visit method
def firstvisit_func(S, epsilon = 0.05):

    if S[0] == 0 or S[0] == 5:
        return 'T'

    min_val = random.random() # 0.3
    # min_val = 0.3
    right = (S[0], 'R')
    left = (S[0], 'L')

    if Q_s[left] > Q_s[right]:
        direction =  -1
    elif Q_s[left] < Q_s[right]:
        direction = 1
    else:
        direction = choice([-1, 1])

    counter = 0
    for ii in S_A:
        if ii[0] == S[0]:
            counter += 1
    
    eqn_var = 1 - epsilon + (epsilon/counter)

    if eqn_var > min_val:
        return direction

    else:
        return -1 * direction

# returns the optimal policy for a given state-action pair in the exploring starts method
def exploringstart_func(S):

    if S[0] == 0 or S[0] == 5:
        return 'T'
    
    right = (S[0], 'R')
    left = (S[0], 'L')
  
    if Q_s[left] > Q_s[right]:
        return 'L'
    elif Q_s[left] < Q_s[right]:
        return 'R'
    else:
        return choice(['L', 'R'])


# main loop
def run_this_function(question_number):
    Vs_matrix = np.zeros((6,No_ep))             # 2-D matrix that stores the state-value for all states in all episodes.

    for i in range(No_ep):
        # print(i) 
        S = 0
        G = 0
        Gamma = 0.95
        th = 0
        First_Action = random.choice(S_A)       # make a random first action from all possible state-action pairs
        Episode = []

        # loop until the terminal state is reached
        while S != 'T':
            Episode.append(First_Action)
            for S in Episode:

                # fetch the appropriate policy depending on the question
                if question_number == 'firstvisit':
                    dir_no = firstvisit_func(S)
                    if dir_no == 'T':
                        S = 'T'
                        break
                    elif dir_no == -1:
                        S[1] = 'L'
                    else:
                        S[1] = 'R'
                
                elif question_number == 'exploringstart':
                    S[1] = exploringstart_func(S)
                    if S[1] == 'T':
                        S = 'T'
                        break
                if S == 'T':
                    break  

                # make stochastic decisions
                probab = randint(1, 100)

                if S[1] == 'L' and probab <= 80:
                    State = S[0] - 1

                elif S[1] == 'L' and probab > 95:
                    State = S[0] + 1

                elif S[1] == 'L' and probab > 80 and probab <= 95:
                    State = S[0]

                elif S[1] == 'R' and probab <= 80:
                    State = S[0] + 1

                elif S[1] == 'R' and probab > 95:
                    State = S[0] - 1

                elif S[1] == 'R' and probab > 80 and probab <= 95:
                    State = S[0]

                else:
                    S = 'T'
                    break    
            
                for p in Policy_Pi:
                    if p[0] == State:
                        New_state_Action_Pair = p
                        Episode.append(New_state_Action_Pair)
                        th += 1
                        break
                    else:
                        continue
                if th == threshold:
                    S = 'T'
                    break
            if th == threshold:
                break
            else:
                G_next = 0
                for St_Action in reversed(Episode):
                    if St_Action[0] == 0:
                        Reward = 1
                    elif St_Action[0] == 5:
                        Reward = 5
                    else:
                        Reward = 0
                    G = Reward + (Gamma * G_next)           # calculate the total rewards
                    G_next = G
                    for update in Returns:
                        if update == tuple(St_Action):
                            Returns[update].append(G)
                            break
                        else:
                            continue

                for Q in Q_s:
                    for r in Returns:
                        if Q == r:
                            if len(Returns[r]) == 0:
                                Q_s[Q] = sum(Returns[r])
                            else:
                                Q_s[Q] = sum(Returns[r]) / len(Returns[r])
                            break    
                        else:
                            continue
            # print(Q_s)
            for j in range(6):
                L = []
                M = []
                n = 0
                for q in Q_s:
                    if q[0] == j:
                        #L[n] = {q: Q_s[q]}
                        #M[n] = Q_s[q]
                        L.append({q: Q_s[q]})
                        M.append(Q_s[q])
                        n += 1
                    else:
                        continue
                # find the maximum of the action-value for a given state
                Max = max(M)
                Vs_matrix[j][i] = Max

                St = 0
                t = []
                for l in L:
                    T = l.items()
                    for key, value in T:
                        t.append([key, value])
                for key, value in t:
                    if value == Max:
                        St = key
                for p in Policy_Pi:
                    if p[0] == j:
                        no = Policy_Pi.index(p)

                        Policy_Pi[no] = list(St)
                        break
                    else:
                        continue

    # prints outputs
    print(" ")
    print("optimal action value function: ")
    print(Q_s)
    print(" ")
    print("optimal policy: ")
    print(Policy_Pi)
    print(" ")

    plot_state_value(Vs_matrix, question_number)

def plot_state_value(Vs_matrix, question_number):
    # plot the results
    fig = plt.figure()
    fig.suptitle(question_number)
    plt.plot(Vs_matrix[1][:], linewidth=2, color='r', label='state 1')
    plt.plot(Vs_matrix[2][:], linewidth=2, color='k', label='state 2')
    plt.plot(Vs_matrix[3][:], linewidth=2, color='m', label='state 3')
    plt.plot(Vs_matrix[4][:], linewidth=2, color='y', label='state 4')
    plt.xlabel('Number of Episodes')
    plt.ylabel('State value function')
    plt.legend()
    # plt.savefig('results.png', dpi=300)
    plt.show()


if __name__ == "__main__":

    print("Exploring Starts")
    Policy_Pi = [[0, 'T'], [1, 'L'], [2, 'L'], [3, 'L'], [4, 'R'], [5, 'T']]
    S_A = [[0, 'T'], [1, 'L'], [1, 'R'], [2, 'L'], [2, 'R'], [3, 'L'], [3, 'R'], [4, 'L'], [4, 'R'], [5, 'T']]
    Q_s = {(0, 'T'): 0, (1, 'L'): 0, (1, 'R'): 0, (2, 'L'): 0, (2, 'R'): 0, (3, 'L'): 0, (3, 'R'): 0, (4, 'L'): 0, (4, 'R'): 0, (5, 'T'): 0}
    Returns = {(0, 'T'): [], (1, 'L'): [], (1, 'R'): [], (2, 'L'): [], (2, 'R'): [], (3, 'L'): [], (3, 'R'): [], (4, 'L'): [], (4, 'R'): [], (5, 'T'): []}

    # Question 1
    question_number = 'exploringstart'
    run_this_function(question_number)

    Episode = []
    State = []
    New_state = []
    Reward = 0
    threshold = 100
    print("First Visit")

    # Question 2
    Policy_Pi = [[0, 'T'], [1, 'L'], [2, 'L'], [3, 'L'], [4, 'R'], [5, 'T']]
    S_A = [[0, 'T'], [1, 'L'], [1, 'R'], [2, 'L'], [2, 'R'], [3, 'L'], [3, 'R'], [4, 'L'], [4, 'R'], [5, 'T']]
    Q_s = {(0, 'T'): 0, (1, 'L'): 0, (1, 'R'): 0, (2, 'L'): 0, (2, 'R'): 0, (3, 'L'): 0, (3, 'R'): 0, (4, 'L'): 0, (4, 'R'): 0, (5, 'T'): 0}
    Returns = {(0, 'T'): [], (1, 'L'): [], (1, 'R'): [], (2, 'L'): [], (2, 'R'): [], (3, 'L'): [], (3, 'R'): [], (4, 'L'): [], (4, 'R'): [], (5, 'T'): []}

    question_number = 'firstvisit'
    run_this_function(question_number)