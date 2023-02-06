#!/usr/bin/env python

# import libraries
import numpy as np
import matplotlib.pyplot as plt


class Bandit(object):

    def __init__(self, ActionValuesArray, epsilon, totalsteps):
        # ActionValuesArray - are expectations of rewards for arms
        # epsilon - epsilon probability value for selecting non-greedy actions
        # totalsteps - number of tatal steps used to simulate the solution of the mu

        self.arm_no = np.size(ActionValuesArray)    # the number of elements in the ActionValuesArray
        self.epsilon = epsilon                      #the epsilon factor to be taken into consideration
        self.currentStep = 0                        #initializing current simulation step
        self.totalsteps = totalsteps                #total simulation steps
        self.ActionValuesArray = ActionValuesArray  #array for expectations of rewards for arms
        self.currentReward = 0                      #initializing current rewards

        self.times_each_arm_selected_counter = np.zeros(
            self.arm_no)  # counts how many times a particular arm is being selected
        self.reward_array = np.zeros(self.totalsteps + 1)  # array for storing the current rewards
        self.optimalarray = np.zeros(self.totalsteps + 1)  # array for plotting the %optimal action
        self.AvgArmRewards = np.zeros(self.arm_no)         # array for storing the mean rewards of each arm

    # select actions according to the epsilon-greedy approach
    def e_greedy(self):

        randomdraw = np.random.rand()  # draw a random number between 0 and 1

        # choose a random arm number if the probability is smaller than epsilon or in the initial step
        if (self.currentStep == 0) or (randomdraw <= self.epsilon):
            selectedArmIndex = np.random.choice(self.arm_no)

        # choose the arm with the largest mean reward in the past
        if (randomdraw > self.epsilon):
            selectedArmIndex = np.argmax(self.AvgArmRewards)

        self.currentStep = self.currentStep + 1  # increament step

        self.times_each_arm_selected_counter[selectedArmIndex] = self.times_each_arm_selected_counter[
                                                                     selectedArmIndex] + 1  # increament the particular arm that got selected

        self.currentReward = np.random.normal(self.ActionValuesArray[selectedArmIndex],
                                              1)  # using probability distribution, draw the reward of the selected arm


        self.reward_array[
            self.currentStep] = self.currentReward  # store the current reward at the current step in the reward array

        self.AvgArmRewards[selectedArmIndex] = self.AvgArmRewards[selectedArmIndex] + (
                    1 / (self.times_each_arm_selected_counter[selectedArmIndex])) * (
                                                           self.currentReward - self.AvgArmRewards[
                                                       selectedArmIndex])  # update the estimate of the mean reward for the selected arm (i.e., Q(A))

        self.optimalarray[self.currentStep] = (selectedArmIndex == np.argmax(
            self.ActionValuesArray))  # assigning the highest estimated reward at the current step of the optimal array

    # run the simulation
    def mainloop(self):
        for i in range(self.totalsteps):
            self.e_greedy()
        


# epsilon values
epsilon1 = 0
epsilon2 = 0.1
epsilon3 = 0.01

totalsteps = 1000  # total number of simulation steps

eps1_avg_reward = np.zeros(totalsteps + 1)
eps2_avg_reward = np.zeros(totalsteps + 1)
eps3_avg_reward = np.zeros(totalsteps + 1)
eps2_optimal = np.zeros(totalsteps + 1)
eps3_optimal = np.zeros(totalsteps + 1)
eps1_optimal = np.zeros(totalsteps + 1)

for _ in range(2000):
    actionValues = np.random.normal(0, 1, 10)  # average of the normal distributions used to generate random rewards

    Bandit1 = Bandit(actionValues, epsilon1, totalsteps)  # creating object of the class
    Bandit1.mainloop()  # calling the main function to run
    eps1_avg_reward += Bandit1.reward_array / 2000.0  # average sum of the rewards to plot


    Bandit2 = Bandit(actionValues, epsilon2, totalsteps)
    Bandit2.mainloop()
    eps2_avg_reward += Bandit2.reward_array / 2000.0

    Bandit3 = Bandit(actionValues, epsilon3, totalsteps)
    Bandit3.mainloop()
    eps3_avg_reward += Bandit3.reward_array / 2000.0

    # find the average of the percentages of the optimal array
    eps1_optimal += Bandit1.optimalarray / 20.0
    eps2_optimal += Bandit2.optimalarray / 20.0
    eps3_optimal += Bandit3.optimalarray / 20.0
print(eps1_avg_reward)
# plot the results
plt.plot(eps1_avg_reward, linewidth=1, color='r', label='epsilon =0')
plt.plot(eps2_avg_reward, linewidth=1, color='k', label='epsilon =0.1')
plt.plot(eps3_avg_reward, linewidth=1, color='m', label='epsilon =0.01')
plt.xlabel('Steps')
plt.ylabel('Average reward')
plt.legend()
plt.savefig('results.png', dpi=300)
plt.show()

plt.plot(eps1_optimal, linewidth=1, color='r', label='epsilon =0')
plt.plot(eps2_optimal, linewidth=1, color='k', label='epsilon =0.1')
plt.plot(eps3_optimal, linewidth=1, color='m', label='epsilon =0.01')
plt.xlabel('Steps')
plt.ylabel('% Optimal Action')
plt.legend()
plt.savefig('results.png', dpi=300)
plt.show()
