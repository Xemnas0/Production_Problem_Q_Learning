# coding: utf-8

# # Monte Carlo method

# Inspired by https://github.com/rlcode/reinforcement-learning/blob/master/1-grid-world/5-q-learning/q_learning_agent.py

# ## Importing modules

# In[ ]:


import time
import sys
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Disable scientific notation when printing
np.set_printoptions(suppress=True)

# ## Parameters definition

# In[2]:


# Time interval in hours
delta_T = 8

# Total time in days
tot_T = 50

# Number of states = totTime * (hours_in_a_day / interval_in_hours)
n_states = int(tot_T * 24 / delta_T)

# Episodes
n_episodes = 10000

# Discount factor (gamma)
discount_factor = 0.99

# Learning rate (alpha)
learning_rate = 0.5

# Epsilon greedy
epsilon = 0.5

# Initial balance
start_balance = 60

# Cost and production rate of each factory
# costs = np.array([29, 299, 399, 499, 1999, 4999, 12999])
# prod_rates = np.array([2, 22, 35, 50, 250, 750, 2500])*0.07
costs = np.array([29, 299, 399])
prod_rates = np.array([3, 22, 35]) * 0.05
n_factories = len(costs)

# Approximative max balance to spend at each state
# TODO: use last factory's cost as reference
final_prod_approx = 2000

# Actions
actions = []


# ## Generating actions

# In[3]:


def generate_actions(costs, final_prod_approx):
    """
    Generates all the possible states, that is all the purchase combinations
    """

    global actions

    actions = np.array([])

    cost_cheapest_factory = costs[0]
    max_balance = final_prod_approx

    # Number of iterations equal to number purchasable cheapest factory
    # n = int(np.ceil(max_balance / cost_cheapest_factory) + 1)

    # Init return array
    # np.empty((n,9)).astype(int)

    # INFO: action format (with "balance" as additional info that will be discarded in q_table)
    # action[i] = [balance, pass, buy[0], buy[1], ...]

    # Append "pass" action
    actions = np.append(actions, np.array([0] + [1] + [0] * len(costs)))

    # Init with cells for balance, pass_flag, and a cell for each factory
    n_possibles = np.zeros((len(costs) + 2)).astype(int)
    # Calculate n_possibles for each factory
    tmp = np.ceil(max_balance / costs)
    n_possibles[2:] = np.ceil(max_balance / costs)

    curr_action = np.zeros((len(costs) + 2))

    # Multiplication principal (recursive)
    def mult_princR(pos, n_possibles, curr_action, curr_tot_cost):

        global actions
        global costs

        # If we have filled the last cell of the action array, terminate
        if pos >= len(n_possibles):
            curr_action[0] = curr_tot_cost
            # print("Action: "+str(curr_action))
            actions = np.append(actions, curr_action)
            # print(str(actions)+"\n====")
            return

        # From 0 to max n. of that factory (max n. included)
        for i in range(n_possibles[pos] + 1):

            dcost = i * costs[pos - 2]
            if curr_tot_cost + dcost > final_prod_approx:
                break

            curr_action[pos] = i

            curr_tot_cost += dcost

            mult_princR(pos + 1, n_possibles, curr_action, curr_tot_cost)

            curr_tot_cost -= dcost

    mult_princR(2, n_possibles, curr_action, 0)

    actions = np.reshape(actions, (-1, len(costs) + 2)).astype(int)

    actions = actions[actions[:, 0].argsort()]

    # delete vector [0, 0, 0, ..., 0] with no action
    actions = np.delete(actions, 1, 0)

    return actions


# In[4]:


actions = generate_actions(costs, final_prod_approx)
print(actions)
n_actions = actions.shape[0]


# ## Create q_table

# In[5]:


# q_table's size will be (n_time_instants x actions) = (n_states x n_actions)
def init_q_table(n_states, n_actions):
    q_table = np.zeros(shape=(int(n_states), n_actions))
    return q_table


q_table = init_q_table(n_states, n_actions)


# ## Agent's functions

# ### Choose action

# In[6]:


def search_closest_index_below(x, v):
    # Check if there are more equals, if yes, take randomly
    i = np.searchsorted(v, x, side='right')

    return i


# index = search_closest_index_below(0, actions[:,0])

# print(actions[:, 0])
# print(index)
# print("size = "+str(actions.shape))


# In[7]:


def arg_max(state_action, range_limit):
    max_index_list = []
    max_value = state_action[0]
    for index, value in enumerate(state_action[:range_limit]):
        if value > max_value:
            max_index_list.clear()
            max_value = value
            max_index_list.append(index)
        elif value == max_value:
            max_index_list.append(index)
    return np.random.choice(max_index_list)


# In[8]:


def choose_action(state, balance):
    """
    Return the index of the chosen action
    """

    global epsilon
    global actions

    max_index = search_closest_index_below(balance, actions[:, 0])

    #     print("Possible purchases: \n"+str(actions[:max_index]))

    if np.random.rand() < epsilon:
        # Take random action
        action = np.random.choice(range(max_index))
    else:
        # Take action according to the q function table
        state_action = q_table[state]
        action = arg_max(state_action, max_index)
    return action


# In[9]:


def step(factories, action, balance, curr_prod_rate):
    global prod_rates
    global delta_T
    global actions

    # Take vector action ([nfact1, nfact2, ...]) cancelling balance and pass_flag
    action_v = actions[action, 2:]

    # Update number of factories taking that action (cancel pass_flag)
    factories = factories + action_v

    # Additional rate of prod
    #     print("prod_rates.shape = "+str(prod_rates.shape))
    #     print("action_v = "+str(action_v.shape))
    d_rate = np.dot(prod_rates, action_v.T)

    #     print("d_rate = "+str(d_rate))

    curr_prod_rate += d_rate

    #     print("curr_prod_rate = "+str(curr_prod_rate))

    # Reward = Total production for an interval with additional factories
    reward = curr_prod_rate * delta_T

    #     print("reward = "+str(reward))

    return factories, reward, curr_prod_rate


# In[10]:


# def learn(self, state, action, reward, next_state):
#         current_q = self.q_table[state][action]
#         # using Bellman Optimality Equation to update q function
#         new_q = reward + self.discount_factor * max(self.q_table[next_state])
#         self.q_table[state][action] += self.learning_rate * (new_q - current_q)


def learn(state, action, reward):
    global q_table
    global discount_factor
    global learning_rate

    current_q = q_table[state][action]
    #     print("current_q = "+str(current_q))
    #     print("q_table[next_state] ="+str(max(q_table[next_state])))
    # using Bellman Optimality Equation to update q function

    if state + 1 < q_table.shape[0]:
        new_q = reward + discount_factor * max(q_table[state + 1])
    else:
        new_q = reward
    #     print("reward = "+str(reward))
    #     print("discount_factor = "+str(discount_factor))
    #     print("new_q = "+str(new_q))
    q_table[state][action] += learning_rate * (new_q - current_q)


# In[11]:


def print_info(episode, current_balance, owned_factories):
    print("Episode " + str(episode) + ":\tBalance = " + str(current_balance) + "\tFactories = " + str(owned_factories))


# In[12]:


def print_time(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    time = str(int(h)) + ":" + str(int(m)) + ":" + str(int(s))
    return time


# In[13]:


reset = False
if reset:
    q_table = init_q_table(n_states, n_actions)

# Init exp. moving average
all_avg = [0]
# Number of episodes to take in count for avg (necessary only for calculating beta)
n_avg = 100
beta = 1 - 1 / n_avg

time_to_end = 0

for episode in range(n_episodes):

    # Resetting environment

    current_balance = start_balance
    owned_factories = np.zeros(n_factories)
    curr_prod_rate = 0
    tmp_avg = 0
    total_reward = 0
    # Start timer
    # tic = time.time()

    for state in range(n_states):
        #         print("State: "+str(current_state)+"\tBalance: "+str(current_balance))

        # Take index of vector action ([balance, pass_flag, nfact1, nfact2, ...])
        action = choose_action(state, current_balance)

        #         print("Take action: "+str(action)+" = "+str(actions[action]))

        cost_factories = actions[action, 0]
        current_balance -= cost_factories

        #         print("It will costs: "+str(cost_factories))

        owned_factories, reward, curr_prod_rate = step(owned_factories, action, current_balance, curr_prod_rate)

        learn(state, action, reward)

        current_balance += reward

        # Never decrease
        total_reward += reward

    # Stop timer
    # tac = time.time()

    #     time_elapsed = tac - tic
    #     time_to_end = 0.999*time_to_end + 0.001*(n_episodes-episode)*time_elapsed

    # After the episode, update average
    if episode > 0:
        # print(str(all_avg))
        tmp_avg = beta * all_avg[episode - 1] + (1 - beta) * total_reward
        all_avg.append(tmp_avg)
        # tmp_avg = tmp_avg /(1-np.power(beta,episode))
    if episode % 100 == 0:
        print("Episode " + str(episode) + "(" + str(
            round(episode / n_episodes * 100, 2)) + "%)" + " Avg total_reward = " + str(tmp_avg))
    #         print("Time to end: "+print_time(time_to_end))
    # print_info(episode, current_balance, owned_factories)
    # print("Exp-avg = "+str(tmp_avg))
    # if episode % 5000 == 0:
    #     plt.plot(all_avg)
    #     plt.show()

# In[14]:


# Greedy approach
current_balance = start_balance
owned_factories = np.zeros(n_factories)
current_state = 0
curr_prod_rate = 0
total_reward = 0

max_balance = 0

for current_state in range(n_states):
    print("Stato " + str(current_state))
    print("Balance: " + str(current_balance))
    print("Reward : " + str(total_reward))
    for factory in range(n_factories):
        # Start to buy from most expensive
        factory = n_factories - 1 - factory
        n = current_balance // costs[factory]
        if n > 0:
            print("COMPRO " + str(n) + " di tipo " + str(factory))
            owned_factories[factory] += n
            curr_prod_rate += n * prod_rates[factory]
            current_balance -= n * costs[factory]

    print("Owned = " + str(owned_factories))
    print("Prod_rate = " + str(curr_prod_rate))

    earn = curr_prod_rate * delta_T

    current_balance += earn
    total_reward += earn

    if max_balance < current_balance:
        max_balance = current_balance

    print("Balance: " + str(current_balance))
    print("Reward : " + str(total_reward) + "\n======\n")

print("Final reward: " + str(total_reward))
print("Max balance: " + str(max_balance))

# In[ ]:


X = np.arange(q_table.shape[1])
Y = np.arange(q_table.shape[0])
X, Y = np.meshgrid(X, Y)
Z = q_table / np.max(q_table)
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='hot', linewidth=0, antialiased=False)
ax.set_zlim(0, 1)

fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

# In[ ]:


im = plt.imshow(Z, cmap='hot')
plt.colorbar(im, orientation='horizontal')
plt.show()
