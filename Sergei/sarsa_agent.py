import sys, os
import pylab as plb
from tqdm import tqdm
import numpy as np
import mountaincar
from matplotlib import pyplot as plt
import pickle, hashlib, warnings

class SARSAEligibilityAgent():
    """
    A SARSA + eligibility trace agent for Mountain Car task
    """

    def __init__(self, mountain_car = None, x_linspace = (-150, 30, 20),
                v_linspace = (-15, 15, 20), w = None, tau = 1, gamma = 0.95,
                 eta = 0.001, lambda_ = 0.95):
        ''' Initialize the object '''
        
        # saving the environment object
        if mountain_car is None:
            self.mountain_car = mountaincar.MountainCar()
        else:
            self.mountain_car = mountain_car
        
        # range for x neurons grid
        self.x_values = np.linspace(*x_linspace)

        # range for v neurons grid
        self.v_values = np.linspace(*v_linspace)

        # steps x and v
        self.delta_x = self.x_values[1] - self.x_values[0]
        self.delta_v = self.v_values[1] - self.v_values[0]

        # sigmas x and v
        self.sigma_x = np.array([self.delta_x] * len(self.x_values))
        self.sigma_v = np.array([self.delta_v] * len(self.v_values))

        # number of actions
        self.n_actions = 3

        # number of neurons
        self.n_neurons = len(self.x_values) * len(self.v_values)

        # weight matrix
        if w is None:
            #self.w = np.random.randn(self.n_actions, self.n_neurons)
            self.w = np.zeros((self.n_actions, self.n_neurons))
        else:
            self.w = w

        # history of w
        self.w_history = []
        self.w_history.append(self.w)

        # history of escape latency
        self.escape_latency = []

        # sampling softmax temperature
        self.tau = tau
        
        # reward discount factor
        self.gamma = gamma
        
        # learning rate
        self.eta = eta
        
        # eligibility trace parameter
        self.lambda_ = lambda_

    def r(self, x, v):
        ''' get neuron activations for s = (x, v) '''
        # x in rows, v in columns
        part_x = np.reshape(np.divide((self.x_values - x) ** 2, self.sigma_x ** 2), (-1, 1))
        part_v = np.reshape(np.divide((self.v_values - v) ** 2, self.sigma_v ** 2), (1, -1))
        return np.exp(-(part_x + part_v))

    def get_Q(self, x, v):
        ''' Get Q-function at given s = (x, v) with weights w '''
        
        return np.reshape(self.w @ np.reshape(self.r(x, v), (-1, 1)), (-1,))
    
    def get_Q_matrix(self):
        ''' Returns matrices indexed by (x, v) with values argmax Q and max Q '''
        
        result_index = np.zeros((len(self.x_values), len(self.v_values)))
        result_value = np.zeros((len(self.x_values), len(self.v_values)))
        for i, x in enumerate(self.x_values):
            for j, v in enumerate(self.v_values):
                Q = self.get_Q(x, v)
                result_index[i, j] = np.argmax(Q)
                result_value[i, j] = np.max(Q)
        return result_index, result_value
    
    def plot_Q_matrix(self):
        ''' Plot vector field action(x, v) '''
        
        A, V = self.get_Q_matrix()
        
        fig = plt.figure()
        plt.title('Action selection')
        plt.xlabel('$x$')
        plt.ylabel('$\dot{x}$')
        plt.xlim((np.min(self.x_values) * 1.1, np.max(self.x_values) * 1.1))
        plt.ylim((np.min(self.v_values) * 1.1, np.max(self.v_values) * 1.1))
        for i, x in enumerate(self.x_values):
            for j, v in enumerate(self.v_values):
                action = A[i, j]
                value = V[i, j]
                if action == 1:
                    plt.scatter(x, v, 3, c = 'black')
                else:
                    direction = action - 1
                    plt.arrow(x, v, 3 * direction, 0, head_width=0.5, head_length=1, color = 'red' if direction < 0 else 'green')
        return fig

    def plot_Q_matrix_process(self, title, png_fn):
        ''' call plot_Q_matrix in a separate python process '''

        # create unique hash
        m = hashlib.md5()
        m.update(bytes(title + '#' + png_fn, 'ascii'))

        # dump self
        pickle_fn = m.hexdigest() + '.p'
        pickle.dump(self, open(pickle_fn, "wb"))

        # filename to save code
        filename = pickle_fn + '.py'

        code = """from sarsa_agent import *

pickle_fn = '""" + pickle_fn + """'
title = '""" + title + """'
filename = '""" + png_fn + """'

agent = pickle.load(open(pickle_fn, 'rb' ))
fig = agent.plot_Q_matrix()
plt.title(title)
plt.savefig(filename, bbox_inches = 'tight')"""

        f = open(filename, 'w')
        f.write(code)
        f.close()

        os.system('python ' + filename)
        os.system('rm ' + ' '.join([filename, pickle_fn]))

    def get_action_probas(self, Q):
        ''' get action probabilities as a vector '''

        with warnings.catch_warnings():
            # trying to get true vector
            warnings.filterwarnings('error')
            try:
                vector = np.exp(Q / self.tau)
                res = vector / np.sum(vector)
                assert np.abs(np.sum(res) - 1) < 1e-1
                return res
            
            # otherwise just outputting the argmax (happens if tau is very small and components are huge, in this case only one wins)
            except:
                res = np.zeros(self.n_actions)
                res[np.argmax(Q)] = 1
                return res

    def get_action_index(self, x, v, greedy = False):
        ''' Sample action for s = (x, v) and weights w with parameter tau '''

        # computing Q
        Q = self.get_Q(x, v)
        
        # greedy selection (tau = 0)
        if self.tau == 0 or greedy:
            return np.argmax(Q)
        elif self.tau == np.inf: # uniform selection (infinite tau)
            return np.random.choice(range(self.n_actions))
        else: # all other cases. returning greedy if can't do choice
            action_probas = self.get_action_probas(Q)
            try:
                return np.random.choice(range(self.n_actions), p = action_probas)
            except: return np.argmax(Q)

    def update_w(self, x, v, a_index, delta):
        ''' Perform gradient descent on Q(s, a) by delta given s and a'''
        
        dQ_dwa = np.reshape(self.r(x, v), -1)
        self.w[a_index, :] += delta * dQ_dwa

    def visualize_trial(self, n_steps = 200):
        """Do a trial without learning, with display.

        Parameters
        ----------
        n_steps -- number of steps to simulate for
        """
        
        # prepare for the visualization
        plb.ion()
        mv = mountaincar.MountainCarViewer(self.mountain_car)
        mv.create_figure(n_steps, n_steps)
        plb.draw()
        plb.pause(1e-3)
            
        # make sure the mountain-car is reset
        self.mountain_car.reset()
        
        for n in (range(n_steps)):
            print('\rt =', self.mountain_car.t)
            sys.stdout.flush()
            
            # get current state
            s = (self.mountain_car.x, self.mountain_car.x_d)

            # selection current action based on softmax
            action_index = self.get_action_index(*s, greedy = True)
            
            # perform the action
            self.mountain_car.apply_force(action_index - 1)
            
            # simulate the timestep
            self.mountain_car.simulate_timesteps(100, 0.01)

            # update the visualization
            mv.update_figure()
            plb.draw()
            plb.pause(1e-3)
            
            # check for rewards
            if self.mountain_car.R > 0.0:
                print("\rreward obtained at t = ", self.mountain_car.t)
                break

    def learn(self, max_steps = 1000, use_tqdm = False):
        """Do a trial with learning, with no display.

        Parameters
        ----------
        n_steps -- number of steps to simulate for
        """
            
        # make sure the mountain-car is reset
        self.mountain_car.reset()

        # saved previous state
        old_s = None
        old_a = None
        
        # all states and actions array
        all_s_a = []
        
        # is finished
        finished = False
        
        # conditional tqdm
        tqdm_obj = []
        if use_tqdm:
            tqdm_obj = tqdm(total = max_steps)
            
        # loop over states
        for n in range(max_steps):
            # get current state
            s = (self.mountain_car.x, self.mountain_car.x_d)

            # selection current action based on softmax
            action_index = self.get_action_index(*s)

            # save s, a
            all_s_a.append((s, action_index))

            # perform the action
            self.mountain_car.apply_force(action_index - 1)

            # simulate the timestep
            self.mountain_car.simulate_timesteps(100, 0.01)

            # check for rewards (runs at the end once)
            if self.mountain_car.R > 0.0:
                # print the obtained reward
                #print("reward obtained at t = " + str(self.mountain_car.t))

                # compute vector [xi ^ (T-1), ..., 1] where xi = gamma * lambda
                eligibility_trace = np.flip(np.array([self.gamma * self.lambda_]) **
                                            np.arange(len(all_s_a)), axis = 0)

                # compute the update for the Q function
                # update = eta * delta (from lectures)

                # old Q
                Q = self.get_Q(*old_s)[old_a]

                # new Q
                Q1 = np.max(self.get_Q(*s))

                # eta * (R + gamma * Qnew - Qold)
                update = self.eta * (self.mountain_car.R + self.gamma * Q1 - Q)

                # loop over history
                i = 0 
                for s0, a0 in all_s_a:
                    # updating Q based on SARSA and eligibility traces
                    self.update_w(s0[0], s0[1], a0, update * eligibility_trace[i])
                    i += 1

                # no steps after the reward
                finished = True
    
                if use_tqdm:
                    pbar.update()
                    pbar.set_postfix(info = 'Reward obtained')
                self.escape_latency.append(n)
                self.w_history.append(self.w)
                return n

            # saving old state
            old_s = s 
            old_a = action_index
                
            if use_tqdm:
                pbar.update()

        #if not finished:
        #    print('No reward')

        # saving data
        result = max_steps + 1
        self.escape_latency.append(-1)
        self.w_history.append(self.w)
        return result

def get_agent(seed = None, iterations = 50, max_steps = 1000, use_tqdm = True, **kwargs):
    # implement seed
    if seed is not None: np.random.seed(seed)

    # create an agent
    agent = SARSAEligibilityAgent(**kwargs)

    # number of iterations with reward
    finished = 0

    # learning
    if use_tqdm: pbar = tqdm(total = iterations)

    for i in range(iterations):
        result = agent.learn(max_steps)
        if result >= max_steps + 1:
            result = 0
        if result > 0:
            finished += 1
            if use_tqdm: pbar.set_postfix(with_reward = finished, with_reward_percent = round(100 * finished / (i + 1)))
        if use_tqdm: pbar.update(1)

    return agent
