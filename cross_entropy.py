import tensorflow as tf
import numpy as np

def network_variable(shape):
    return tf.Variable(tf.zeros(shape), dtype=tf.float32)

def agent_variable(shape, mu, sigma):
    initial = tf.truncated_normal(shape, mean=mu, stddev=sigma)
    return tf.Variable(initial, dtype=tf.float32)

L1_VARIABLES = 10
L2_VARIABLES = 10
L3_VARIABLES = 10

class cross_entropy():

    def __init__(self, state_space, action_space, agents,
                 sample_mu, sample_sigma, max_sigma=100,
                 inheritance=0.5):


        self.state_space  = state_space
        self.action_space = action_space
        self.agents       = agents

        self.sample_mu     = sample_mu
        self.sample_sigma  = sample_sigma
        self.max_sigma     = max_sigma
        self.inheritance   = inheritance

        self.inputs, self.outputs = self.build_network()
        self.network_variables    = tf.trainable_variables()

        update_ops       = []
        tensor_variables = []

        for _ in range(agents):
            variables, update_op = self.build_agent()

            tensor_variables.append(variables)
            update_ops.append(update_op)

        self.tensor_variables = tf.stack(tensor_variables)
        self.number_of_vars   = self.state_space * L1_VARIABLES\
                              + L2_VARIABLES * L2_VARIABLES\
                              + L3_VARIABLES * self.action_space

    def build_network(self):

        layer1 = network_variable([self.state_space, L1_VARIABLES])
        layer2 = network_variable([L2_VARIABLES, L2_VARIABLES])
        layer3 = network_variable([L3_VARIABLES, self.action_space])

        inputs = tf.placeholder([None, self.state_space], dtype=tf.float32)

        #########
        # Graph #
        #########

        h1 = tf.matmul(inputs, layer1)
        h1 = tf.nn.relu(h1)

        h2 = tf.matmul(h1, layer2)
        h2 = tf.nn.relu(h2)

        out = tf.matmul(h2, layer3)
        out = tf.nn.tanh(out)

        return inputs, out

    def build_agent(self):

        agent_variables = []
        for variable in self.network_variables:
            agent_variables.append(
                    agent_variable(variable.shape, 
                                   self.sample_mu,
                                   self.sample_sigma))


        set_agent_op = [network_var.assign(agent_var)
                            for network_var, agent_var in 
                                zip(self.network_variables, agent_variables)]

        return agent_variables, set_agent_op

    def build_distributions(self):

        sigmas    = tf.Variable(tf.zeros(self.number_of_vars))
        mus       = tf.Variable(tf.zeros(self.number_of_vars))

        dist_vars = tf.transpose(self.tensor_variables)

        fitness = tf.placeholder([self.agents], dtype=tf.float32)
        apex    = tf.nn.top_k(fitness, k=self.agents * self.inheritance, sorted=False)

        def sigmify(variable_row):
            variable_row = tf.gather_nd(variable_row, apex)
            _, variance = tf.moments(variable_row)
            return tf.sqrt(variance)

        def muify(variable_row):
            variables = tf.gather_nd(variable_row, apex) 
            return tf.reduce_mean(variables)


        new_sigmas = tf.map_fn(sigmify, dist_vars)
        new_mus    = tf.map_fn(muify, dist_vars)

        update_sigmas_op = [old_sigma.assign(new_sigma) 
                                for old_sigma, new_sigma 
                                    in zip(sigmas, new_sigmas)]

        update_mus_op    = [old_mu.assign(new_mu) 
                                for old_mu, new_mu 
                                    in zip(mus, new_mus)]

        def update_agent(agent_vars):
            agent_new_vars = tf.random_normal(agent_vars.shape, 

        update_agents_op =

        
        


    def __call__(self, actor, state):
        return 0
    
    def __len__(self):
        return 2


