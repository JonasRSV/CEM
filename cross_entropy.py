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

    def __init__(self, sess, state_space, action_space, agents=10,
                 sample_mu=0, sample_sigma=4, max_sigma=100,
                 inheritance=0.2):

        self.sess         = sess
        self.state_space  = state_space
        self.action_space = action_space
        self.agents       = agents

        self.sample_mu     = sample_mu
        self.sample_sigma  = sample_sigma
        self.max_sigma     = max_sigma
        self.inheritance   = inheritance

        #################
        # For summaries #
        #################
        self.global_step = 0
        self.summaries   = []

        self.inputs, self.outputs = self.build_network()
        network_variables    = tf.trainable_variables()

        self.number_of_vars   = self.state_space * L1_VARIABLES\
                              + L2_VARIABLES * L2_VARIABLES\
                              + L3_VARIABLES * self.action_space
        
        self.apex_variables, s_apex_op = self.build_agent("Apex", network_variables)

        s_agents_ops  = []
        as_vars       = []

        for i in range(agents):
            a_vars, s_agent_op = self.build_agent(str(i), network_variables)

            as_vars.append(a_vars)
            s_agents_ops.append(s_agent_op)

        fitness\
        , u_sigma_op\
        , u_mus_op\
        , u_agents_op\
        , a_apex_op\
        , apexes = self.build_distributions(as_vars)

        self.s_agents_ops = s_agents_ops
        self.fitness      = fitness
        self.u_random_op  = (u_sigma_op, u_mus_op)
        self.u_agents_ops = (u_agents_op, a_apex_op)
        self.s_apex_op    = s_apex_op
        self.apexes       = apexes

        ##################
        # Summary Writer #
        ##################
        self.summary_w   = tf.summary.FileWriter("summaries/", sess.graph)

    def build_network(self):

        layer1 = network_variable([self.state_space, L1_VARIABLES])
        layer2 = network_variable([L2_VARIABLES, L2_VARIABLES])
        layer3 = network_variable([L3_VARIABLES, self.action_space])

        inputs = tf.placeholder(shape=[None, self.state_space], dtype=tf.float32)

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

    def build_agent(self, name, network_variables):

        a_vars = agent_variable([self.number_of_vars], 
                               self.sample_mu, 
                               self.sample_sigma)

        #############
        # Summaries #
        #############
        # mean, variance = tf.nn.moments(a_vars, [0])
        # self.summaries.append(tf.summary.scalar(name + "/mean", mean))

        previous_index     = 0
        agent_network_vars = []
        for variable in network_variables:
            network_var_shape = variable.shape

            var_start = previous_index
            var_end   = previous_index + np.prod(network_var_shape)
            
            layer_variables  = a_vars[var_start:var_end]
            layer_variables  = tf.reshape(layer_variables, network_var_shape)

            agent_network_vars.append(layer_variables)

            previous_index = var_end

        set_agent_op = [network_var.assign(agent_var)
                            for network_var, agent_var in 
                                zip(network_variables, agent_network_vars)]

        return a_vars, set_agent_op

    def build_distributions(self, as_vars):

        as_vars_stack = tf.stack(as_vars)

        sigmas   = tf.Variable(tf.zeros(self.number_of_vars), dtype=tf.float32)
        mus      = tf.Variable(tf.zeros(self.number_of_vars), dtype=tf.float32)
        CEM_dist = tf.distributions.Normal(loc=mus, scale=sigmas)

        #############
        # Summaries #
        #############
        mean, variance = tf.nn.moments(sigmas, [0])
        self.summaries.append(tf.summary.scalar("sigmas/mean", mean))
        self.summaries.append(tf.summary.scalar("sigmas/variance", variance))

        mean, variance = tf.nn.moments(mus, [0])
        self.summaries.append(tf.summary.scalar("mus/mean", mean))
        self.summaries.append(tf.summary.scalar("mus/variance", variance))

        ###################
        # Get best agents #
        ###################
        fitness = tf.placeholder(shape=[self.agents], dtype=tf.float32)
        _, apexes = tf.nn.top_k(fitness, k=int(self.agents * self.inheritance), sorted=True)

        ######################
        # Remember best apex #
        ######################

        apex_agent = apexes[0]
        a_apex_op  = self.apex_variables.assign(as_vars_stack[apex_agent])

        def ev_sigma(variable_row):
            variable_row = tf.gather(variable_row, apexes)
            _, variance = tf.nn.moments(variable_row, 0)
            return tf.clip_by_value(tf.sqrt(variance),
                                    -self.max_sigma,
                                    self.max_sigma)

        def ev_mu(variable_row):
            variables = tf.gather(variable_row, apexes) 
            return tf.reduce_mean(variables)

        variable_rows = tf.transpose(as_vars_stack)

        n_sigmas = tf.map_fn(ev_sigma, variable_rows)
        n_mus    = tf.map_fn(ev_mu, variable_rows)

        update_sigmas_op = sigmas.assign(n_sigmas) 
        update_mus_op    = mus.assign(n_mus)

        n_agents_dists   = tf.unstack(CEM_dist.sample(self.agents))

        update_agents_op = [agent.assign(dist)
                                for agent, dist, in zip(as_vars, n_agents_dists)]

        return fitness\
               ,update_sigmas_op\
               , update_mus_op\
               , update_agents_op\
               , a_apex_op\
               , apexes

    def set_apex(self):
        self.sess.run(self.s_apex_op)

    def set_agent(self, agent):
        self.sess.run(self.s_agents_ops[agent])

    def __call__(self, state):
        return self.sess.run(self.outputs, feed_dict={self.inputs: state})

    def train(self, fitnesses):
        self.sess.run(self.u_random_op, feed_dict={self.fitness: fitnesses})
        self.sess.run(self.u_agents_ops, feed_dict={self.fitness: fitnesses})
        return None
    
    def __len__(self):
        return self.agents

    def summarize(self):
        summaries = self.sess.run(self.summaries)

        for summary in summaries:
            self.summary_w.add_summary(summary, self.global_step)

        self.summary_w.flush()
        self.global_step += 1


    def add_scalar(self, tag, value):
        summary = tf.Summary()
        summary.value.add(tag=tag, simple_value=float(value))

        self.summary_w.add_summary(summary, self.global_step)
        self.summary_w.flush()



