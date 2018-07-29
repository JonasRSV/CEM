import tensorflow as tf
import numpy as np


def weigth_variable(shape, mu, sigma):
    initial = tf.random_normal(shape, mean=mu, stddev=sigma)
    return tf.Variable(initial, dtype=tf.float32)


class CE(object):

    def __init__(self, state_space, action_space,
                 agents=100, init_mu=0, init_sigma=10,
                 max_sigma=100, inheritance=0.5,
                 model_blueprint=None):

        self.state_space  = state_space
        self.action_space = action_space 
        self.agents       = agents

        self.init_mu     = init_mu
        self.init_sigma  = init_sigma
        self.max_sigma   = max_sigma
        self.inheritance = inheritance


        if model_blueprint is None:
            model_blueprint = [(state_space, 10),
                               (10, 10),
                               (10, action_space)]

        a_vs, a_ls = zip(*[self.__build_agent(model_blueprint) for _ in range(agents)])

        fitness\
            , u_sg_op\
            , u_mu_op\
            , u_apex_op\
            , u_agents_op\
            , apex = self.__build_entropic_normal_dist_search(a_vs)



        self.a_vs = a_vs
        self.a_ls = a_ls
        self.apex = apex

        self.fitness     = fitness
        self.u_entropy   = (u_sg_op, u_mu_op, u_apex_op)
        self.u_agents_op = u_agents_op

    def __build_agent(self, blueprint):
        """ Build agent variables """

        num_v = 0
        for in_dim, out_dim in blueprint:
            num_v += out_dim * (in_dim + 1)


        v = weigth_variable([num_v],
                            self.init_mu,
                            self.init_sigma)

        s      = 0
        layers = []
        for in_dim, out_dim in blueprint:
            chunck = out_dim * (in_dim + 1)
            var_s  = v[s: s + chunck]

            layer_sq = tf.reshape(var_s, (out_dim, in_dim + 1))
            weigths  = layer_sq[:, :-1]
            bias     = tf.expand_dims(layer_sq[:, -1], [1])

            layers.append((weigths, bias))

            s = chunck

        return v, layers

    def __build_entropic_normal_dist_search(self, vs):
        num_vars = vs[0].shape[0]
        stack    = tf.stack(vs)

        mu = tf.Variable(tf.zeros(num_vars), dtype=tf.float32)
        sg = tf.Variable(tf.zeros(num_vars), dtype=tf.float32)

        #############
        # Summaries #
        #############
        mean, var = tf.nn.moments(mu, [0])
        self.norm_mu_mean = tf.summary.scalar("mus/mean", mean)
        self.norm_mu_var  = tf.summary.scalar("mus/variance", var)

        mean, var = tf.nn.moments(sg, [0])
        self.norm_sg_mean = tf.summary.scalar("sigmas/mean", mean)
        self.norm_sg_var  = tf.summary.scalar("sigmas/variance", var)

        ##############
        # Update ops #
        ##############
        norm   = tf.distributions.Normal(loc=mu, scale=sg)
        num_a  = int(self.agents * self.inheritance)
        pa     = (tf.Variable(0, dtype=tf.float32), tf.Variable(0, dtype=tf.int32))

        fitness = tf.placeholder(shape=[self.agents], dtype=tf.float32)
        s, a    =  tf.nn.top_k(fitness, k=num_a, sorted=True)

        def a_op(s_, a_, ps_, pa_):
            p_as = ps_.assign(s_)
            a_as = pa_.assign(a_)

            return (p_as, a_as)

        u_apex_op = tf.cond(tf.greater(s[0], pa[0]),
                           true_fn=lambda: a_op(s[0], a[0], pa[0], pa[1]),
                           false_fn=lambda: a_op(pa[0], pa[1], pa[0], pa[1]))


        def u_sg(var_row):
            var_row = tf.gather(var_row, a)

            _, variance = tf.nn.moments(var_row, 0)
            return tf.clip_by_value(tf.sqrt(variance),
                                    -self.max_sigma,
                                    self.max_sigma)


        def u_mu(var_row):
            var_row = tf.gather(var_row, a)
            return tf.reduce_mean(var_row)

        var_rows = tf.transpose(stack)
        u_sg     = tf.map_fn(u_sg, var_rows)
        u_mu     = tf.map_fn(u_mu, var_rows)

        u_sg_op = sg.assign(u_sg)
        u_mu_op = mu.assign(u_mu)

        a_norms = tf.unstack(norm.sample(self.agents))

        u_agents_op = [tf.cond(
                        tf.not_equal(pa[1], tf.constant(i)),
                        true_fn=lambda: agent.assign(d),
                        false_fn=lambda: agent.assign(agent))
                            for i, (agent, d) in enumerate(zip(vs, a_norms))]


        return fitness\
                , u_sg_op\
                , u_mu_op\
                , u_apex_op\
                , u_agents_op\
                , pa[1]

    def __len__(self):
        return self.agents

    def get_apex(self):
        return self.session.run(self.apex)

    def summary_add_scalar(self, tag, value):
        summary = tf.Summary()
        summary.value.add(tag=tag, simple_value=float(value))

        self.summary_writer.add_summary(summary, self.global_step)
        self.summary_writer.flush()


    ###############################
    # Code for running this in a  #
    # single python process       #
    ###############################
    def spi(self, session):
        self.single_process = True
        self.session        = session

        self.a_ts = [self.__build_agent_tensors(a_l) for a_l in self.a_ls]

        #############
        # Summaries #
        #############
        self.summary_writer = tf.summary.FileWriter("summaries/", session.graph)
        self.global_step    = 0


    def __build_agent_tensors(self, a_l):
        state = tf.placeholder(dtype=tf.float32, shape=[self.state_space, 1])
        prev  = state
        for weigth, bias in a_l:
            prev = tf.nn.tanh(tf.matmul(weigth, prev) + bias)

        return state, prev


    def pred_sp(self, state, agent):
        state_t, output_t = self.a_ts[agent]
        return self.session.run(output_t, feed_dict={state_t: state})

    def train_sp(self, fitnesses):
        self.session.run(self.u_entropy, feed_dict={self.fitness: fitnesses})
        self.session.run(self.u_agents_op, feed_dict={self.fitness: fitnesses})
        return None

    def summarize_sp(self):
        summaries = self.session.run((self.norm_mu_var,
                                   self.norm_mu_mean,
                                   self.norm_sg_mean,
                                   self.norm_sg_var))
        for s in summaries:
            self.summary_writer.add_summary(s, self.global_step)

        self.summary_writer.flush()
        self.global_step += 1

    ##############################
    # Code for running this over #
    # multiple processes         #
    ##############################

    # TODO:
    def multi_process_init(self):
        pass






























