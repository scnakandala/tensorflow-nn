import tensorflow as tf

class BNNState(object):
    
    def __init__(self, rk_variables, channel_params, iteration, fired):
        self.rk_variables = []
        for rk_var in rk_variables:
            self.rk_variables.append(tf.Variable(rk_var, dtype=tf.float32))
            
        self.channel_params = []
        for param in channel_params:
            self.channel_params.append(tf.Variable(param, dtype=tf.float32))
            
        self.iteration = tf.Variable(iteration)
        self.fired = tf.Variable(fired)
    

class TFBNN(object):
    
    def __init__(self, num_neurons, iterations, ve, h, firing_threshold, firing_reset,
                 current_functions, initial_state, inputs, connections, connection_weights):
        self.num_neurons = num_neurons
        self.iterations = iterations
        
        self.ve = tf.constant(ve)
        self.h = tf.constant(h)
        self.firing_threshold = tf.constant(firing_threshold)
        self.firing_reset = tf.constant(firing_reset)
        
        self.current_functions = current_functions
        self.initial_state = initial_state
        self.inputs = tf.Variable(inputs, dtype=tf.float32, name='inputs')
        
        self.connections = []
        for conn in connections:
            self.connections.append(tf.Variable(conn, dtype=tf.int32))

        self.connection_weights = []
        for conn_weight in connection_weights:
            self.connection_weights.append(tf.Variable(conn_weight, dtype=tf.float32))
        
        self.output = tf.scan(self.__neuron_combined_steps, self.inputs, initializer=[self.initial_state.rk_variables,
            self.initial_state.channel_params, self.initial_state.iteration, self.initial_state.fired])
        
    def run_simulation(self):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            iteration_vals = sess.run(self.output)
      
            return iteration_vals
    
    def __neuron_combined_steps(self, state, step_input):
        with tf.name_scope('RK1'):
            rk1 = self.__call_currents([self.ve, self.h, self.firing_threshold, self.firing_reset],
                                       state[0], state[1], state[2], state[3], step_input)
            
        with tf.name_scope('RK2'):
            rk2_in = []
            for i in range(len(rk1)):
                rk2_in.append(tf.add(tf.multiply(tf.divide(self.h, tf.constant(2.)), rk1[i]), state[0][i]))
                
            iteration_rk2 = tf.add(state[2], tf.divide(self.h, tf.constant(2.)))
            rk2 = self.__call_currents([self.ve, self.h, self.firing_threshold, self.firing_reset],
                                       rk2_in, state[1], iteration_rk2, state[3], step_input)
            
        with tf.name_scope('RK3'):
            rk3_in = []
            for i in range(len(rk2)):
                rk3_in.append(tf.add(tf.multiply(tf.divide(self.h, tf.constant(2.)), rk2[i]), state[0][i]))
                
            iteration_rk3 = tf.add(state[2], tf.divide(self.h, tf.constant(2.)))
            rk3 = self.__call_currents([self.ve, self.h, self.firing_threshold, self.firing_reset],
                                       rk3_in, state[1], iteration_rk3, state[3], step_input)
            
        with tf.name_scope('RK4'):
            rk4_in = []
            for i in range(len(rk3)):
                rk4_in.append(tf.add(tf.multiply(self.h, rk3[i]), state[0][i]))
                
            iteration_rk4 = tf.add(state[2], self.h)
            rk4 = self.__call_currents([self.ve, self.h, self.firing_threshold, self.firing_reset],
                                       rk4_in, state[1], iteration_rk4, state[3], step_input)
            
        rk_next = []
        for i in range(len(rk4)):
            rk_next.append(tf.add(tf.multiply(tf.constant(1/6), tf.add_n([rk1[i], tf.multiply(tf.constant(2.),
                        rk2[i]),tf.multiply(tf.constant(2.), rk3[i]), rk4[i]])), state[0][i]))
            
        
        fired = tf.where(tf.greater(rk_next[0], self.firing_threshold), tf.ones(self.num_neurons), tf.zeros(self.num_neurons))
        
        # peak thresholding for neuron outputs
        rk_next[0] = tf.where(tf.greater(rk_next[0], self.firing_threshold), self.firing_reset*tf.ones_like(rk_next[0]),
                              rk_next[0])
            
        state[0] = rk_next
        state[3] = fired
        
        return state
            
        
    def __call_currents(self, constants, rk_variables, channel_params, iteration, fired, step_input):
        with tf.name_scope('call_currents'):
            current_aggregate = None
            for fn in self.current_functions:
                if current_aggregate is None:
                    current_aggregate = fn(constants, rk_variables, channel_params, iteration, fired, step_input,
                                  self.connections, self.connection_weights)
                else:
                    rk_temp = fn(constants, rk_variables, channel_params, iteration, fired, step_input,
                                  self.connections, self.connection_weights)
                    for i in range(len(rk_temp)):
                        current_aggregate[i] = tf.add(current_aggregate[i],rk_temp[i])
                        
        return current_aggregate