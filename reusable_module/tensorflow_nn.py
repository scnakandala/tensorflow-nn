import tensorflow as tf

class BNNState(object):
    
    def __init__(self, rk_variables, iteration, fired, fired_iteration):
        self.rk_variables = rk_variables
        self.iteration = iteration
        self.fired = fired
        self.fired_iteration = fired_iteration
    

class TFBNN(object):
    
    def __init__(self, num_neurons, iterations, h, firing_threshold, firing_reset,
                 current_functions, initial_state, inputs, connections, connection_weights):
        self.num_neurons = num_neurons
        self.iterations = iterations
        
        self.h = h
        self.firing_threshold = firing_threshold
        self.firing_reset = firing_reset
        
        self.current_functions = current_functions
        self.initial_state = initial_state
        self.inputs = inputs
        
        self.connections = connections

        self.connection_weights = connection_weights
        
        self.output = tf.scan(self.__neuron_combined_steps, self.inputs, initializer=[self.initial_state.rk_variables,
            self.initial_state.iteration, self.initial_state.fired, self.initial_state.fired_iteration])
        
    def run_simulation(self):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            iteration_vals = sess.run(self.output)
      
            return iteration_vals
    
    def __neuron_combined_steps(self, state, step_input):
        with tf.name_scope('RK1'):
            rk1 = self.__call_currents(state[0], state[1], state[2], state[3], step_input)
            
        with tf.name_scope('RK2'):
            rk2_in = []
            for i in range(len(rk1)):
                rk2_in.append(tf.add(tf.multiply(tf.divide(self.h, tf.constant(2.)), rk1[i]), state[0][i]))
                
            iteration_rk2 = tf.add(state[2], tf.divide(self.h, tf.constant(2.)))
            rk2 = self.__call_currents(rk2_in, iteration_rk2, state[2], state[3], step_input)
            
        with tf.name_scope('RK3'):
            rk3_in = []
            for i in range(len(rk2)):
                rk3_in.append(tf.add(tf.multiply(tf.divide(self.h, tf.constant(2.)), rk2[i]), state[0][i]))
                
            iteration_rk3 = tf.add(state[2], tf.divide(self.h, tf.constant(2.)))
            rk3 = self.__call_currents(rk3_in, iteration_rk3, state[2], state[3], step_input)
            
        with tf.name_scope('RK4'):
            rk4_in = []
            for i in range(len(rk3)):
                rk4_in.append(tf.add(tf.multiply(self.h, rk3[i]), state[0][i]))
                
            iteration_rk4 = tf.add(state[2], self.h)
            rk4 = self.__call_currents(rk4_in, iteration_rk4, state[2], state[3], step_input)
            
        rk_next = []
        for i in range(len(rk4)):
            rk_next.append(tf.add(tf.multiply(tf.constant(1/6), tf.add_n([rk1[i], tf.multiply(tf.constant(2.),
                        rk2[i]),tf.multiply(tf.constant(2.), rk3[i]), rk4[i]])), state[0][i]))
            
        fired = tf.where(tf.greater(rk_next[0], self.firing_threshold), tf.ones(self.num_neurons), tf.zeros(self.num_neurons))
        state[3] = tf.where(tf.greater(rk_next[0], self.firing_threshold), tf.add(tf.constant(1.), state[3]), state[3])
        
        # peak thresholding for neuron outputs
        rk_next[0] = tf.where(tf.greater(rk_next[0], self.firing_threshold), self.firing_reset*tf.ones_like(rk_next[0]),
                              rk_next[0])

        state[0] = rk_next
        state[1] = tf.add(state[1], 1.)
        state[2] = fired

        return state
            
        
    def __call_currents(self, rk_variables, iteration, fired, fired_iteration, step_input):
        with tf.name_scope('call_currents'):
            current_aggregate = None
            for fn in self.current_functions:
                if current_aggregate is None:
                    current_aggregate = fn(rk_variables, iteration, fired, fired_iteration, step_input,
                                  self.connections, self.connection_weights)
                    for i in range(len(current_aggregate)):
                        if current_aggregate[i] is None:
                            current_aggregate[i] = tf.zeros(self.num_neurons)
                else:
                    rk_temp = fn(rk_variables, iteration, fired, fired_iteration, step_input,
                                  self.connections, self.connection_weights)
                    for i in range(len(rk_temp)):
                        if rk_temp[i] is not None:
                            current_aggregate[i] = tf.add(current_aggregate[i],rk_temp[i])
                        
        return current_aggregate