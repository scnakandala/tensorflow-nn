import tensorflow as tf

class BNNState(object):
    
    def __init__(self, rk_variables, iteration, fired, fired_iteration):
        self.rk_variables = rk_variables
        self.iteration = iteration
        self.fired = fired
        self.fired_iteration = fired_iteration
    

class TFBNN(object):
    
    def __init__(self, num_neurons, iterations, h, firing_threshold, firing_reset,
                 current_func, initial_state, inputs, connections, connection_weights):
        self.num_neurons = num_neurons
        self.iterations = iterations
        
        self.h = h
        self.firing_threshold = firing_threshold
        self.firing_reset = firing_reset
        
        self.current_func = current_func
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

        [rk_variables, iteration, fired, fired_iteration] = state
        
        with tf.name_scope('RK1'):
            rk1 = self.__call_currents(rk_variables, iteration, fired, fired_iteration, step_input)
            
        with tf.name_scope('RK2'):
            rk2_in = []
            for i in range(len(rk1)):
                temp = []
                for j in range(len(rk1[i])):
                    temp.append(tf.add(tf.multiply(tf.divide(self.h, tf.constant(2.)), rk1[i][j]), rk_variables[i][j]))
                rk2_in.append(temp)
            iteration_rk2 = tf.add(iteration, tf.divide(self.h, tf.constant(2.)))
            rk2 = self.__call_currents(rk2_in, iteration_rk2, fired, fired_iteration, step_input)

        with tf.name_scope('RK3'):
            rk3_in = []
            for i in range(len(rk2)):
                temp = []
                for j in range(len(rk2[i])):
                    temp.append(tf.add(tf.multiply(tf.divide(self.h, tf.constant(2.)), rk2[i][j]), rk_variables[i][j]))
                rk3_in.append(temp)
            iteration_rk3 = tf.add(iteration, tf.divide(self.h, tf.constant(2.)))
            rk3 = self.__call_currents(rk3_in, iteration_rk3, fired, fired_iteration, step_input)
            
        with tf.name_scope('RK4'):
            rk4_in = []
            for i in range(len(rk3)):
                temp = []
                for j in range(len(rk3[i])):
                    temp.append(tf.add(tf.multiply(self.h, rk3[i][j]), rk_variables[i][j]))
                rk4_in.append(temp)                
            iteration_rk4 = tf.add(iteration, self.h)
            rk4 = self.__call_currents(rk4_in, iteration_rk4, fired, fired_iteration, step_input)
        
        rk_next = []
        for i in range(len(rk4)):
            temp = []
            for j in range(len(rk4[i])):
                temp.append(
                    tf.add(tf.multiply(tf.constant(1/6), tf.add_n([rk1[i][j], tf.multiply(tf.constant(2.),
                        rk2[i][j]),tf.multiply(tf.constant(2.), rk3[i][j]), rk4[i][j]])), rk_variables[i][j]))
            rk_next.append(temp)
            
        fired_new = []
        fired_iteration_new = []
        
        for i in range(len(rk_next)):
            fired_new.append(tf.where(tf.greater(rk_next[i][0], self.firing_threshold),
                                  tf.ones_like(rk_next[i][0]), tf.zeros_like(rk_next[i][0])))
            
            fired_iteration_new.append(tf.where(tf.greater(rk_next[i][0], self.firing_threshold),
                                            tf.add(tf.constant(1.), fired_iteration[i]), fired_iteration[i]))

            # peak thresholding for neuron outputs
            rk_next[i][0] = tf.where(tf.greater(rk_next[i][0], self.firing_threshold),
                                     self.firing_reset*tf.ones_like(rk_next[i][0]), rk_next[i][0])
        
        return [rk_next, tf.add(state[1], 1.), fired_new, fired_iteration_new]
            
        
    def __call_currents(self, rk_variables, iteration, fired, fired_iteration, step_input):
        return self.current_func(rk_variables, iteration, fired, fired_iteration, step_input,
                                  self.connections, self.connection_weights)