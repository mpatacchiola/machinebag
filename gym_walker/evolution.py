

from __future__ import print_function
import tensorflow as tf
import gym
import numpy as np
import bitstring
import sys

gym_iterations = 300
total_episodes = 1
action_type = "continous"
gym_environment = "BipedalWalker-v2"
IS_RECURRENT = False
MUTATION_PROBABILITY = 0.05

# Parameters
total_epochs = 100
total_subjects = 300
fitness_array = np.zeros(total_subjects)

# Network Parameters
n_hidden_1 = 128 # 1st layer number of features
#n_hidden_2 = 256 # 2nd layer number of features
n_output = 4 # MNIST total classes (0-9 digits)
if(IS_RECURRENT == True): n_input = 24 + n_output
else: n_input = 24

# tf Graph input
x = tf.placeholder("float", [1, n_input])
y = tf.placeholder("float", [1, n_output])
tf_weights_h1 =tf.placeholder("float", [n_input, n_hidden_1])
#tf_weights_h2 = tf.placeholder("float", [n_hidden_1, n_hidden_2])
tf_weights_out = tf.placeholder("float", [n_hidden_1, n_output])
tf_biases_b1 = tf.placeholder("float", [n_hidden_1])
#tf_biases_b2 = tf.placeholder("float", [n_hidden_2])
tf_biases_out = tf.placeholder("float", [n_output])


# Create model
def multilayer_perceptron(x, weights_h1, biases_b1, weights_out, biases_out):
    # Hidden layer
    layer_1 = tf.add(tf.matmul(x, weights_h1), biases_b1)
    layer_1 = tf.nn.tanh(layer_1)
    # Hidden layer 
    #layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    #layer_2 = tf.nn.sigmoid(layer_2)
    # Output layer 
    out_layer = tf.matmul(layer_1, weights_out) + biases_out
    return tf.nn.tanh(out_layer)


def return_subjects_list(n=100, std = 0.5):
    subjects_list = list()
    for i in range(0, n):
        #std = np.random.uniform(0, std)
        weights_h1 = np.random.normal(0, std, [n_input, n_hidden_1]).astype(np.float32)       
        biases_h1 = np.random.normal(0, std, [n_hidden_1]).astype(np.float32) 
        weights_out = np.random.normal(0, std, [n_hidden_1, n_output]).astype(np.float32) 
        biases_out = np.random.normal(0, std, [n_output]).astype(np.float32)
        subjects_list.append((weights_h1, biases_h1, weights_out, biases_out))
    return subjects_list


def _return_mutated_subject(subject, mutation_probability=0.05, std=0.01):
    for weight_matrix in subject:
        for x in np.nditer(weight_matrix, op_flags=['readwrite']):
            bit_counter = 0
            x_bits = bitstring.BitArray(float=x, length=32)
            x_bits_list = x_bits.bin
            for bit in x_bits_list:
                random_sorted = np.random.uniform(0,1)
                if(random_sorted <= mutation_probability and bit=='0'): x_bits_list[bit_counter].replace('0', '1')
                elif(random_sorted <= mutation_probability and bit=='1'): x_bits_list[bit_counter].replace('1', '0')
                bit_counter += 1
            x_bits.bin = ''.join(x_bits_list)
            x[...] = x_bits.float
            #flip a coin which return true if lower than mutation_probability
            #if(np.random.uniform(0,1) <= mutation_probability): coin = True
            #else: coin = False
            #if(coin == True): x[...] = np.random.normal(x, std).astype(np.float16) 
    return subject

def return_mutated_subjects_list(subjects_list, fitness_array, mutation_probability=0.02, mutation_std=0.1, elite=10, newborns=24):
    mutated_list = list()
    argmax_fitness_array = np.argsort(fitness_array)[-elite:]
    #Copy the elite without mutations
    for subject_index in argmax_fitness_array:
        mutated_list.append(subjects_list[subject_index])
        for newbor_id in range(newborns):
            mutated_list.append(_return_mutated_subject(subjects_list[subject_index], mutation_probability))
    return mutated_list    

# Construct model
output = multilayer_perceptron(x, tf_weights_h1, tf_biases_b1, tf_weights_out, tf_biases_out)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    env = gym.make(gym_environment)
    subjects_list = return_subjects_list(total_subjects)
    # Training cycle
    for epoch in range(total_epochs):
        subject_counter = 0
        print("==========================")
        print("Epoch: " + str(epoch))
        print("Total subjects: " + str(len(subjects_list)))
        print("Mutation Probability: " + str(MUTATION_PROBABILITY))
        print("Fitness Min: " + str(np.amin(fitness_array)) + " of index [" + str(np.argmin(fitness_array)) + "]")
        print("Fitness Mean|Std: " + str(np.mean(fitness_array)) + " | " + str(np.std(fitness_array)))
        print("Fitness Max: " + str(np.amax(fitness_array)) + " of index [" + str(np.argmax(fitness_array)) + "]")

        fitness_array.fill(0.0)
        for subject_pair in subjects_list:
            sys.stdout.write("Subject: " + str(subject_counter+1) + '\r' )
            sys.stdout.flush()
            for episode in range(total_episodes):
                env.reset()
                observation, reward, done, info = env.step(env.action_space.sample())
                if(IS_RECURRENT == True): observation = np.append(observation, np.zeros(n_output))
                for t in range(gym_iterations):                
                    net_output = sess.run([output], feed_dict={x: observation.reshape(1,-1), tf_weights_h1: subject_pair[0], 
                                                           tf_biases_b1: subject_pair[1], tf_weights_out: subject_pair[2], tf_biases_out: subject_pair[3]})

                    if(action_type == "discrete"): action = np.argmax(net_output[0][0])
                    elif(action_type == "continous"): action = net_output[0][0]
                    observation, reward, done, info = env.step(action)
                    if(IS_RECURRENT == True): observation = np.append(observation, net_output[0][0])
                    fitness_array[subject_counter] += reward                
                    if done: break

            subject_counter += 1
        #Call the mutation function here
        subjects_list = return_mutated_subjects_list(subjects_list, fitness_array, MUTATION_PROBABILITY)
        leftover = total_subjects - len(subjects_list)
        subjects_list = subjects_list + return_subjects_list(leftover)

        if(epoch % 5 == 0):
            env.monitor.start('./evolution_log', force=True)
            env.reset()
            observation, reward, done, info = env.step(env.action_space.sample())
            if(IS_RECURRENT == True): observation = np.append(observation, np.zeros(n_output))
            for t in range(gym_iterations*2):
                env.render()
                net_output = sess.run([output], feed_dict={x: observation.reshape(1,-1), tf_weights_h1: subjects_list[0][0], 
                                                           tf_biases_b1: subjects_list[0][1], tf_weights_out: subjects_list[0][2], tf_biases_out: subjects_list[0][3]})

                if(action_type == "discrete"): action = np.argmax(net_output[0][0])
                elif(action_type == "continous"): action = net_output[0][0]
                observation, reward, done, info = env.step(action)
                if(IS_RECURRENT == True): observation = np.append(observation, net_output[0][0])             
                if done: break
            env.monitor.close()
            print("Last output: " + str(net_output[0][0]))


    

    print("Evolution Finished!")









