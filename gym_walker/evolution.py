

from __future__ import print_function
import tensorflow as tf
import gym
import numpy as np

gym_iterations = 1000

# Parameters
learning_rate = 0.01
total_epochs = 100
display_step = 1
subjects_total = 100
fitness_array = np.zeros(subjects_total)


# Network Parameters
n_hidden_1 = 512 # 1st layer number of features
#n_hidden_2 = 256 # 2nd layer number of features
n_input = 24 
n_output = 4 # MNIST total classes (0-9 digits)

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
    layer_1 = tf.nn.sigmoid(layer_1)
    # Hidden layer 
    #layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    #layer_2 = tf.nn.sigmoid(layer_2)
    # Output layer 
    out_layer = tf.matmul(layer_1, weights_out) + biases_out
    return tf.nn.sigmoid(out_layer)


def return_subjects_list(n=100, std = 0.9):
    subjects_list = list()
    for i in range(0, n):
        weights_h1 = np.random.normal(0, std, [n_input, n_hidden_1],)
        biases_h1 = np.zeros([n_hidden_1])
        weights_out = np.random.normal(0, std, [n_hidden_1, n_output])
        biases_out = np.zeros([n_output])
        subjects_list.append((weights_h1, biases_h1, weights_out, biases_out))
    return subjects_list


def _return_mutated_subject(subject, mutation_probability=0.05, std=0.01):
    for weight_matrix in subject:
        for x in np.nditer(weight_matrix, op_flags=['readwrite']):
            #flip a coin which return true if lower than mutation_probability
            if(np.random.uniform(0,1) <= mutation_probability): coin = True
            else: coin = False
            if(coin == True): x[...] = np.random.normal(x, std)
    return subject

def return_mutated_subjects_list(subjects_list, fitness_array, elite=10, newborns=9):
    mutated_list = list()
    argmax_fitness_array = np.argsort(fitness_array)[-elite:]
    #Copy the elite without mutations
    for subject_index in argmax_fitness_array:
        mutated_list.append(subjects_list[subject_index])
        for newbor_id in range(newborns):
            mutated_list.append(_return_mutated_subject(subjects_list[subject_index]))
    return mutated_list    

# Construct model
output = multilayer_perceptron(x, tf_weights_h1, tf_biases_b1, tf_weights_out, tf_biases_out)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    env = gym.make('BipedalWalker-v2')
    subjects_list = return_subjects_list(subjects_total)
    # Training cycle
    for epoch in range(total_epochs):
        subject_counter = 0
        print("==========================")
        print("Epoch: " + str(epoch))
        print("Total subjects: " + str(len(subjects_list)))
        print("Fitness Mean: " + str(np.sum(fitness_array)/fitness_array.shape[0]))   
        print("fitness Max: " + str(np.amax(fitness_array)))  
        fitness_array.fill(0.0)
        for subject_pair in subjects_list:
            env.reset()
            observation, reward, done, info = env.step(env.action_space.sample())
            for t in range(gym_iterations):
                net_output = sess.run([output], feed_dict={x: observation.reshape(1,n_input), tf_weights_h1: subject_pair[0], 
                                                           tf_biases_b1: subject_pair[1], tf_weights_out: subject_pair[2], tf_biases_out: subject_pair[3]})

                #action = env.action_space.sample()   # np.array([net_output[0][0]])
                #action = np.argmax(net_output)
                #print(type(net_output[0]))
                #print(net_output[0][0][0])
                #action = np.array(net_output)
                observation, reward, done, info = env.step(net_output[0][0]) # take a random action
                fitness_array[subject_counter] += reward                
                if done: break

            #print("Subject: " + str(subject_counter) + "; Fitness: " + str(fitness_array[subject_counter]))
            subject_counter += 1
        #Call the mutation function here
        subjects_list = return_mutated_subjects_list(subjects_list, fitness_array)

        if(epoch % 2 == 0):
            env.reset()
            observation, reward, done, info = env.step(env.action_space.sample())
            for t in range(gym_iterations):
                env.render()
                net_output = sess.run([output], feed_dict={x: observation.reshape(1,n_input), tf_weights_h1: subjects_list[0][0], 
                                                           tf_biases_b1: subjects_list[0][1], tf_weights_out: subjects_list[0][2], tf_biases_out: subjects_list[0][3]})
                observation, reward, done, info = env.step(net_output[0][0]) # take a random action                
                if done: break


    

    print("Evolution Finished!")









