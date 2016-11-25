

from __future__ import print_function
import tensorflow as tf
import gym
import numpy as np

gym_iterations = 1000

# Parameters
learning_rate = 0.01
training_epochs = 10000
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_input = 3 
n_classes = 1 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [1, n_input])
y = tf.placeholder("float", [1, n_classes])
tf_reward = tf.Variable(0.0)

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)
    # Hidden layer 
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.sigmoid(layer_2)
    # Output layer 
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return tf.nn.sigmoid(out_layer)

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], stddev=0.1)),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev=0.1)),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes], stddev=0.1))
}
biases = {
    'b1': tf.Variable(tf.zeros([n_hidden_1])),
    'b2': tf.Variable(tf.zeros([n_hidden_2])),
    'out': tf.Variable(tf.zeros([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(tf_reward)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    env = gym.make('Pendulum-v0')

    # Training cycle
    for epoch in range(training_epochs):
        total_reward = 0.0

        for i_episode in range(1):
            env.reset()
            observation, reward, done, info = env.step(env.action_space.sample())
            for t in range(gym_iterations):
                # Run optimization op (backprop) and cost op (to get loss value)
                net_output = sess.run([pred], feed_dict={x: observation.reshape(1,3)})
                #print(net_output)
                #if(net_output[0] > net_output[1]): action=0
                #else: action=1
                #action = np.argmax(net_output) #env.action_space.sample()
                #print(action)
                action= np.array([net_output[0][0]])
                observation, reward, done, info = env.step(action) # take a random action
                total_reward += -reward
                #tf.assign(tf_reward, reward)
                if done: break

        assign_op = tf.assign(tf_reward, total_reward)
        sess.run(assign_op)
        print("Epoch: " + str(epoch) + "; Total Reward: " + str(total_reward) + "; Reward: " + str(tf_reward.eval()))
        sess.run([optimizer])
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch: " + str(epoch) + "; Total Reward: " + str(total_reward) + "; Reward: " + str(tf_reward.eval()))

    print("Optimization Finished!")
