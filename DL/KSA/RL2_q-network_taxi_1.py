import gym
import numpy as np
import random
import tensorflow as tf

#====== 환경 로딩 =======#
env = gym.make('Taxi-v2')

#====== 신경망 구현 =======#
tf.reset_default_graph()

#행동을 선택하는데 사용되는 신경망의 피드-포워드 부분을 구축한다.
inputs = tf.placeholder(shape=[1, env.observation_space.n], dtype=tf.float32)  #1*500 matrix
weights = tf.Variable(tf.random_uniform([env.observation_space.n,env.action_space.n], 0, 0.01))  #500*6 matrix
q_out = tf.matmul(inputs, weights)  #1*6 matrix
predict = tf.argmax(q_out,1)

# 목표 Q값(ext_q)과 예측 Q값(q_out)의 제곱합을 구함으로써 비용을 얻게 된다.
next_q = tf.placeholder(shape=[1,env.action_space.n],dtype=tf.float32)
loss = tf.reduce_sum(tf.square(next_q - q_out))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
loss_update = trainer.minimize(loss)

#====== 신경망 학습하기 =======#
init = tf.global_variables_initializer()

# 학습 관련 파라미터를 설정한다.
gamma = 0.7
epsilon = 0.2
epsilon_decay = .99
episodes = 100

total_epochs = 0
total_rewards = 0

with tf.Session() as sess:
    sess.run(init)
    for episode in range(episodes):
        #환경을 리셋하고 첫번째 새로운 관측값을 얻는다.
        state = env.reset()
        rewards_this_episode = 0
        epochs = 0

        done = False
        
        # q-network 
        while not done:
            #Q-네트워크로부터 (e의 확률로 랜덤한 액션과 함께) 그리디하게 액션을 선택한다.
            action, q = sess.run([predict,q_out], feed_dict={inputs:np.identity(env.observation_space.n)[state:state + 1]})
            if np.random.rand(1) < epsilon:
                action[0] = env.action_space.sample()

            #환경으로부터 새로운 상태와 보상을 얻는다.                
            next_state, reward, done, info = env.step(action[0])
            #새로운 상태를 네트워크에 피드해줌으로써 Q’값을 구한다.
            curr_q = sess.run(q_out, feed_dict = {inputs:np.identity(env.observation_space.n)[next_state:next_state+1]})
            #maxQ'값을 구하고 선택된 행동에 대한 타겟 값을 설정한다.
            max_next_q = np.max(curr_q)
            target_q = q
            target_q[0, action[0]] = reward + gamma * max_next_q

            #타겟과 예측된 Q값을 이용하여 네트워크를 학습시킨다.
            info, new_weights = sess.run([loss_update, weights], feed_dict={inputs:np.identity(env.observation_space.n)[state:state+1], next_q:target_q})
            rewards_this_episode += reward
            state = next_state
            epochs += 1
        #모델을 학습해 나감에 따라 랜덤 액션의 가능성을 줄여간다.    
        epsilon = epsilon * epsilon_decay
        
        total_epochs += epochs
        total_rewards += rewards_this_episode
        
print ("Percent of succesful episodes: " + str(total_rewards/episodes))

