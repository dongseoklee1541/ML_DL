"""
Deep Q-Network: Mountain Car에 적용한 DQN 예제
"""

import numpy as np
import tensorflow as tf
import gym

class DQN:
    def __init__(self, learning_rate, gamma, n_features, n_actions, epsilon, parameter_changing_pointer, memory_size):
        self.learning_rate = learning_rate  #학습률
        self.gamma = gamma            #할인율
        self.n_features = n_features  #자동차의 위치(0)와 속도(1)
        self.n_actions = n_actions    #왼쪽으로 밀기(0), 보류(1), 오른쪽으로 밀기(2) 
        self.epsilon = epsilon        #탐욕 정책 시 활용되는 탐욕의 초기 값       
        self.batch_size = 100         #재생 메모리로부터 추출되는 표본의 크기
        self.experience_counter = 0   #현재 재생 메모리에 저장된 표본의 수
        self.experience_limit = memory_size  #재생 메모리의 최대 용량
        self.replace_target_pointer = parameter_changing_pointer  #target network 갱신 기준 학습 단계
        self.learning_counter = 0                                 #primary network의 학습 단계
        self.memory = np.zeros([self.experience_limit,self.n_features*2+2])  #재생 메모리의 초기값  

        self.build_networks() #primary network과 target network을 생성
        p_params = tf.get_collection('primary_network_parameters')
        t_params = tf.get_collection('target_network_parameters')
        self.replacing_target_parameters = [tf.assign(t,p) for t,p in zip(t_params,p_params)]

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

#========== DQN 모형의 기본 신경망 및 목표 신경망을 설정하는 단계 ============#

    def build_networks(self):
        hidden_units = 10
#.....................................................................#
        # Primary Network: 각 10개의 은닉노드를 갖는 2개의 은닉층 
        self.s = tf.placeholder(tf.float32,[None,self.n_features])
        self.qtarget = tf.placeholder(tf.float32,[None,self.n_actions])

        with tf.variable_scope('primary_network'): #변수 볌위를 관리한다.
            c = ['primary_network_parameters', 
					tf.GraphKeys.GLOBAL_VARIABLES]
            # first layer
            with tf.variable_scope('layer1'):
                w1 = tf.get_variable('w1', [self.n_features, hidden_units],
			initializer=tf.contrib.layers.xavier_initializer(),
                                     dtype=tf.float32,collections=c)

                b1 = tf.get_variable('b1', [1, hidden_units],
			initializer=tf.contrib.layers.xavier_initializer(),
                                     dtype=tf.float32,collections=c)

                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            # second layer
            with tf.variable_scope('layer2'):
                w2 = tf.get_variable('w2', [hidden_units, self.n_actions],
			initializer=tf.contrib.layers.xavier_initializer(),
                                     dtype=tf.float32,collections=c)

                b2 = tf.get_variable('b2', [1, self.n_actions],
			initializer=tf.contrib.layers.xavier_initializer(),
                                     dtype=tf.float32,collections=c)

                self.qeval = tf.matmul(l1, w2) + b2

        with tf.variable_scope('loss'):
                self.loss = tf.reduce_mean(tf.squared_difference(
						self.qtarget, self.qeval))
        with tf.variable_scope('optimiser'):
                self.train = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

#.....................................................................#
        # Target Network
        self.st = tf.placeholder(tf.float32,[None,self.n_features])

        with tf.variable_scope('target_network'):
            c = ['target_network_parameters', tf.GraphKeys.GLOBAL_VARIABLES]
            # first layer
            with tf.variable_scope('layer1'):
                w1 = tf.get_variable('w1', [self.n_features, hidden_units],
				initializer=tf.contrib.layers.xavier_initializer(),
                                     dtype=tf.float32,collections=c)

                b1 = tf.get_variable('b1', [1, hidden_units],
				initializer=tf.contrib.layers.xavier_initializer(),
                                     dtype=tf.float32,collections=c)

                l1 = tf.nn.relu(tf.matmul(self.st, w1) + b1)

            # second layer
            with tf.variable_scope('layer2'):
                w2 = tf.get_variable('w2', [hidden_units, self.n_actions],
				initializer=tf.contrib.layers.xavier_initializer(),
                                     dtype=tf.float32,collections=c)

                b2 = tf.get_variable('b2', [1, self.n_actions],
				initializer=tf.contrib.layers.xavier_initializer(),
                                     dtype=tf.float32,collections=c)

                self.qt = tf.matmul(l1, w2) + b2

#-----------------------------------------------
#기본 신경망의 학습 결과를 목표 신경망에 대입하기 위한 세션을 구성한다
    def target_params_replaced(self):
        self.sess.run(self.replacing_target_parameters)

    def store_experience(self,obs,a,r,obs_):
        index = self.experience_counter % self.experience_limit
        self.memory[index,:] = np.hstack((obs,[a,r],obs_))
        self.experience_counter+=1

#---------------------------------------------------------------
#재생 메모리에 저장된 과거 경험을 설정한 배치 크기만큼 랜덤으로 추출하여 학습 데이터 집합으로 설정한다.
    def fit(self):
        # sample batch memory from all memory
        if self.experience_counter < self.experience_limit:
            indices = np.random.choice(self.experience_counter, size=self.batch_size)
        else:
            indices = np.random.choice(self.experience_limit, size=self.batch_size)

        batch = self.memory[indices,:]
        qt,qeval = self.sess.run([self.qt,self.qeval],
	feed_dict={self.st:batch[:,-self.n_features:],self.s:batch[:,:self.n_features]})

        qtarget = qeval.copy()    
        batch_indices = np.arange(self.batch_size, dtype=np.int32)
        actions = self.memory[indices,self.n_features].astype(int)
        rewards = self.memory[indices,self.n_features+1]
        qtarget[batch_indices,actions] = rewards + self.gamma * np.max(qt,axis=1)

        _ = self.sess.run(self.train,feed_dict = {self.s:batch[:,:self.n_features],
							self.qtarget:qtarget})
        if self.epsilon < 0.9:
            self.epsilon += 0.0002

#---------------------------------------------------------------
#학습 시 기본 신경망의 가중치를 가져와 목표 신경망의 가중치를 갱신한다.
        if self.learning_counter % self.replace_target_pointer == 0:
            self.target_params_replaced()
            print("target parameters changed")
        self.learning_counter += 1

#---------------------------------------------------------------
#탐욕 정책을 통해 행동을 선택하는 함수를 정의한다.
    def epsilon_greedy(self,obs):
        #epsilon greedy implementation to choose action
        if np.random.uniform(low=0,high=1) < self.epsilon:
            return np.argmax(self.sess.run(self.qeval,
					feed_dict={self.s:obs[np.newaxis,:]}))
        else:
            return np.random.choice(self.n_actions)

#---------------------------------------------------------------
#DQN 객체를 생성해 에이전트를 학습시키고 결과를 도출하는 단계
if __name__ == "__main__":
    env = gym.make('MountainCar-v0')
    env = env.unwrapped
    dqn = DQN(learning_rate=0.001, gamma=0.9, n_features=env.observation_space.shape[0], 	 n_actions=env.action_space.n, epsilon=0.0, parameter_changing_pointer=500, 
	 memory_size=5000)

    episodes = 10
    total_steps = 0

    for episode in range(episodes):
        steps = 0		
        obs = env.reset()
        episode_reward = 0
        while True:
            env.render()
            action = dqn.epsilon_greedy(obs)
            obs_,reward,terminate,_ = env.step(action)
            reward = abs(obs_[0]+0.5)
            dqn.store_experience(obs,action,reward,obs_)
            if total_steps > 1000:
                dqn.fit()
            episode_reward+=reward
            if terminate:
                break
            obs = obs_
            total_steps+=1
            steps+=1
        print("Episode {} with Reward : {} at epsilon {} in steps {}".
			format(episode+1,episode_reward,dqn.epsilon,steps))

    while True:  
        env.render()	

