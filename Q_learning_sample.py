import numpy as np
import pandas as pd
import time

np.random.seed(2)

#  距离终点的状态
N_STATES = 6
# 可以选择的操作策略
ACTIONS = ['left','right']
# 贪心策略， 90%选择最优策略，10%选择随机策略
EPSILON = 0.9
# 学习率
ALPHA = 0.1
# 衰减度
LAMBDA = 0.9
# 最大的回合数
MAX_EPISODES = 100
# 每次行为的时间
FRESH_TIME = 0.003

# 建立Q table
def build_q_table(n_states, actions):
    table = pd.DataFrame(np.zeros((n_states,len(actions))),columns = actions)
    print(table)
    return table



def choose_action(state, q_table):
    # this is how to choose a action
    state_actions = q_table.iloc[state,:]
    if(np.random.uniform() > EPSILON) or (state_actions.all() == 0):
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.idxmax()
    return action_name

def get_env_feedback(S,A):
    # This is how agent will interact with the environment
    if A == "right":
        if S == N_STATES - 2:
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = 0
    else:
        R = 0
        if S == 0:
            S_ = S
        else:
            S_ = S - 1
    return S_, R


def update_env(S , episode, step_counter):
    # this is hov environment update
    env_list = ['-']*(N_STATES-1) + ['T']

    if S == 'terminal':
        interaction = 'Episode %s : total_step = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                     ',end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction),end='')
        time.sleep(FRESH_TIME)

def rl():
    # main part of rl
    q_table = build_q_table(N_STATES,ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = 0
        is_terminated = False
        update_env(S,episode,step_counter)
        while not is_terminated:

            A = choose_action(S, q_table)
            S_, R = get_env_feedback(S,A)
            q_predict = q_table.loc[S,A]
            if S_ != 'terminal':
                q_target = R + LAMBDA * q_table.iloc[S_,:].max()
            else:
                q_target = R
                is_terminated = True
            q_table.loc[S,A] += ALPHA * (q_target - q_predict)
            S = S_

            update_env(S, episode,step_counter + 1)
            step_counter += 1
    return q_table

if __name__ == '__main__':
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)
