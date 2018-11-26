# ref to:
# https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/tree/master/contents/1_command_line_reinforcement_learning
# all in one demo to rl
import logging
import pandas as pd
import numpy as np

N_STATES = 6
ACTIONS = ["left", "right"]
ALPHA = 0.1  # learning rate
MAX_EPISODES = 13
EPSILON = 0.1  # epsilon-greddy
GAMMA = 0.9


def choose_action(s, q_table):
    # epsilon = 0, greedy
    state_actions = q_table.iloc[s, :]
    if (np.random.uniform() < EPSILON) or ((state_actions == 0).all()):
        # exploring
        a = np.random.choice(ACTIONS)
    else:
        # exploiting
        a = state_actions.idxmax()  # greedy
    return a


def get_env_feedback(s, a):
    """

    :param s: state
    :param a: action
    :return: r: reward{terminal: 1, otherwise: 0}
    """

    if a == "right":
        if s == N_STATES - 2:
            next_state = "terminal"
            r = 1
        else:
            next_state = s + 1
            r = 0
    else:
        r = 0
        if s == 0:
            next_state = s
        else:
            next_state = s - 1

    return next_state, r


def update_env(state, episode, step_counter):
    # just for response
    env_list = ["-"] * (N_STATES - 1) + ["T"]


def rl():
    np.random.seed(2)
    q_table = pd.DataFrame(np.zeros((N_STATES, len(ACTIONS))), columns=ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        s = 0
        is_terminated = False
        # update_env(s, episode, step_counter)
        while not is_terminated:
            a = choose_action(s, q_table)
            s_, r = get_env_feedback(s, a)

            q_pred = q_table.loc[s, a]

            if s_ != "terminal":
                q_target = r + GAMMA * q_table.iloc[s_, :].max()
            else:
                q_target = r
                is_terminated = True

            q_table.loc[s, a] += ALPHA * (q_target - q_pred)
            s = s_

            step_counter += 1
            # update_env(s, episode, step_counter)
        logger.info("episode: %d, step: %d" % (episode, step_counter))

    return q_table


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    q_table = rl()
    logger.info(q_table)
