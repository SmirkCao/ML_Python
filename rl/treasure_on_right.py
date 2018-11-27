# ref to:
# https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/
# all in one demo to rl
import logging
import pandas as pd
import numpy as np


class RL(object):
    def __init__(self,
                 max_episodes=13,
                 actions=[],
                 alpha=0.1,
                 gamma=0.9,
                 env=None):
        self.max_episodes = max_episodes
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.env = env

    def choose_action(self, s):
        raise NotImplemented

    def learn(self):
        raise NotImplemented


class EpsilonGreddyQ(RL):
    def __init__(self,
                 max_episodes=13,
                 actions=[],
                 alpha=0.1,
                 epsilon=0.1,
                 gamma=0.9,
                 n_states=None,
                 env=None):
        super().__init__(max_episodes=max_episodes,
                         actions=actions,
                         alpha=alpha,
                         gamma=gamma,
                         env=env)
        self.epsilon = epsilon
        self.q_table = pd.DataFrame(np.zeros((n_states, len(self.actions))),
                                    columns=self.actions)

    def choose_action(self, s):
        # epsilon greedy
        # epsilon = 0, greedy
        state_actions = self.q_table.iloc[s, :]
        if (np.random.uniform() < self.epsilon) or ((state_actions == 0).all()):
            # exploring
            a = np.random.choice(self.actions)
        else:
            # exploiting
            a = state_actions.idxmax()  # greedy
        return a

    def learn(self):
        if self.env is None:
            raise ValueError
        np.random.seed(2016)
        for episode in range(self.max_episodes):
            step_counter = 0
            s = 0
            is_terminated = False
            # update_env(s, episode, step_counter)
            while not is_terminated:
                a = self.choose_action(s)
                s_, r = self.env.get_env_feedback(s, a)

                q_pred = self.q_table.loc[s, a]

                if s_ != "terminal":
                    q_target = r + self.gamma * self.q_table.iloc[s_, :].max()
                else:
                    q_target = r
                    is_terminated = True

                self.q_table.loc[s, a] += self.alpha * (q_target - q_pred)
                s = s_

                step_counter += 1
                # update_env(s, episode, step_counter)
            logger.info("episode: %d, step: %d" % (episode, step_counter))

        return self.q_table


class Environment(object):
    def __init__(self,
                 states=None,
                 actions=None):
        if states is None:
            raise ValueError
        else:
            self.states = states
        if actions is None:
            raise ValueError
        else:
            self.actions = actions

    def get_env_feedback(self, s, a):
        raise NotImplemented

    def update(self):
        pass


class TreasureOnRight(Environment):
    def get_env_feedback(self, s, a):
        """

           :param s: state
           :param a: action
           :return: r: reward{terminal: 1, otherwise: 0}
           """

        if a == "right":
            if s == self.states - 2:
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


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    E = TreasureOnRight(states=6, actions=["left", "right"])
    Q = EpsilonGreddyQ(actions=["left", "right"], n_states=6, env=E)
    q_table = Q.learn()
    logger.info(q_table)
