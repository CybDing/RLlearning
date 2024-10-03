from typing import Optional, List
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt


class Bandit:
    # class docstring here
    def __init__(self, arms, action_value: np.array, initial: np.array = None, Sample_Mean=False,
                 Const_Stepsize=False, Gradient=False, UCB=False, alpha: Optional[float] = None, var_value=10., eps=.0):
        self.Arms = arms
        assert len(action_value) == self.Arms
        self.Action_Value = action_value
        if initial is None:
            self.Initial = np.zeros(self.Arms)
        else:
            assert len(initial) == self.Arms
            self.Initial = initial
        self.Eps = eps
        if Const_Stepsize or Gradient or UCB:
            assert alpha is not None
            self.Alpha = alpha
        elif Sample_Mean:
            self.Alpha = None
        self.Sample_Mean = Sample_Mean
        self.UCB = UCB
        self.Const_Stepsize = Const_Stepsize
        self.Gradient = Gradient
        self.indices = np.arange(0, self.Arms)
        self.Time = 1
        self.Action_Time = list(np.ones(self.Arms))
        self.Average_Reward = 0
        self.q_True = None
        self.q_Est = self.Initial
        self.q_Best_Bandit = None
        self.Var_Value = var_value
        self.H_t = None

    def Reset(self):
        self.Time = 1
        self.Action_Time = np.ones(self.Arms)
        self.Average_Reward = 0
        self.q_True = self.Action_Value + np.random.normal(0, self.Var_Value, self.Arms)
        self.q_Est = self.Initial
        self.q_Best_Bandit = np.argmax(self.q_True)

    def Reward(self, ind) -> float:
        return self.q_True[ind] + np.random.normal(0, 2)

    def Decision_making(self) -> int:
        if np.random.rand() < self.Eps:
            return np.random.choice(self.indices)
        q_t = None
        if self.Gradient:
            self.H_t = np.exp(self.q_Est)
            self.H_t = self.H_t / np.sum(self.H_t)
            return np.random.choice(self.indices, p=self.H_t)
        if self.Sample_Mean or self.Const_Stepsize:
            q_t = self.q_Est
        if self.UCB:
            q_t = [self.q_Est[i] + self.Alpha * np.sqrt(self.Time / self.Action_Time[i]) for i in range(self.Arms)]
        return np.argmax(q_t)

    def Update(self, action: int) -> float:
        rewards = self.Reward(action)
        self.Action_Time[action] += 1
        self.Average_Reward += 1 / self.Time * (rewards - self.Average_Reward)
        self.Time += 1
        if self.Sample_Mean or self.UCB:
            self.q_Est[action] += 1 / self.Action_Time[action] * (rewards - self.q_Est[action])
        if self.Const_Stepsize:
            self.q_Est[action] += self.Alpha * (rewards - self.q_Est[action])
        if self.Gradient:
            self.q_Est[action] += self.Alpha * (rewards - self.Average_Reward) * (1 - self.H_t[action])
            for i in range(self.Arms):
                if i != action:
                    self.q_Est[action] -= self.Alpha * (rewards - self.Average_Reward) * self.H_t[action]
        return rewards


def Simulation(runs: int, steps: int, bandits: List[Bandit]) -> list:
    rewards = np.zeros((len(bandits), runs, steps))
    best_action_value = np.zeros((len(bandits), runs, steps))
    for i, bandit in enumerate(bandits):
        for r in trange(runs):
            bandit.Reset()
            # print(bandit.q_Best_Bandit)
            # print(bandit.q_True)
            for s in range(steps):
                decision = bandit.Decision_making()
                reward = bandit.Update(decision)
                rewards[i, r, s] = reward
                # print(decision)
                # print(bandit.q_Est)
                if decision == bandit.q_Best_Bandit:
                    best_action_value[i, r, s] = 1
    average_best_action = np.mean(best_action_value, axis=1)
    average_rewards = np.mean(rewards, axis=1)
    return [average_best_action, average_rewards]


def SampleAverage(r=1000, t=20000):
    bandit = []
    bandit.append(Bandit(arms=10, action_value=np.zeros(10), initial=np.zeros(10), var_value=10, Sample_Mean=True,
                         eps=0.1))
    # bandit.append(Bandit(arms=10, action_value=np.zeros(10), initial=np.zeros(10), var_value=10, Sample_Mean=True,
    #                      eps=0.2))
    bandit.append(Bandit(arms=10, action_value=np.zeros(10), initial=np.zeros(10), var_value=10, UCB=True,
                         eps=0.1, alpha=0.1))
    Result = Simulation(runs=r, steps=t, bandits=bandit)
    average_best_action = Result[0]
    average_rewards = Result[1]
    plt.figure(figsize=(10, 10))
    plt.plot(np.arange(1, 20000 + 1), average_best_action[0, :], label='eps=0.1')
    plt.plot(np.arange(1, 20000 + 1), average_best_action[1, :], label='alpha=0.1')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    SampleAverage()
