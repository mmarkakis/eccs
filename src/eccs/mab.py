import numpy as np


class MAB:
    """Implements the multi-armed bandit part of selecting how to modify the graph next."""

    def __init__(self, num_arms: int, epsilon: float, banlist: list[int]) -> None:
        """Initialize the MAB object.

        Parameters:
            num_arms: The number of arms in the bandit.
            epsilon: The epsilon value for the epsilon-greedy algorithm.
            banlist: The list of arms that are banned from being selected.
        """
        self._n = num_arms
        self._rewards = {
            i: (0, 0) for i in range(self._n) if i not in banlist
        }  # 2 columns: 0th for the number of times the arm was pulled, 1st for the average reward
        self._epsilon = epsilon

    def select_arm(self) -> int:
        """Select an arm according to the epsilon-greedy algorithm.

        Returns:
            The index of the arm to select.
        """
        if np.random.random() < self._epsilon:
            # Randomly select an arm
            return np.random.choice(list(self._rewards.keys()))
        else:
            # Select the arm with the highest average reward
            return max(self._rewards, key=lambda x: self._rewards[x][1])
          
    def update(self, arm: int, reward: float) -> None:
        """Update the rewards for the selected arm.

        Parameters:
            arm: The index of the arm that was selected.
            reward: The reward for selecting that arm.
        """
        n, avg = self._rewards[arm]
        self._rewards[arm] = (n + 1, (n * avg + reward) / (n + 1))