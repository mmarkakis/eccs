import numpy as np


class MAB:
    """Implements the multi-armed bandit part of selecting how to modify the graph next."""

    def __init__(self, num_arms: int, epsilon: float) -> None:
        """Initialize the MAB object.

        Parameters:
            num_arms: The number of arms in the bandit.
            epsilon: The epsilon value for the epsilon-greedy algorithm.
        """
        self._n = num_arms
        self._rewards = np.zeros(
            (self._n, 2)
        )  # 2 columns: 1st for the number of times the arm was pulled, 2nd for the average reward

    def select_arm(self) -> int:
        """Select an arm according to the epsilon-greedy algorithm.

        Returns:
            The index of the arm to select.
        """
        if np.random.random() < self._epsilon:
            return np.random.randint(self._n)
        else:
            return np.argmax(self._rewards[:, 1])

    def update(self, arm: int, reward: float) -> None:
        """Update the rewards for the selected arm.

        Parameters:
            arm: The index of the arm that was selected.
            reward: The reward for selecting that arm.
        """
        n, r = self._rewards[arm]
        self._rewards[arm] = (n + 1, (n * r + reward) / (n + 1))
