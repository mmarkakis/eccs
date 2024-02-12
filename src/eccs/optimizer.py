import pandas as pd
import networkx as nx
from gurobipy import Model, GRB, quicksum
import numpy as np
from typing import Optional


class Optimizer:
    """A class that addresses the problem of finding the ATE-change-maximizing graph."""

    def __init__(
        self,
        data: pd.DataFrame,
        treatment_idx: int,
        outcome_idx: int,
    ) -> None:
        """Initialize the Optimizer object.

        Parameters:
            data: The data to be used for causal analysis.
            treatment_idx: The index of the treatment variable.
            outcome_idx: The index of the outcome variable.
        """
        self._data = data
        self._treatment_idx = treatment_idx
        self._outcome_idx = outcome_idx

    def optimize(
        self,
        ATE_init: float,
        excl_set: list[int],
        incl_set: list[int],
        lambda_1: float,
        lambda_2: float,
    ) -> Optional[list[int]]:
        """Find the ATE-change-maximizing adjustment set.

        Parameters:
            ATE_init: The initial ATE value.
            excl_set: The set of variables to defintiely exclude from the adjustment set.
            incl_set: The set of variables to definitely include in the adjustment set.
            lambda_1: The weight for the sparsity penalty.
            lambda_2: The weight for the ATE difference penalty.

        Returns:
            The set of variables that should be included in the adjustment set to maximize the change in ATE.
        """

        y = self._data.iloc[:, self._outcome_idx].values
        X = (
            self._data
        )  # Keep y here to avoid changing the variable indexing. We ensure that we don't actually regress y
        # on itself by adding it to the excl_set later on.

        # Initialize model
        model = Model("SparseRegression")

        # Decision variables
        N = self._data.shape[0]
        M = self._data.shape[1]
        beta = model.addVars(M, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="beta")
        A = model.addVars(M, vtype=GRB.BINARY, name="A")

        # Update model to integrate new variables
        model.update()

        # Objective function components
        # Define the MSE part of the objective
        MSE = (
            quicksum(
                (y[i] - quicksum(beta[j] * A[j] * X[i, j] for j in range(M))) ** 2
                for i in range(N)
            )
            / N
        )

        # Penalty for the distance of beta_1 from ATE_init
        ATE_diff = (ATE_init - beta[self._treatment_idx]) * (
            ATE_init - beta[self._treatment_idx]
        )

        # Sparsity penalty
        non_free_vars = (
            excl_set.union(incl_set).union(self._outcome_idx).union(self._treatment_idx)
        )
        sparsity_penalty = quicksum(A[j] for j in range(M) if j not in non_free_vars)

        # Set objective
        model.setObjective(
            MSE + lambda_1 * sparsity_penalty - lambda_2 * ATE_diff, GRB.MINIMIZE
        )

        # Constraints for A_j based on excl_set and incl_set
        for j in excl_set.union(self._outcome_idx):
            model.addConstr(A[j] == 0, "Exclude_%d" % j)

        for j in incl_set.union(self._treatment_idx):
            model.addConstr(A[j] == 1, "Include_%d" % j)

        # Solve model
        model.optimize()

        # Extract solution (if model is solvable)
        # We want the indices of the nonzero A's
        if model.status == GRB.OPTIMAL:
            return [j for j in range(M) if A[j].x > 0.5 and j != self._treatment_idx]
        else:
            return None
