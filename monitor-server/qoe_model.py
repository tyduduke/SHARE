#!/usr/bin/env python3
"""
QoE-QoS Model Implementation

Implementation of the agent-specific QoE models from the SHARE paper (Eq. 1).

For AR/human agents:
    QoE_h = alpha_h * exp(-beta_h * max(0, d - d0)) * (1 - exp(-gamma_h * a))
    where:
        P_i = max(0, d - d0)  -- latency penalty, d0 = 20 ms AR threshold
        Q_i = a               -- accuracy contribution (diminishing returns)

For robot agents:
    QoE_r = alpha_r * exp(-beta_r * max(0, a_min - a)) * (1 - exp(-gamma_r * (1/(d+theta))))
    where:
        P_i = max(0, a_min - a)     -- accuracy penalty
        Q_i = (d + theta)^{-1}      -- latency contribution (diminishing returns)

alpha is set automatically so that the maximum achievable QoE equals 1.0,
ensuring the QoE target of 0.9 is always reachable.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


class QoEQoSModel:
    """
    Agent-specific QoE model based on the IQX hypothesis (SHARE paper §4.2).
    """

    def __init__(self, agent_type: str, params: dict):
        """
        Initialise the QoE model and auto-normalise alpha so max QoE = 1.

        Parameters
        ----------
        agent_type : str
            'human' or 'robot'
        params : dict
            Model hyperparameters.  Required keys vary by agent type:
            Human:  alpha, beta, gamma, d_max (= d0 = 20 ms)
            Robot:  alpha, beta, gamma, a_min, k (= theta)
        """
        self.agent_type = agent_type
        self.params = params

        # Normalise alpha so that the maximum achievable QoE equals 1.0.
        # This guarantees that the QoE target (default 0.9) is reachable.
        # Originally left as an explicit caller responsibility so that alpha
        # could be set manually during grid-search tuning without __init__
        # overwriting it. Moved here once the grid search was finalised.
        # self.normalize_alpha()
        self.normalize_alpha()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_qoe(self, delay: float, accuracy: float) -> float:
        """
        Compute QoE from current delay and accuracy measurements.

        Parameters
        ----------
        delay : float
            End-to-end latency in milliseconds.
        accuracy : float
            SLAM tracking accuracy estimate in centimetres (from DNN evaluator).

        Returns
        -------
        float
            QoE value in [0, 1].
        """
        if self.agent_type == 'human':
            return self._qoe_human(delay, accuracy)
        elif self.agent_type == 'robot':
            return self._qoe_robot(delay, accuracy)
        else:
            raise ValueError(f"Unknown agent type: {self.agent_type}")

    # ------------------------------------------------------------------
    # Internal QoE formulas  (paper §4.2, Eq. 1)
    # ------------------------------------------------------------------

    def _qoe_human(self, delay: float, accuracy: float) -> float:
        """
        QoE_h = alpha_h * exp(-beta_h * max(0, d - d0)) * (1 - exp(-gamma_h * a))

        d0 = 20 ms  (paper's AR motion-to-photon latency threshold, §3.1.2)
        """
        alpha = self.params['alpha']
        beta  = self.params['beta']
        gamma = self.params['gamma']
        d0    = self.params['d_max']   # 20 ms per paper

        # Agent-centric penalty: steep collapse above the 20 ms threshold
        penalty_term = np.exp(-beta * max(0.0, delay - d0))

        # Marginal contribution: accuracy improves QoE with diminishing returns
        accuracy_term = 1.0 - np.exp(-gamma * max(0.0, accuracy))

        # print(['penalty_term', penalty_term, 'accuracy_term', accuracy_term, 'alpha', alpha])
        return float(alpha * penalty_term * accuracy_term)

    def _qoe_robot(self, delay: float, accuracy: float) -> float:
        """
        QoE_r = alpha_r * exp(-beta_r * max(0, a_min - a))
                        * (1 - exp(-gamma_r * (1000 / (d + theta))))

        a_min = 1 cm  (paper's minimum acceptable accuracy threshold)
        theta (k)     normalises latency sensitivity
        """
        alpha = self.params['alpha']
        beta  = self.params['beta']
        gamma = self.params['gamma']
        a_min = self.params['a_min']
        theta = self.params.get('k', 100.0)

        # Agent-centric penalty: collapse when accuracy falls below a_min
        penalty_term = np.exp(-beta * max(0.0, a_min - accuracy))

        # Marginal contribution: latency has diminishing returns for robots
        # Paper uses Q_i = (d + theta)^{-1}; scaled by 1000 for unit consistency
        latency_q    = 1000.0 / (delay + theta)
        latency_term = 1.0 - np.exp(-gamma * latency_q)

        return float(alpha * penalty_term * latency_term)

    # ------------------------------------------------------------------
    # Alpha normalisation (ensures max QoE = 1.0)
    # ------------------------------------------------------------------

    def _max_product(self) -> float:
        """Return the maximum value of (penalty_term * marginal_term)."""
        if self.agent_type == 'human':
            # Maximum when d <= d0 (penalty = 1) and accuracy → large
            max_penalty  = 1.0
            max_marginal = 1.0 - np.exp(-self.params['gamma'] * 1.0)
        else:  # robot
            # Maximum when a >= a_min (penalty = 1) and delay is very small
            theta        = self.params.get('k', 100.0)
            max_penalty  = 1.0
            latency_q    = 1000.0 / (0.0 + theta)          # delay → 0
            max_marginal = 1.0 - np.exp(-self.params['gamma'] * latency_q)

        return max_penalty * max_marginal

    def normalize_alpha(self) -> float:
        """
        Set alpha so that the maximum achievable QoE equals exactly 1.0.
        Called automatically in __init__.
        """
        max_prod = self._max_product()
        if max_prod > 1e-9:
            self.params['alpha'] = 1.0 / max_prod
        else:
            logger.warning("Could not normalise alpha (max product near zero); keeping alpha=1.")
            self.params['alpha'] = 1.0

        logger.debug(f"[{self.agent_type}] alpha set to {self.params['alpha']:.4f} "
                     f"(max product = {max_prod:.4f})")
        return self.params['alpha']
