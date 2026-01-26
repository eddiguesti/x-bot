"""Glicko-2 rating system implementation for ranking creators."""

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class Glicko2Rating:
    """
    Glicko-2 rating components.

    Attributes:
        rating: Skill estimate (mu), default 1500
        rd: Rating deviation (uncertainty), default 350
        volatility: Rating volatility (sigma), default 0.06
    """
    rating: float = 1500.0
    rd: float = 350.0
    volatility: float = 0.06

    def to_glicko1(self) -> tuple[float, float]:
        """Convert to Glicko-1 scale for display."""
        return (
            self.rating * 173.7178 + 1500,
            self.rd * 173.7178
        )

    @property
    def confidence_interval(self) -> tuple[float, float]:
        """95% confidence interval for true skill."""
        return (
            self.rating - 2 * self.rd,
            self.rating + 2 * self.rd
        )

    @property
    def weight(self) -> float:
        """
        Calculate weight for consensus voting.

        Higher rating and lower RD = higher weight.
        """
        # Normalize rating to 0-1 scale
        base_weight = max(0, (self.rating - 1000) / 1000)

        # Penalize high uncertainty
        confidence = max(0.1, 1 - (self.rd / 350))

        return max(0.01, base_weight * confidence)


class Glicko2Calculator:
    """
    Glicko-2 rating calculator.

    Based on Mark Glickman's paper:
    "Example of the Glicko-2 system"
    http://www.glicko.net/glicko/glicko2.pdf
    """

    # System constant (constrains volatility change)
    TAU = 0.5

    # Convergence tolerance
    EPSILON = 0.000001

    # Scale factor between Glicko-1 and Glicko-2
    SCALE = 173.7178

    def __init__(self, tau: float = 0.5):
        """
        Initialize calculator.

        Args:
            tau: System constant, smaller = less volatility change
        """
        self.tau = tau

    def update_rating(
        self,
        player: Glicko2Rating,
        outcomes: list[tuple[bool, Optional[Glicko2Rating]]],
    ) -> Glicko2Rating:
        """
        Update a player's rating based on match outcomes.

        For our use case, a "match" is a prediction, and we treat the
        market as an opponent with fixed rating (1500, low RD).

        Args:
            player: Current player rating
            outcomes: List of (won, opponent_rating) tuples
                      For predictions, opponent is None (uses market baseline)

        Returns:
            Updated Glicko2Rating
        """
        if not outcomes:
            # No games - only increase RD for inactivity
            return self._apply_inactivity(player)

        # Convert to Glicko-2 scale
        mu = (player.rating - 1500) / self.SCALE
        phi = player.rd / self.SCALE
        sigma = player.volatility

        # Calculate v and delta
        v_sum = 0.0
        delta_sum = 0.0

        for won, opponent in outcomes:
            # Use market baseline if no opponent specified
            if opponent is None:
                opp_mu = 0.0  # 1500 in original scale
                opp_phi = 200 / self.SCALE  # Lower RD = more reliable opponent
            else:
                opp_mu = (opponent.rating - 1500) / self.SCALE
                opp_phi = opponent.rd / self.SCALE

            g_phi = self._g(opp_phi)
            e_val = self._E(mu, opp_mu, opp_phi)

            v_sum += g_phi ** 2 * e_val * (1 - e_val)

            score = 1.0 if won else 0.0
            delta_sum += g_phi * (score - e_val)

        v = 1 / v_sum if v_sum > 0 else 1000
        delta = v * delta_sum

        # Calculate new volatility
        new_sigma = self._compute_volatility(sigma, phi, v, delta)

        # Update phi (pre-rating period)
        phi_star = math.sqrt(phi ** 2 + new_sigma ** 2)

        # Update phi and mu
        new_phi = 1 / math.sqrt(1 / phi_star ** 2 + 1 / v)
        new_mu = mu + new_phi ** 2 * delta_sum

        # Convert back to original scale
        return Glicko2Rating(
            rating=new_mu * self.SCALE + 1500,
            rd=new_phi * self.SCALE,
            volatility=new_sigma
        )

    def update_single(
        self,
        player: Glicko2Rating,
        won: bool,
    ) -> Glicko2Rating:
        """
        Convenience method for single prediction outcome.

        Args:
            player: Current rating
            won: Whether prediction was correct

        Returns:
            Updated rating
        """
        return self.update_rating(player, [(won, None)])

    def _apply_inactivity(self, player: Glicko2Rating) -> Glicko2Rating:
        """Increase RD for inactive period."""
        phi = player.rd / self.SCALE
        sigma = player.volatility

        new_phi = math.sqrt(phi ** 2 + sigma ** 2)

        return Glicko2Rating(
            rating=player.rating,
            rd=min(350, new_phi * self.SCALE),  # Cap at initial RD
            volatility=sigma
        )

    def _g(self, phi: float) -> float:
        """G function - reduces impact of uncertain opponents."""
        return 1 / math.sqrt(1 + 3 * phi ** 2 / math.pi ** 2)

    def _E(self, mu: float, mu_j: float, phi_j: float) -> float:
        """Expected score function."""
        return 1 / (1 + math.exp(-self._g(phi_j) * (mu - mu_j)))

    def _compute_volatility(
        self,
        sigma: float,
        phi: float,
        v: float,
        delta: float
    ) -> float:
        """
        Compute new volatility using Illinois algorithm.

        This is the most complex part of Glicko-2.
        """
        a = math.log(sigma ** 2)
        delta_sq = delta ** 2
        phi_sq = phi ** 2

        def f(x):
            ex = math.exp(x)
            num1 = ex * (delta_sq - phi_sq - v - ex)
            denom1 = 2 * (phi_sq + v + ex) ** 2
            return num1 / denom1 - (x - a) / self.tau ** 2

        # Find bounds
        A = a
        if delta_sq > phi_sq + v:
            B = math.log(delta_sq - phi_sq - v)
        else:
            k = 1
            while f(a - k * self.tau) < 0:
                k += 1
            B = a - k * self.tau

        # Illinois algorithm
        fA = f(A)
        fB = f(B)

        while abs(B - A) > self.EPSILON:
            C = A + (A - B) * fA / (fB - fA)
            fC = f(C)

            if fC * fB <= 0:
                A = B
                fA = fB
            else:
                fA = fA / 2

            B = C
            fB = fC

        return math.exp(A / 2)

    def expected_score(
        self,
        player: Glicko2Rating,
        opponent: Optional[Glicko2Rating] = None
    ) -> float:
        """
        Calculate expected score against opponent.

        Args:
            player: Player rating
            opponent: Opponent rating (uses market baseline if None)

        Returns:
            Expected score (0 to 1)
        """
        mu = (player.rating - 1500) / self.SCALE

        if opponent is None:
            opp_mu = 0.0
            opp_phi = 200 / self.SCALE
        else:
            opp_mu = (opponent.rating - 1500) / self.SCALE
            opp_phi = opponent.rd / self.SCALE

        return self._E(mu, opp_mu, opp_phi)
