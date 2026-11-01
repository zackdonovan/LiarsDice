"""Observation-only reference policies. Neither is a learned/equilibrium strategy."""
from math import comb
from random import Random
from .engine import Bid, CALL, Observation


def random_policy(obs: Observation, rng: Random):
    return rng.choice(obs.legal_actions)


def bid_probability(obs: Observation, bid: Bid) -> float:
    """P(bid true | own dice), assuming unseen dice remain independent/fair.

    Does NOT condition on informative opponent bids. This is a baseline prior,
    not a full Bayesian belief state in strategic play.
    """
    own = sum(d == bid.face or (obs.ones_wild and bid.face != 1 and d == 1) for d in obs.own_hand)
    need = bid.quantity - own
    unseen = sum(obs.counts) - len(obs.own_hand)
    if need <= 0:
        return 1.0
    if need > unseen:
        return 0.0
    p = 2 / 6 if obs.ones_wild and bid.face != 1 else 1 / 6
    return sum(comb(unseen, k) * p**k * (1 - p)**(unseen - k) for k in range(need, unseen + 1))


def probability_policy(obs: Observation, rng: Random):
    """Call a bid below 50% prior truth; otherwise choose most plausible legal bid.

    Ties use the lowest bid. Threshold is fixed, not tuned on benchmark games.
    """
    if obs.last_bid is not None and bid_probability(obs, obs.last_bid) < 0.5:
        return CALL
    bids = [a for a in obs.legal_actions if isinstance(a, Bid)]
    if not bids:
        return CALL
    return max(bids, key=lambda a: (bid_probability(obs, a), -a.quantity, -a.face))


POLICIES = {'random': random_policy, 'probability': probability_policy}
