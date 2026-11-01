"""Immutable multiplayer Liar's Dice engine for a precisely specified rules variant."""
from dataclasses import dataclass, replace
from random import Random


@dataclass(frozen=True, order=True)
class Bid:
    quantity: int
    face: int

    def __post_init__(self):
        if type(self.quantity) is not int or self.quantity < 1:
            raise ValueError('quantity must be a positive integer')
        if type(self.face) is not int or not 1 <= self.face <= 6:
            raise ValueError('face must be an integer from 1 to 6')


@dataclass(frozen=True)
class Call:
    pass


CALL = Call()
Action = Bid | Call


@dataclass(frozen=True)
class Rules:
    players: int = 2
    dice: int = 2
    ones_wild: bool = True

    def __post_init__(self):
        if type(self.players) is not int or not 2 <= self.players <= 10:
            raise ValueError('players must be 2..10')
        if type(self.dice) is not int or not 1 <= self.dice <= 10:
            raise ValueError('dice must be 1..10')
        if type(self.ones_wild) is not bool:
            raise ValueError('ones_wild must be bool')


@dataclass(frozen=True)
class State:
    hands: tuple[tuple[int, ...], ...]
    turn: int
    last_bid: Bid | None = None
    last_bidder: int | None = None
    history: tuple[tuple[int, Bid], ...] = ()
    round_number: int = 1

    @property
    def counts(self):
        return tuple(map(len, self.hands))

    @property
    def terminal(self):
        return sum(n > 0 for n in self.counts) == 1

    @property
    def winner(self):
        return next((i for i, n in enumerate(self.counts) if n), None) if self.terminal else None


@dataclass(frozen=True)
class Observation:
    player: int
    own_hand: tuple[int, ...]
    counts: tuple[int, ...]
    last_bid: Bid | None
    last_bidder: int | None
    history: tuple[tuple[int, Bid], ...]
    legal_actions: tuple[Action, ...]
    ones_wild: bool


@dataclass(frozen=True)
class Challenge:
    bidder: int
    caller: int
    bid: Bid
    actual_count: int
    winner: int
    loser: int
    revealed_hands: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class Transition:
    state: State
    rewards: tuple[int, ...]
    challenge: Challenge | None = None


def next_active(counts: tuple[int, ...], after: int) -> int:
    for offset in range(1, len(counts) + 1):
        candidate = (after + offset) % len(counts)
        if counts[candidate] > 0:
            return candidate
    raise ValueError('no active players')


def _roll(counts, rng):
    return tuple(tuple(rng.randint(1, 6) for _ in range(n)) for n in counts)


def reset(rules: Rules, rng: Random, starter: int = 0) -> State:
    if type(starter) is not int or not 0 <= starter < rules.players:
        raise ValueError('starter outside player range')
    return State(_roll((rules.dice,) * rules.players, rng), starter)


def validate_state(rules: Rules, state: State) -> None:
    if type(state.hands) is not tuple or any(type(h) is not tuple for h in state.hands):
        raise ValueError('hands must be immutable tuples')
    if type(state.turn) is not int or type(state.round_number) is not int or state.round_number < 1:
        raise ValueError('turn and round number must be integers; round positive')
    if type(state.history) is not tuple or any(type(e) is not tuple or len(e) != 2 or type(e[0]) is not int or not isinstance(e[1], Bid) for e in state.history):
        raise ValueError('history must contain immutable (player, bid) tuples')
    if state.last_bid is not None and not isinstance(state.last_bid, Bid):
        raise ValueError('invalid bid type')
    if len(state.hands) != rules.players:
        raise ValueError('wrong player count')
    if any(len(hand) > rules.dice or any(type(d) is not int or not 1 <= d <= 6 for d in hand) for hand in state.hands):
        raise ValueError('invalid hand')
    if not any(state.counts) or not 0 <= state.turn < rules.players or not state.counts[state.turn]:
        raise ValueError('turn must belong to active player')
    if (state.last_bid is None) != (state.last_bidder is None):
        raise ValueError('bid and bidder must be present together')
    if state.last_bid is not None:
        if type(state.last_bidder) is not int or not 0 <= state.last_bidder < rules.players or not state.counts[state.last_bidder]:
            raise ValueError('invalid bidder')
        if state.turn != next_active(state.counts, state.last_bidder) or state.turn == state.last_bidder:
            raise ValueError('turn must follow the last bidder')
        if state.last_bid.quantity > sum(state.counts):
            raise ValueError('bid exceeds dice in play')


def legal_actions(rules: Rules, state: State) -> tuple[Action, ...]:
    validate_state(rules, state)
    if state.terminal:
        return ()
    bids = tuple(Bid(q, f) for q in range(1, sum(state.counts) + 1) for f in range(1, 7)
                 if state.last_bid is None or Bid(q, f) > state.last_bid)
    return bids if state.last_bid is None else bids + (CALL,)


def observe(rules: Rules, state: State) -> Observation:
    validate_state(rules, state)
    return Observation(state.turn, state.hands[state.turn], state.counts, state.last_bid,
                       state.last_bidder, state.history, legal_actions(rules, state), rules.ones_wild)


def matching_count(hands, bid: Bid, ones_wild: bool) -> int:
    return sum(d == bid.face or (ones_wild and bid.face != 1 and d == 1)
               for hand in hands for d in hand)


def step(rules: Rules, state: State, action: Action, rng: Random) -> Transition:
    """One player's move. Rewards are +1/-1 to challenge winner/loser, zero otherwise."""
    if action not in legal_actions(rules, state):
        raise ValueError('illegal action (or terminal state)')
    if isinstance(action, Bid):
        sp = replace(state, turn=next_active(state.counts, state.turn), last_bid=action,
                     last_bidder=state.turn, history=state.history + ((state.turn, action),))
        return Transition(sp, (0,) * rules.players)
    bid, bidder, caller = state.last_bid, state.last_bidder, state.turn
    count = matching_count(state.hands, bid, rules.ones_wild)
    winner, loser = (bidder, caller) if count >= bid.quantity else (caller, bidder)
    counts = list(state.counts)
    counts[loser] -= 1
    rewards = tuple(1 if i == winner else -1 if i == loser else 0 for i in range(rules.players))
    challenge = Challenge(bidder, caller, bid, count, winner, loser, state.hands)
    # The survivor is the terminal turn. Otherwise start strictly after the loser.
    starter = next_active(tuple(counts), loser)
    sp = State(_roll(counts, rng), starter, round_number=state.round_number + 1)
    return Transition(sp, rewards, challenge)
