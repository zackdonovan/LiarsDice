from dataclasses import asdict
from math import sqrt
from random import Random
from .engine import Rules, observe, reset, step
from .policies import POLICIES


def run_game(rules: Rules, names: tuple[str, ...], seed: int, starter: int = 0, trace: bool = False):
    if len(names) != rules.players or any(n not in POLICIES for n in names):
        raise ValueError('one known policy per player required')
    dice_rng = Random(seed)
    policy_rngs = [Random(seed * 1009 + i + 1) for i in range(rules.players)]
    state = reset(rules, dice_rng, starter)
    moves = challenges = 0
    rewards = [0] * rules.players
    events = []
    # At most total_initial_dice-1 rounds, each with <=6*initial_dice bids + call.
    bound = (rules.players * rules.dice - 1) * (6 * rules.players * rules.dice + 1)
    while not state.terminal:
        obs = observe(rules, state)
        action = POLICIES[names[state.turn]](obs, policy_rngs[state.turn])
        result = step(rules, state, action, dice_rng)
        moves += 1
        if trace:
            events.append({'move': moves, 'round': state.round_number, 'player': state.turn,
                           'action': asdict(action) or {'call': True},
                           'challenge': asdict(result.challenge) if result.challenge else None})
        if result.challenge:
            challenges += 1
        rewards = [x + y for x, y in zip(rewards, result.rewards)]
        state = result.state
        if moves > bound:
            raise AssertionError('monotone bidding/finite dice termination bound violated')
    result = {'seed': seed, 'starter': starter, 'policies': names, 'winner': state.winner,
              'moves': moves, 'challenges': challenges, 'round_rewards': rewards,
              'final_dice': state.counts}
    if trace:
        result['events'] = events
    return result


def wilson(wins: int, games: int):
    z = 1.959963984540054
    p = wins / games
    denom = 1 + z*z/games
    center = (p + z*z/(2*games))/denom
    half = z*sqrt(p*(1-p)/games + z*z/(4*games*games))/denom
    return [center - half, center + half]


def benchmark(games: int = 2000, seed: int = 20261101, dice: int = 2):
    if type(games) is not int or games < 4 or games % 4:
        raise ValueError('games must be a positive multiple of 4 (at least 4)')
    output = {'rules': asdict(Rules(2, dice, True)), 'games_per_matchup': games,
              'seed_start': seed, 'ci': '95% Wilson binomial interval; descriptive Monte Carlo uncertainty',
              'design': 'independent game seeds; all four (policy seat, starter) combinations balanced',
              'matchups': []}
    for label, a, b in [('random vs random', 'random', 'random'),
                         ('probability vs random', 'probability', 'random')]:
        records = []
        wins = 0
        for i in range(games):
            seat = i % 2
            starter = (i // 2) % 2
            names = (a, b) if seat == 0 else (b, a)
            result = run_game(Rules(2, dice, True), names, seed + i, starter)
            result['focal_seat'] = seat
            records.append(result)
            wins += result['winner'] == seat
        output['matchups'].append({'label': label, 'focal_policy': a, 'opponent': b,
                                  'wins': wins, 'games': games, 'win_rate': wins/games,
                                  'interval': wilson(wins, games),
                                  'mean_moves': sum(r['moves'] for r in records)/games,
                                  'records': records})
    return output
