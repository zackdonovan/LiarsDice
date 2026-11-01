import itertools
import random
import unittest
from dataclasses import fields, replace
from liars_dice.engine import *
from liars_dice.policies import bid_probability, probability_policy
from liars_dice.simulation import run_game, benchmark


class RulesTests(unittest.TestCase):
    def test_invalid_config(self):
        for kwargs in ({'players': 1}, {'dice': 0}, {'dice': True}, {'ones_wild': 1}):
            with self.assertRaises(ValueError): Rules(**kwargs)

    def test_invalid_bid(self):
        for q, f in ((0, 2), (1, 0), (1, 7), (True, 3), (2, 1.2)):
            with self.assertRaises(ValueError): Bid(q, f)

    def test_bid_order(self):
        self.assertGreater(Bid(2, 1), Bid(1, 6))
        self.assertGreater(Bid(1, 6), Bid(1, 5))

    def test_initial_actions_and_mask(self):
        r = Rules(3, 2)
        s = reset(r, random.Random(1))
        self.assertEqual(len(legal_actions(r, s)), 36)
        self.assertNotIn(CALL, legal_actions(r, s))
        t = step(r, s, Bid(4, 3), random.Random(2))
        self.assertEqual(t.state.turn, 1)
        self.assertEqual(len(legal_actions(r, t.state)), 16)
        self.assertEqual(t.state.history, ((0, Bid(4, 3)),))

    def test_invalid_moves_do_not_change_state(self):
        r = Rules()
        s = reset(r, random.Random(2))
        for a in (CALL, Bid(5, 1), 'bid'):
            rng = random.Random(9); before = rng.getstate()
            with self.assertRaises(ValueError): step(r, s, a, rng)
            self.assertEqual(before, rng.getstate())
        t = step(r, s, Bid(2, 3), random.Random(9)).state
        for a in (Bid(1, 6), Bid(2, 3), Bid(2, 2)):
            with self.assertRaises(ValueError): step(r, t, a, random.Random(2))

    def test_wild_count_exhaustive(self):
        for dice in itertools.product(range(1, 7), repeat=3):
            for wild in (False, True):
                for face in range(1, 7):
                    expected = dice.count(face)
                    if wild and face != 1: expected += dice.count(1)
                    self.assertEqual(matching_count((dice,), Bid(1, face), wild), expected)

    def test_eliminated_players_skipped_and_uninvolved_reward(self):
        r = Rules(4, 2, False)
        s = State(((2,), (), (3,), (4,)), 3, Bid(2, 3), 2)
        t = step(r, s, CALL, random.Random(2))
        self.assertEqual(t.rewards, (0, 0, -1, 1))
        self.assertEqual(t.state.counts, (1, 0, 0, 1))
        self.assertEqual(t.state.turn, 3)
        s2 = step(r, t.state, Bid(1, 2), random.Random(2)).state
        self.assertEqual(s2.turn, 0)

    def test_truthful_bid_loses_caller(self):
        r = Rules(3, 2)
        s = State(((1,), (2,), (3,)), 1, Bid(2, 2), 0)
        t = step(r, s, CALL, random.Random(3))
        self.assertEqual(t.rewards, (1, -1, 0))
        self.assertEqual(t.challenge.actual_count, 2)
        self.assertEqual(t.state.turn, 2)

    def test_terminal_consistency(self):
        r = Rules(2, 1, False)
        s = State(((2,), (3,)), 1, Bid(2, 6), 0)
        t = step(r, s, CALL, random.Random(1))
        self.assertTrue(t.state.terminal)
        self.assertEqual(t.state.winner, 1)
        self.assertEqual(t.state.counts, (0, 1))
        self.assertEqual(legal_actions(r, t.state), ())
        with self.assertRaises(ValueError): step(r, t.state, Bid(1, 1), random.Random(1))

    def test_observation_has_no_opponent_hand(self):
        r = Rules(3, 2)
        a = State(((1, 2), (3, 4), (5, 6)), 0)
        b = State(((1, 2), (6, 6), (1, 1)), 0)
        self.assertEqual(observe(r, a), observe(r, b))
        self.assertNotIn('hands', [f.name for f in fields(Observation)])
        with self.assertRaises(Exception): observe(r, a).counts[0] = 0

    def test_no_state_aliasing(self):
        r = Rules()
        s = reset(r, random.Random(4))
        before = repr(s)
        step(r, s, Bid(1, 1), random.Random(4))
        self.assertEqual(repr(s), before)

    def test_rng_replay(self):
        a = run_game(Rules(3, 2), ('probability', 'random', 'random'), 44, trace=True)
        random.seed(999); [random.random() for _ in range(10)]
        b = run_game(Rules(3, 2), ('probability', 'random', 'random'), 44, trace=True)
        self.assertEqual(a, b)

    def test_finite_games_and_conservation(self):
        for n in (2, 3, 5):
            for wild in (False, True):
                r = Rules(n, 2, wild)
                for seed in range(25):
                    rng = random.Random(seed); s = reset(r, rng)
                    while not s.terminal:
                        old = sum(s.counts)
                        action = rng.choice(legal_actions(r, s))
                        t = step(r, s, action, rng)
                        self.assertEqual(sum(t.rewards), 0)
                        self.assertEqual(sum(t.state.counts), old - (t.challenge is not None))
                        self.assertGreater(t.state.counts[t.state.turn], 0)
                        s = t.state
                    self.assertEqual(sum(n > 0 for n in s.counts), 1)

    def test_probability_exhaustive_against_dice_oracle(self):
        for wild in (False, True):
            r = Rules(2, 2, wild)
            s = State(((1, 3), (2, 2)), 0)
            obs = observe(r, s)
            for face in range(1, 7):
                for q in range(1, 5):
                    successes = 0
                    for hidden in itertools.product(range(1, 7), repeat=2):
                        vals = (1, 3) + hidden
                        count = sum(v == face for v in vals)
                        if wild and face != 1: count += vals.count(1)
                        successes += count >= q
                    self.assertAlmostEqual(bid_probability(obs, Bid(q, face)), successes/36)

    def test_reject_invalid_external_states(self):
        r = Rules(2, 1)
        cases = [State([[2], [3]], 0), State(((2,), (3,)), True),
                 State(((2,), (3,)), 0.5), State(((2,), (3,)), 0, Bid(1, 2), 0),
                 State(((2,), (3,)), 0, history=[]),
                 State(((2,), (3,)), 0, round_number=0)]
        for state in cases:
            with self.assertRaises(ValueError): observe(r, state)

    def test_probability_forced_call(self):
        r = Rules(2, 1)
        s = State(((2,), (4,)), 0, Bid(2, 6), 1)
        self.assertEqual(probability_policy(observe(r, s), random.Random(0)), CALL)

    def test_benchmark_balance_and_replay(self):
        a = benchmark(8, 100)
        self.assertEqual(a, benchmark(8, 100))
        for m in a['matchups']:
            self.assertEqual(len(m['records']), 8)
            for seat, start in itertools.product((0, 1), repeat=2):
                self.assertEqual(sum(r['focal_seat'] == seat and r['starter'] == start for r in m['records']), 2)
        with self.assertRaises(ValueError): benchmark(5)


if __name__ == '__main__': unittest.main()
