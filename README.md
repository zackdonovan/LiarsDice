# Liar's Dice: hidden-information simulation

A small, complete game engine with reproducible policy comparisons. The interesting engineering problem is not rolling dice: it is keeping private information private, defining legal transitions precisely, and measuring strategies without confusing a weak baseline with optimal play.

**Runnable implementation:** dependency-free Python 3.10+. The original Julia simulator/POMDP sketch is preserved in `original/`; this November 2026 revision adds a Python port, tests and documentation. See [provenance](docs/PROVENANCE.md) for what changed. No learned policy or equilibrium solver is claimed.

![Fixed baseline comparison](results/benchmark.svg)

## Run in one minute

From this folder, with Python 3.10 or newer:

```sh
python3 -m unittest discover -s tests -v
python3 -m liars_dice demo --seed 42
python3 -m liars_dice benchmark --games 2000 --seed 20261101 --output /tmp/dice-benchmark.json
```

No package installation, credentials, network or external data are needed. On Windows, use `py` in place of `python3` and choose a Windows output path. The source is portable, but the executed acceptance checks were on Linux/Python 3.12, not Windows/macOS. An existing output file is never overwritten.

Optional installation: `python3 -m pip install .` provides the `liars-dice` command. Installation needs setuptools in the build environment; the module commands above do not.

## What is implemented

- Immutable full game state, separate observation type, and deterministic replay from explicit RNG seeds.
- Multiplayer elimination and a finite, monotone bidding ladder.
- Challenge resolution, round rewards and complete game win detection.
- Observation-only random and probability-based reference policies.
- A balanced-seat/starter benchmark with per-game JSON records and uncertainty intervals.
- Exhaustive small-dice probability checks, transition tests and multiplayer invariants.

## Exact rules (a deliberate variant)

Players are numbered from **0** in Python. Each starts with the same number of fair six-sided dice; each round begins with fresh independent rolls.

1. A bid `(quantity, face)` means **at least** that many matching dice across all live players.
2. Bids increase lexicographically: a higher quantity, or the same quantity with a higher face. A bid may not exceed the total number of remaining dice. This includes bids on ones; there is **no special ones conversion rule**.
3. Ones are wild for bids on 2–6 by default. For a bid on ones, only ones count. `Rules(..., ones_wild=False)` disables wild ones.
4. Calling is legal only after a bid. If the bid is true, the caller loses one die; otherwise the bidder loses one. Exact equality is a true bid.
5. Every player rerolls. The next live player strictly after the loser starts the next round. Players at zero dice are skipped. This starter rule follows the intent of the original prototype; it differs from common variants where the loser starts.
6. The last player with dice wins the **game**. A challenge winner need not be the final game winner.
7. A challenge produces round rewards `+1` to its winner, `−1` to its loser, and `0` to uninvolved players. Bids have zero reward. Benchmark scores use final wins, not summed round rewards.

The engine supports 2–10 players and 1–10 initial dice each. The included benchmark is specifically **two players, two dice, wild ones**. There is no betting, money, online play, or connection to a gambling service.

## Hidden information and the probability baseline

A policy receives only its own dice, public dice counts, the current bid/bidder, current-round bid history and legal actions. The simulator's full state is never passed to it. Terminal observations have no legal actions. Trace output intentionally reveals dice **after** a challenge for auditing; it is not policy input.

For a candidate bid, the probability baseline counts its own matching dice and computes a binomial tail for the unseen dice: success probability `1/6`, or `2/6` for a non-one face with wild ones. It calls if the current bid has less than 50% probability; otherwise it makes the most probable legal bid, breaking ties toward the smallest bid.

**Crucial limitation:** this calculation ignores the information in opponents' strategic bids. It is an independent-fair-dice prior conditioned on one's own hand, not a correct posterior after informative bidding. The policy is a fixed heuristic, not a POMDP solution, optimal strategy or model of human bluffing.

## Results you can reproduce

The checked-in run uses 2,000 independent game seeds per matchup, beginning at 20261101. Focal seat and starting seat cover all four combinations equally. Dice and policy randomness use separate explicit RNG streams. The 0.5 threshold was fixed before the recorded benchmark, not selected for best results.

| Focal policy vs opponent | Wins | Win rate | 95% Wilson interval |
|---|---:|---:|---:|
| Random vs random | 992 / 2000 | 49.60% | 47.41–51.79% |
| Probability vs random | 1917 / 2000 | 95.85% | 94.89–96.64% |

These are synthetic outcomes against an intentionally weak, uniformly random legal-action policy. They do **not** demonstrate skill against humans, learned agents or equilibrium opponents. Wilson intervals describe Monte Carlo uncertainty for this fixed design; they do not account for model/rule mismatch. Both matchups reuse the same seed range; they are not independent of each other. Different Python versions may change RNG implementation details: reproduce the exact recorded records with Python 3.12.

Full records: [`results/benchmark.json`](results/benchmark.json). A single auditable game: [`results/demo.json`](results/demo.json). [Verification receipt](docs/VERIFICATION.md).

## Project map

- `liars_dice/engine.py`: rules, state, observations, transitions.
- `liars_dice/policies.py`: random and binomial-prior baseline.
- `liars_dice/simulation.py`: game loop and balanced evaluation.
- `liars_dice/__main__.py`: CLI with argument validation/no-clobber output.
- `tests/`: standard-library tests; no testing dependency required.
- `original/`: unchanged upstream text snapshot (not a release entry point).
- `julia_preview/`: statically repaired Julia core, **not executed here**.

## Walkthrough for a technical conversation

1. Run the seeded demo and identify what the acting player can observe.
2. Show the regression test where an eliminated player is skipped and an uninvolved player receives zero reward.
3. Explain why the exact binomial calculation is still not a strategic posterior.
4. Reproduce the benchmark and explain why beating random play is a sanity check, not a strong game-AI result.
5. Extend one rule or policy and add a test before changing the result claim.

The provenance notes distinguish the original Julia implementation from the November 2026 Python port, repairs, tests and documentation.
