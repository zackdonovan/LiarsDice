# Verification receipt — November 1, 2026

Executed on Linux x86-64, Python 3.12.14; no external runtime packages. No network or paid service is used by the game.

- 17 standard-library unittest methods passed, including 150 randomized multiplayer games, exhaustive three-die matching counts, and binomial probability comparisons against all 36 possible hidden two-die rolls.
- The final 17-method suite and both recorded 2,000-game benchmark matchups were independently rerun by a separate reviewer; records matched exactly.
- Reviewer identified invalid externally constructed state bugs (self-challenge, mutable hands, invalid index types). These were corrected; the 17th regression method covers them.
- Original seed-42 demo and replayed benchmark records match the checked-in JSON.
- Additional independent stress tests: ten players × ten dice, wild ones on/off, terminate in 99 challenges and conserve challenge reward.
- Invalid CLI configuration returns status 2. Existing output is refused; exclusive file creation protects against a later race.
- No Julia interpreter available: `julia_preview/` is static-only, and the original POMDP adapter is not supported.
- Python CLI is verified on Linux only; no interactive browser, desktop or cross-platform runtime success is claimed.

Recheck after edits with `python3 -m unittest discover -s tests -v`. Reproduce JSON using the README command with a fresh output filename. The figure is a visualization of the saved results, not an independent experiment.
