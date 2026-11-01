# Provenance and scope of this cleanup

Source repository: https://github.com/zackdonovan/LiarsDice

Snapshot: `48dfbb65a3468965219d478144f888becce21db8` (default branch inspected November 1, 2026). The commit metadata attributes this revision to GitHub account `zackdonovan123`.

Original text files are preserved byte-for-byte in `original/`:

| File | Git blob SHA-1 |
|---|---|
| main.jl | 2df1b253979f8f4e4fbe3facc62d0f2b5d5998cb |
| liars_dice_pomdp.jl | 3ac4a151614b7dcad9fffeeae38bbf6971841994 |
| .gitattributes | dfe0770424b2a19faf507a501ebfc23be8f54e7b |

The upstream `.DS_Store` is omitted; it is operating-system metadata, not source. No license file was present in this inspected source tree. This package does **not** invent an open-source license or broaden source rights. The owner should choose an appropriate license before public redistribution. No GitHub changes or submissions were made as part of preparing this local package.

## What was added

The November 2026 revision adds the Python engine, policies, benchmark, tests, CLI, figures and documentation. The original design supplied the turn-based dice simulator, lexicographic bids, optional wild ones and explicit observation/state separation. The original POMDP adapter was an interface sketch; it did not implement inference or a solved policy.

## Correctness changes versus the original

- Skip eliminated players on both bids and new rounds.
- End the complete game only when one player remains, rather than when a selected ego is eliminated. This is an explicitly changed episode convention.
- Give uninvolved players zero challenge reward instead of an erroneous loss.
- Reject out-of-range bids and illegal/terminal actions.
- Keep terminal hands consistent with remaining dice and give terminal states no legal moves.
- Thread explicit RNGs through all simulated randomness and isolate dice/policy streams.
- No import-time game execution.
- Immutable observation data, guarded external state construction and distinct bidder/caller validation.
- A heuristic probability baseline and reproducible evaluation are new additions, not recovered training results.

`julia_preview/main.jl` contains narrowly repaired original-core code; `julia-core-preview.patch` shows exact text changes. It has **not been executed or syntax-validated with Julia** in this environment. The runnable release entry point is Python. The Julia POMDP adapter remains only in `original/`: forwarding another player's observation and ignoring the supplied RNG are not a correct fixed-ego POMDP implementation. A properly validated single-agent adapter must fix an opponent policy, advance opponents to ego decision points and maintain private observations; no such solver integration is claimed here.

## Evaluation interpretation

The check-in of a 95.85% win rate is limited to a fixed heuristic versus uniformly random legal actions in a small synthetic game. It is not a learned strategy, game-theoretic optimum, exploitability measurement, or evidence of performance against people. Changing rules or opponents changes the task. Public history is exposed to policies, but the simple probability baseline deliberately does not infer from it.
