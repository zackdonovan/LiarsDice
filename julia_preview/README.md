# Static-only Julia repair preview

`main.jl` is a narrowly repaired copy of the source simulator. It threads RNG state, skips eliminated players, validates actions, fixes uninvolved-player rewards, represents terminal states consistently and guards direct execution.

**Not executed here:** Julia is not installed in the execution environment. This preview is not the tested release and should not be submitted as an independently verified Julia implementation. Use the Python entry point in the root README. The original POMDP adapter has not been carried into this directory because a proper fixed-ego stochastic opponent interface requires separate implementation and tests.

The exact proposed core diff is in `../docs/julia-core-preview.patch`; original source is in `../original/`.
