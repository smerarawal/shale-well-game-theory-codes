# Archived code

Everything below was removed from main but is preserved. Tag `pre-cleanup` is the full pre-cleanup tree.

| branch | what it holds |
|---|---|
| `archive/physics-fno-v1-v2` | Physics v1/v2 (single-phase diffusion), the first FNO surrogate, dataset generators for them, stage-1 greedy well placement, and the early visualisations. |
| `archive/coalition-games-blotto-shapley` | Cooperative-game work on the v1 FNO: Shapley, core/nucleolus, Blotto allocation, submodularity. Revive this when you build the 3-operator (core/Shapley) extension. |
| `archive/surrogate-scaffolds-v1-v3` | Per-architecture scaffolds and the old comparison drivers. Superseded by the single model zoo (fair_run/model_zoo_v3.py / v4.py), which holds all 16 architectures. |
| `archive/sequential-game-v1-v2` | Game engines v1 and v2 and their studies. Their headline results were retracted (collapsing belief, H=2 planner), so they are kept only for provenance. |

To use one: `git switch archive/<name>`.
