# Study v6 kit

Files (copy into `sequential_game/`, overwrite the three existing ones):
- `game3.py` (changed: margins, rationing rules cea/cel, grouped best response; defaults reproduce v5 exactly)
- `eps_check.py` (changed: stronger epsilon search with pair moves)
- `run_study4.py` (changed: new job kind `eps_strong`)
- `run_study6.py`, `analyze6.py`, `test_v6.py` (new)

Run (Kaggle, CPU, from `/kaggle/working/repo/sequential_game`):
    python test_v6.py                      # ~3 min, must end ALL OK
    python run_study6.py --count_only
    python run_study6.py --out seq3_results --resume_from <old results dirs> --tiers A
    python run_study6.py --out seq3_results --tiers B
    python run_study6.py --out seq3_results --tiers C     # slow (~10 min per job)
    python run_study6.py --out seq3_results --tiers D     # optional: doubles tier A sample
    python analyze6.py --dir seq3_results                 # writes analysis6.md

Experiments: 12 scarcity curve; 13 rationing rules (claims problem); 14 payoff-scale sensitivity;
15 heterogeneous margins, price vs quantity; 16 three operators, Shapley value and core; 17 stronger epsilon.
Tier A needs the study v4 e6 H=3 arms only for one comparison row; the table skips it if absent.
