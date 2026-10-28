Placement policies on candidate set SIMULATED-qwen7-pressure-v2 (sha256 8fc52ce2814729d38436860c1b3e010c705cccdd5280b83b9bc441745ef866cb).

Simulated: each plan passes or fails a stated memory estimate; none is executable or runtime-verified.

| policy | outcome | plan_id | logical split | device order | headroom bytes per stage |
| --- | --- | --- | --- | --- | --- |
| best_single_device | FEASIBLE | candidate-7cb504ce3de7854e | [28] | SIMULATED-second-mac | 19798163456 |
| equal_split | INFEASIBLE | candidate-f52712c34dc8ad3a | [14, 14] | SIMULATED-m4-pressure, SIMULATED-second-mac | -838336512, 21710241792 |
| memory_proportional | FEASIBLE | candidate-bfa599b48584765b | [1, 27] | SIMULATED-m4-pressure, SIMULATED-second-mac | 937164800, 19934740480 |
