# Baseline Gate

## Goal

The baseline gate decides whether the project is ready to move from bring-up into the actual scheduler comparison.

For the 24h version, the gate required:

- `Balanced MoE` `SMOKE`
- `Hot-Expert` `PILOT`
- `Hot-Rank` `PILOT`
- one winning hot workload selected for `FULL`

## Workloads exercised

- `Balanced MoE`
- `Hot-Expert`
- `Hot-Rank`

Baseline artifacts:

- [results/baseline/balanced_full.json](C:/26spring/nv项目/trtllm-moe-runtime-exp/results/baseline/balanced_full.json)
- [results/baseline/hot_expert_full.json](C:/26spring/nv项目/trtllm-moe-runtime-exp/results/baseline/hot_expert_full.json)
- [results/baseline/hot_rank_full.json](C:/26spring/nv项目/trtllm-moe-runtime-exp/results/baseline/hot_rank_full.json)

## Gate result

### Balanced MoE

- baseline is stable at 3 batches of 4 requests
- used as the non-regression control

### Hot-Expert

- clearly produces heavier tail behavior under the fixed batch-of-4 baseline
- patched path showed the cleanest improvement in latency/tail metrics

### Hot-Rank

- also showed improvement under the patched path
- but the `Hot-Expert` signal was slightly cleaner and was chosen as the final `FULL` extension

## Winning workload

`Hot-Expert`

Reason:

- more direct alignment with the project story of MoE pressure and batch stragglers
- clear p90 E2E improvement in the pilot compare
- easy to explain in an interview
