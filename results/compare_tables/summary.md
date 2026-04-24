# Result Tables

## Experiment Summaries

| Label | Requests | Batches | Avg Req/Batch | Avg Batch ms | Step Std ms | TTFT p90 s | E2E p90 s | TPOT p90 s | Throughput tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| balanced_baseline | 12 | 3 | 4.00 | 1608.35 | 230.68 | 0.0816 | 1.4878 | 0.0115 | 278.55 |
| balanced_patched | 12 | 3 | 4.00 | 1591.66 | 183.85 | 0.0739 | 1.4748 | 0.0115 | 281.47 |
| hot_expert_baseline | 12 | 3 | 4.00 | 1939.20 | 212.77 | 0.0748 | 1.8668 | 0.0116 | 297.03 |
| hot_expert_patched | 12 | 12 | 1.00 | 1446.18 | 166.68 | 0.0112 | 1.5741 | 0.0098 | 99.57 |
| hot_rank_baseline | 12 | 3 | 4.00 | 2192.92 | 298.73 | 0.0737 | 1.9942 | 0.0115 | 277.26 |
| hot_rank_patched | 12 | 12 | 1.00 | 1529.56 | 170.76 | 0.0131 | 1.7115 | 0.0100 | 99.37 |
| hot_expert_full24_baseline | 24 | 6 | 4.00 | 1855.43 | 158.69 | 0.0733 | 1.8348 | 0.0114 | 310.44 |
| hot_expert_full24_patched | 24 | 24 | 1.00 | 1442.68 | 157.35 | 0.0111 | 1.5983 | 0.0101 | 99.81 |

## Baseline vs Patched

| Workload | TTFT p90 delta s | E2E p90 delta s | TPOT p90 delta s | Step Std delta ms | Throughput delta tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| balanced_full | -0.0077 | -0.0129 | -0.0001 | -46.83 | 2.92 |
| hot_expert_full | -0.0635 | -0.2927 | -0.0017 | -46.10 | -197.46 |
| hot_rank_full | -0.0606 | -0.2827 | -0.0015 | -127.98 | -177.88 |
| hot_expert_full24 | -0.0622 | -0.2365 | -0.0013 | -1.34 | -210.63 |
