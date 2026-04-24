# Interview Talking Points

## 60-second version

I built a MoE-first TensorRT-LLM runtime optimization project around `Qwen/Qwen1.5-MoE-A2.7B-Chat` on a single `4060 Ti 16GB`. Instead of trying to beat TRT-LLM kernels, I targeted the runtime layer. I added a minimal resource model with `request profile`, `pressure budget`, and `step plan`, then used that to drive a MoE-aware scheduler patch. I ran real TensorRT-LLM INT4 weight-only inference and designed `Balanced`, `Hot-Expert`, and `Hot-Rank` workloads. On hot MoE workloads, the patched planner substantially improved TTFT and tail latency, while the balanced control stayed stable. The tradeoff is that v1 is latency-first and sacrifices throughput, which I call out explicitly.

## 3-minute version

I wanted the project to look like TensorRT-LLM inference optimization work, not like a toy scheduler or a kernel side quest. So I fixed the model path to `Qwen/Qwen1.5-MoE-A2.7B-Chat + TensorRT-LLM INT4 weight-only`, got the full conversion/build path running on my own machine, and treated that as non-negotiable.

Then I focused on the MoE runtime problem: requests with different expert routing patterns create different pressure at decode time, but the scheduler does not explicitly treat that pressure as a first-class signal. I introduced a minimal runtime resource model with `RequestProfile`, `RuntimeBudget`, and `StepPlan`, plus three pressure classes: `balanced`, `hot_expert`, and `hot_rank`.

Architecturally, I patched the TRT-LLM 1.2.1 PyTorch backend seam around `BindMicroBatchScheduler` and `executor_request_to_llm_request`, so the project genuinely touches TRT-LLM internals. Quantitatively, because the PyTorch backend on my setup could not serve the quantized MoE path I needed, I ran the real engine benchmarks by externalizing the same planning logic into batch composition on the TensorRT backend. I keep that limitation explicit.

Experimentally, I used a balanced control and two hot workloads. The balanced case did not regress. On the hot workloads, the planner reduced TTFT and E2E tail latency a lot because it stopped stacking high-pressure requests into the same step. The downside is throughput: the v1 pressure budget is very conservative, so the hot batches often shrink to size 1. I treat that as the next optimization target, not as something to hand-wave away.

## Deep-dive prompts

### Why is this not just an external policy?

Because the project has a real TRT-LLM internal patch seam: `BindMicroBatchScheduler` plus the request conversion path. The benchmark path had to be externalized for quantized-model compatibility reasons, but the scheduler architecture work itself is internal.

### Why did you choose a resource model instead of a few heuristic rules?

I wanted the scheduler decision to flow through an explicit contract. `request profile -> runtime budget -> step plan` is much easier to reason about, test, and extend than scattering MoE conditions directly inside the schedule loop.

### What exactly improved?

On the winning `Hot-Expert` full compare with 24 requests:

- TTFT p90: `0.0733s -> 0.0111s`
- E2E p90: `1.8348s -> 1.5983s`
- TPOT p90: `0.0114s -> 0.0101s`

### What got worse?

Throughput. The same `Hot-Expert` full compare dropped from `310.44 tok/s` to `99.81 tok/s` because the current pressure budget isolates hot requests too aggressively.

### Why is the throughput drop still useful in an interview?

Because it proves I am not just presenting a cherry-picked win. The project shows that I can make a real runtime tradeoff visible, explain why it happens, and point to the next engineering step to improve it.

### What would you do next?

Adaptive pressure budgets or a two-tier admission policy, so hot requests can still be isolated without always collapsing to single-request batches.
