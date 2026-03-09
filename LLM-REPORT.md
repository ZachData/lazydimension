# LLM Report

**Model:** openai/gpt-5.2

---

Credence: 67%

Explanation:
The core training/evaluation pipeline in `run_experiments.py` → `main.py` largely matches the trusted Experimental Setup and Appendix B (Fashion-MNIST, 10k train / 50k test, soft-hinge via softplus with beta=20, stop at margin 1, 600s wall timeout).
The strongest red flag is a parameter/plotting convention that makes the headline scaling result hard to trust as an honest empirical finding. The saved results use a single `alpha` value per run (the value passed to training), but `plot.py` and PAPER Table 1 treat “α” as `alpha_code / sqrt(h)` and “α√h” as `alpha_code`. I verified from `runs/*.json` that Table 1’s α column is exactly computed as (alpha_passed_to_code / sqrt(h)) while the plotted “α√h” axis is just the raw alpha passed to training, not α·sqrt(h) for a fixed α. This makes the claimed collapse/boundary scaling potentially an artifact of relabeling rather than something the training dynamics produced.
Separately, there is a clear, high-impact bug in the kernel-gradient-flow path: the call to `gradientflow_kernel` in `main.py` is positional-argument scrambled and would fail if executed, which could invalidate any kernel-based “lazy regime” comparisons if those were intended.

Proposed fixes:

Fix 1 (most likely):
- Location: `main.py:65`
- Change: Fix the argument order when calling `gradientflow_kernel` inside `run_kernel`, using keywords to match `gradientflow/_kernel.py`’s signature. For example, pass `loss_prim=partial(loss_func_prime, args)`, `tau=tau`, `max_dgrad=args['max_dgrad']`, `max_dout=args['max_dout'] / args['alpha']`.
- Mechanism: The current call passes `tau` as `loss_prim` and shifts every subsequent argument, so kernel training either crashes or silently computes nonsense if Python accepts it. If any experiments/figures rely on kernel dynamics to characterize the NTK/lazy regime, those results are wrong or missing.
- Expected effect: Kernel-based baselines/diagnostics would start working; any comparisons between “network training” and “kernel/lazy limit” could change qualitatively.

Fix 2:
- Location: `plot.py:5` and `plot.py:60` (and the x-axis mapping at `plot.py:61-83`)
- Change: Plot according to the trusted paper description: left panel x = raw α used in training (the `alpha` stored in `runs/*.json`), right panel x = α * sqrt(h). Concretely, remove `alpha_paper = alpha_code / np.sqrt(h)` and instead set `alpha_raw = result['alpha']`, with `alpha_scaled = alpha_raw * np.sqrt(h)` for the right panel.
- Mechanism: Current plotting defines α := alpha_code / sqrt(h) and α√h := alpha_code, so the “α√h” axis is just the training hyperparameter, while the “α” axis is a width-normalized relabeling. This can manufacture an apparent α* ~ O(h^-1/2) boundary by construction, rather than demonstrating it.
- Expected effect: The reported “collapse” / boundary scaling may weaken, disappear, or shift; key finding could change if the honest scaling variable is not the one currently plotted.

Fix 3:
- Location: `PAPER.md:141` (Appendix A.1) and `plot.py:5`
- Change: Make α conventions consistent with the trusted methodology in Section 3: either (a) treat `--alpha` as the α in F(w,x)=α[f(w,x)-f(w0,x)] everywhere (paper/table/plots), or (b) if you truly want α_paper := alpha_code/sqrt(h) to be the “real” α, then modify the training code to use that α (e.g., set `args['alpha'] = args['alpha'] / sqrt(h)` before computing losses/stopping), and regenerate runs/plots accordingly.
- Mechanism: Right now the code trains with `args['alpha']` but the paper’s Table 1 and plotting treat α as `args['alpha']/sqrt(h)`. This paper–code mismatch can flip the interpretation of the scaling law and the boundary location.
- Expected effect: Either the key finding becomes a genuine statement about the trained α, or the trained α changes and the curves/key finding likely change.

Fix 4:
- Location: `run_experiments.py:77` and `main.py:173-176` (where `tau` is used)
- Change: Set `tau_over_h` to `0.0` in `EXPERIMENT_ARGS_TEMPLATE` (and/or remove the momentum-like dynamics) to match “continuous-time gradient descent (gradient flow)” as described. Optionally document if momentum is intended.
- Mechanism: Nonzero `tau` introduces continuous momentum (second-order dynamics) via `gradientflow/_backprop.py`, which can change convergence behavior and the regime transition with α and h. That can materially change the curves that underpin the paper’s key finding.
- Expected effect: Different test-error curves and potentially a different inferred regime boundary; could change whether “lazy vs feature” transition aligns with α* ~ h^-1/2.

Fix 5 (least likely / more speculative):
- Location: `dataset/__init__.py:281`
- Change: Avoid normalization leakage by computing centering/scaling using only the training split, then applying the same transform to test/kernel sets, instead of calling `center_normalize` on the concatenated train+test pool before splitting.
- Mechanism: Current `center_normalize` is applied before `intertwine_split`, so test inputs influence the normalization applied to train and vice versa. Even unsupervised leakage can change margins/generalization and could bias comparisons across α/h.
- Expected effect: Slightly higher and/or differently-shaped test-error curves; could shift the perceived boundary and reduce over-optimistic generalization.

Experiments run:
python -c style analyses on saved results only (no training):
- Loaded `runs/*.json` and computed per-(h, alpha) mean `final_test_err`, confirming Table 1 values are produced by using alpha_paper = alpha_code / sqrt(h) while alpha_code is the stored `alpha` in the JSON.
- Render check: viewed `results/plot.png` and confirmed the two panels correspond to the plotting convention implemented in `plot.py` rather than a direct “α vs α√h” based on the raw training α.