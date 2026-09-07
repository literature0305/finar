# Appendix: Regularisation vs Dependency Modelling (FiNAR Experiment 3)

## A.1 The question

Iterative refinement improves forecasts. This experiment asks **why**: is the
second pass modelling dependency inside the forecast target, or is it acting as
a regulariser?

The two explanations make different predictions. A regulariser reduces variance
without adding capacity to fit the training distribution, so its benefit shows
up as a **narrower gap between seen and unseen data**. Dependency modelling adds
capacity that helps wherever the dependency exists, on training and held-out
data alike, and should help *more* as the horizon lengthens because a single
pass has more joint structure to get wrong at once.

## A.2 What was measured

The generalization gap was measured with **GIFT-Eval's training split**
(`/group-volume/ts-dataset/GiftEvalTrainSplit*`): one checkpoint is scored on
data drawn from the portion of GIFT-Eval it was trained on, and on the
held-out GIFT-Eval benchmark, at iteration 1 and iteration 2. The quantity of
interest is not either score but their difference,

$$\text{gap} = \text{metric}_{\text{held-out}} - \text{metric}_{\text{seen}},$$

and how that difference moves from iteration 1 to iteration 2. A second pass
that acts as a regulariser narrows the gap; one that adds modelling capacity
moves both terms together and leaves the gap roughly where it was.

Using the GIFT-Eval training split rather than a freshly drawn training subset
means the "seen" side is the corpus the model was actually trained on, listed
in the training config, rather than a reconstruction of it.

**Two variants exist and they are not interchangeable.** The tsm-trainer v4
training configs list `GiftEvalTrainSplitOverlap/` (34 of 55 configs, at
`custom_weight` 5.0); the `GiftEvalTrainSplit/` no-overlap variant is not listed
in them, because its cut is shifted back by the context length at the base's
coarsest frequency and 54 of its 55 configs end up entirely below `min_past`.
Record which variant a reported gap was measured on — the "seen" side means
something different for each.

## A.3 The code in this directory was not used

`build_subset.py`, `run_eval.py`, `run_exp003.sh` and `configs/` implement a
**different, unused** approach to the same question: draw a 5,000-series subset
of a training mixture using tsm-trainer's own sampling rule (a dataset chosen in
proportion to `data_points × custom_weight`, then a row uniformly inside it),
freeze it as a corpus, and score it at three horizons to see whether the
iteration-1 → iteration-2 gain grows with horizon length.

That path was abandoned in favour of the GIFT-Eval training split described
above. The files are kept because the sampling reproduction is the non-obvious
part and may be wanted again, not because they produced any reported number.
**Nothing in this directory was run to produce the results in the paper.**

| File | Status |
|---|---|
| `build_subset.py` | unused — draws a 5k subset by the trainer's own sampling rule |
| `run_eval.py` | unused — scores that subset at every recursion depth |
| `run_exp003.sh` | unused — end-to-end driver for the two above |
| `configs/train_subset_5k.yaml` | unused — benchmark entries for the three horizons |
