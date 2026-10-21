# XGoBoost

Gradient-boosted decision trees with **monotone constraints**, in pure Go.

- Second-order boosting: Newton leaf values `-G/(H+λ)` and the usual
  second-order split gain, under a pluggable `Loss`.
- Built-in losses: squared error, Huber, logistic (binary), quantile.
- Per-feature monotone constraints (`+1`, `-1`, `0`), guaranteed across
  the whole ensemble, not just per split.
- Histogram split finding (quantile binning, per-node gradient/hessian
  histograms, histogram subtraction) plus an exact greedy mode for small
  data.
- Missing values (`NaN`) with a learned default direction at every split.
- L2 λ, min child weight, min split gain, depth, learning rate, row and
  column subsampling, early stopping on a validation set.
- Deterministic for a given seed, whatever the number of workers.
- JSON model serialisation that round-trips bit-for-bit.
- Standard library only. Go 1.22+.

```sh
go get github.com/natemsz/xgoboost
```

## Usage

```go
package main

import (
	"fmt"
	"log"
	"math"
	"os"

	xgoboost "github.com/natemsz/xgoboost"
)

func main() {
	// X: one row per sample, NaN where a value is unknown.
	X := [][]float64{
		{0.9, 0.1, 0.3},
		{0.2, math.NaN(), 0.8},
		// ...
	}
	y := []float64{9.1, 1.7 /* ... */}
	validX, validY := X, y // use genuinely held-out rows

	p := xgoboost.DefaultParams()
	p.Monotone = []int{+1, -1, 0} // up in feature 0, down in 1, free in 2
	p.Rounds = 500
	p.Subsample, p.ColSample, p.Seed = 0.8, 0.8, 42

	// Optional: early stopping on held-out data.
	p.Validation = &xgoboost.ValidationSet{X: validX, Y: validY}
	p.EarlyStoppingRounds = 20

	m, err := xgoboost.Train(X, y, nil /* unit weights */, p)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(m.Predict([]float64{0.5, 0.2, 0.4}))
	fmt.Println(m.PredictBatch(X))
	fmt.Println(m.BestIteration, m.FeatureImportance(xgoboost.TotalGain))

	// Save and load (error handling elided).
	f, _ := os.Create("model.json")
	_ = m.Save(f)
	f.Close()
	f, _ = os.Open("model.json")
	defer f.Close()
	m2, _ := xgoboost.Load(f)
	fmt.Println(m2.Predict([]float64{0.5, 0.2, 0.4})) // identical
}
```

Classification uses `p.Loss = xgoboost.Logistic{}` with labels in
`[0, 1]`; `Predict` then returns a probability and `PredictRaw` the
log-odds. Quantile regression uses `xgoboost.Quantile{Alpha: 0.9}`, and
robust regression uses `xgoboost.Huber{Delta: 1}`.

### Parameters

| Field | Default | Meaning |
|---|---|---|
| `Loss` | `SquaredError{}` | training objective |
| `Rounds` | 100 | number of trees (upper bound with early stopping) |
| `LearningRate` | 0.1 | shrinkage applied to every tree |
| `MaxDepth` | 6 | maximum tree depth |
| `Lambda` | 1 * | L2 penalty on leaf values |
| `MinChildWeight` | 1 * | least hessian sum per child |
| `MinSplitGain` | 0 | least loss reduction per split (XGBoost's γ) |
| `Monotone` | nil | per-feature `+1` / `-1` / `0` |
| `Subsample` | 1 | share of rows per tree, without replacement |
| `ColSample` | 1 | share of features per tree |
| `Seed` | 0 | sampling seed |
| `Method` | `Auto` | `Exact` up to 2000 rows, else `Hist` |
| `MaxBins` | 256 | histogram bins per feature (2–65535) |
| `Validation`, `EarlyStoppingRounds` | none | early stopping |
| `Workers` | GOMAXPROCS | goroutines for histogram building and split search |

\* In `DefaultParams()`. In a zero `Params`, the fields marked
"0 means default" in the godoc take the defaults above. `Lambda`,
`MinChildWeight` and `MinSplitGain` are used as given, so a zero
`Params` is valid but unregularised.

### Custom losses

Implement `Loss` (`Name`, `InitScore`, `GradHess`, `Transform`, `Metric`).
Optionally also implement:

- `LossValidator` to check labels and parameters before training;
- `ParamLoss` to persist parameters;
- `LeafRenewer` to replace Newton leaf values after a tree is grown
  (the quantile loss does this).

Call `RegisterLoss(name, decoder)` before loading a saved model that
uses the loss. `Transform` must be non-decreasing so that constraints
carry over from raw scores to predictions.

## How the monotone constraint is enforced

The construction is the one XGBoost uses. Each node carries an interval
`[lo, hi]`, and every leaf value in its subtree is clamped into it. The
root's interval is `(-∞, +∞)`.

When the split finder considers a split on feature `j` with constraint
`c[j]`, it computes the children's Newton values `vL`, `vR` and clamps
them to the node's interval. Then:

1. **Reject** the split if it runs the wrong way: `vL > vR` when
   `c[j] = +1`, or `vL < vR` when `c[j] = -1`. Gain is computed with the
   clamped values, `½(score(GL,HL,vL) + score(GR,HR,vR) − score(G,H,v))`,
   where `score(G,H,w) = −(2Gw + (H+λ)w²)`. This is the loss reduction
   the split actually achieves, not the one it would achieve unclamped.
2. If the split is kept, **bound** the children at the midpoint
   `m = (vL + vR)/2`. For `+1`, the left subtree gets `[lo, m]` and the
   right subtree gets `[m, hi]`. For `−1`, the bounds are mirrored. For
   an unconstrained feature both children inherit `[lo, hi]`.

**Why one tree is monotone.** Take two inputs `x ≤ x'` that differ only
in a `+1` feature `j`, both non-missing. Walk them down the tree
together:
- At a split on any other feature, they go the same way.
- At a split on `j`, either they go the same way, or `x` goes left and
  `x'` goes right.

The first time they part, `x` lands in the left subtree, whose leaves
all lie in `[lo, m]`. `x'` lands in the right subtree, whose leaves all
lie in `[m, hi]`. So `f(x) ≤ m ≤ f(x')`. If they never part, they reach
the same leaf. Either way `f(x) ≤ f(x')`.

The bounds are inherited and only ever tighten. A later split on some
other feature, deep inside the left subtree, therefore cannot produce a
leaf above `m`. This is the case a split-local check alone gets wrong
(see `TestMonotoneInteraction`).

**Why the ensemble is monotone.** Each tree is monotone, and so is a
sum of monotone functions. Scaling by a positive learning rate preserves
order. Floating-point addition is monotone too: `a ≤ b` implies
`fl(a + c) ≤ fl(b + c)`. The tests therefore check the property exactly,
with zero tolerance: they sweep each constrained feature from `−∞`
to `+∞` while holding the other features fixed (some of them `NaN`).
`Loss.Transform` is non-decreasing, so `Predict` inherits the guarantee
from `PredictRaw`.

**Leaf renewal.** Losses such as quantile replace a leaf's Newton value
after the tree is grown. The new value is clamped to that leaf's
envelope, so the argument above still holds.

**Missing values.** `NaN` is not ordered, so it has no place on the
constrained axis. A row whose constrained feature is missing follows the
split's learned default direction. The guarantee covers every pair of
non-missing values.

## Missing values

When it evaluates a split, the finder tries sending the node's `NaN`
rows left and then right, and keeps whichever gives the larger gain.
That choice is stored as the node's `DefaultLeft`. If a feature had no
missing values at a node during training, `NaN` at prediction time
follows the child with more hessian cover. A row shorter than
`NumFeatures` is read as `NaN` for the missing positions.

## Histogram vs exact

`Hist` bins each feature once into at most `MaxBins` weighted-quantile
bins. A feature with at most `MaxBins` distinct values gets one bin per
value, with cuts at the midpoints between them. Each node accumulates
per-bin gradient and hessian sums. Only the smaller child's histogram is
built from its rows; the larger child's is the parent's minus the
smaller one.

`Exact` sorts each feature once up front. Every node keeps its rows in
each feature's sorted order by stable partitioning, so no node sorts
again.

When every feature has at most `MaxBins` distinct values, both methods
evaluate the same candidate splits and grow the same trees
(`TestHistMatchesExact`).

## Determinism

Row and column sampling use a PCG generator seeded from `Params.Seed`.
Parallel work writes results by feature index and reduces them in a
fixed order, with ties going to the lower feature index. The same data,
parameters and seed therefore produce a byte-identical model for any
`Workers` value (`TestDeterministic`).

Results can differ across CPU architectures where the compiler fuses
multiply-adds.

## Benchmarks

Measured with `go test -bench . -benchtime 3x` on a 12-thread Windows
desktop with Go 1.27, 50 rounds at depth 6:

| Benchmark | Rows × features | Time per fit |
|---|---|---|
| `TrainHist100k` | 100 000 × 20 | ~1.0 s |
| `TrainHist20k` | 20 000 × 20 | ~0.36 s |
| `TrainHist20kMono` (2 constrained) | 20 000 × 20 | ~0.36 s |
| `TrainExact2k` | 2 000 × 10 | ~0.13 s |
| `PredictBatch` (100 trees) | 20 000 × 20 | ~5 µs / row |

The library is not yet tuned to the level of XGBoost or LightGBM. Both
are far faster on large data.

## Not (yet) supported

- Multiclass, ranking and Poisson/Tweedie losses (all possible through
  the `Loss` interface plus a wrapper, but none is built in).
- Categorical splits, interaction constraints, `max_delta_step`, and
  per-level or per-node column sampling.
- Sparse input, external memory, and GPU.

## License

Apache-2.0. See [LICENSE](LICENSE).
