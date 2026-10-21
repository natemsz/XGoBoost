package xgoboost

import (
	"fmt"
	"math"
	"runtime"
)

// TreeMethod selects how split candidates are found.
type TreeMethod int

const (
	// Auto uses Exact for training sets of at most AutoExactMaxRows
	// rows and Hist above that.
	Auto TreeMethod = iota
	// Hist bins each feature into at most MaxBins quantile bins once,
	// then finds splits on per-node gradient/hessian histograms. Only
	// the smaller child's histogram is built; its sibling's is the
	// parent's minus it.
	Hist
	// Exact enumerates every threshold between distinct values present
	// at a node. It is O(n log n) per node and feature.
	Exact
)

// AutoExactMaxRows is the largest training set Auto fits with Exact.
const AutoExactMaxRows = 2000

// ValidationSet is held-out data scored after every round.
type ValidationSet struct {
	X [][]float64
	Y []float64
	W []float64 // nil means unit weights
}

// Params configures Train. Fields marked "0 means default" take the
// default when left zero; the others are used as given, so the zero
// Params is valid but unregularised. DefaultParams returns the
// recommended starting point.
type Params struct {
	// Loss is the objective. nil means SquaredError.
	Loss Loss
	// Rounds is the number of trees (at most, with early stopping).
	// 0 means 100.
	Rounds int
	// LearningRate scales every tree. 0 means 0.1.
	LearningRate float64
	// MaxDepth limits tree depth; a depth-d tree has at most 2^d
	// leaves. 0 means 6.
	MaxDepth int
	// Lambda is the L2 penalty on leaf values: leaves are -G/(H+λ).
	Lambda float64
	// MinChildWeight is the least hessian sum a child may hold. For
	// squared error with unit weights it is a row count.
	MinChildWeight float64
	// MinSplitGain is the least loss reduction a split must achieve
	// (XGBoost's gamma).
	MinSplitGain float64
	// Monotone holds one constraint per feature: +1 non-decreasing,
	// -1 non-increasing, 0 free. nil leaves every feature free.
	Monotone []int
	// Subsample is the share of rows drawn, without replacement, for
	// each tree. 0 means 1.
	Subsample float64
	// ColSample is the share of features drawn for each tree. 0 means 1.
	ColSample float64
	// Seed drives row and column sampling. Training is deterministic
	// for a given Seed, whatever Workers is.
	Seed uint64
	// Method picks the split finder. The zero value is Auto.
	Method TreeMethod
	// MaxBins bounds the histogram bins per feature, between 2 and
	// 65535. 0 means 256.
	MaxBins int
	// Validation, if set, is scored with Loss.Metric after each round;
	// the model records the best round.
	Validation *ValidationSet
	// EarlyStoppingRounds > 0 stops training once the validation
	// metric has not improved for that many rounds, and keeps only the
	// trees up to the best round. It requires Validation.
	EarlyStoppingRounds int
	// Workers bounds the goroutines used to build histograms and
	// search splits. 0 means GOMAXPROCS.
	Workers int
}

// DefaultParams returns sensible defaults: squared error, 100 rounds of
// depth-6 trees at learning rate 0.1, λ = 1, min child weight 1.
func DefaultParams() Params {
	return Params{
		Loss:           SquaredError{},
		Rounds:         100,
		LearningRate:   0.1,
		MaxDepth:       6,
		Lambda:         1,
		MinChildWeight: 1,
		Subsample:      1,
		ColSample:      1,
		MaxBins:        256,
	}
}

// resolve fills defaults and validates p for nf features and n rows.
func (p Params) resolve(n, nf int) (Params, error) {
	bad := func(name string, v any) (Params, error) {
		return p, fmt.Errorf("xgoboost: invalid %s: %v", name, v)
	}
	finite := func(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) }
	if p.Loss == nil {
		p.Loss = SquaredError{}
	}
	switch {
	case p.Rounds == 0:
		p.Rounds = 100
	case p.Rounds < 0:
		return bad("Rounds", p.Rounds)
	}
	switch {
	case p.LearningRate == 0:
		p.LearningRate = 0.1
	case !(p.LearningRate > 0) || !finite(p.LearningRate):
		return bad("LearningRate", p.LearningRate)
	}
	switch {
	case p.MaxDepth == 0:
		p.MaxDepth = 6
	case p.MaxDepth < 0:
		return bad("MaxDepth", p.MaxDepth)
	}
	if !(p.Lambda >= 0) || !finite(p.Lambda) {
		return bad("Lambda", p.Lambda)
	}
	if !(p.MinChildWeight >= 0) || !finite(p.MinChildWeight) {
		return bad("MinChildWeight", p.MinChildWeight)
	}
	if !(p.MinSplitGain >= 0) || !finite(p.MinSplitGain) {
		return bad("MinSplitGain", p.MinSplitGain)
	}
	if p.Monotone != nil {
		if len(p.Monotone) != nf {
			return p, fmt.Errorf("xgoboost: Monotone has %d entries for %d features", len(p.Monotone), nf)
		}
		for j, c := range p.Monotone {
			if c < -1 || c > 1 {
				return p, fmt.Errorf("xgoboost: Monotone[%d] = %d, want -1, 0 or +1", j, c)
			}
		}
		p.Monotone = append([]int(nil), p.Monotone...)
	}
	if p.Subsample == 0 {
		p.Subsample = 1
	}
	if !(p.Subsample > 0 && p.Subsample <= 1) {
		return bad("Subsample", p.Subsample)
	}
	if p.ColSample == 0 {
		p.ColSample = 1
	}
	if !(p.ColSample > 0 && p.ColSample <= 1) {
		return bad("ColSample", p.ColSample)
	}
	if p.MaxBins == 0 {
		p.MaxBins = 256
	}
	if p.MaxBins < 2 || p.MaxBins > 65535 {
		return bad("MaxBins", p.MaxBins)
	}
	switch p.Method {
	case Auto:
		p.Method = Hist
		if n <= AutoExactMaxRows {
			p.Method = Exact
		}
	case Hist, Exact:
	default:
		return bad("Method", p.Method)
	}
	if p.EarlyStoppingRounds < 0 {
		return bad("EarlyStoppingRounds", p.EarlyStoppingRounds)
	}
	if p.EarlyStoppingRounds > 0 && p.Validation == nil {
		return p, fmt.Errorf("xgoboost: EarlyStoppingRounds needs a Validation set")
	}
	switch {
	case p.Workers == 0:
		p.Workers = runtime.GOMAXPROCS(0)
	case p.Workers < 0:
		return bad("Workers", p.Workers)
	}
	return p, nil
}
