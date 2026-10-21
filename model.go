package xgoboost

import "math"

// Node is one node of a Tree. A split node sends x left when
// x[Feature] <= Threshold, right when greater, and by DefaultLeft when
// x[Feature] is NaN (or x is shorter than Feature+1). Children always
// have larger indices than their parent; node 0 is the root.
type Node struct {
	Leaf bool
	// Value is a leaf's contribution to the raw prediction, already
	// scaled by the learning rate.
	Value       float64
	Feature     int
	Threshold   float64
	DefaultLeft bool
	Left, Right int
	// Gain is the split's loss reduction (training only; used for
	// feature importance).
	Gain float64
	// Cover is the hessian sum of the training rows that reached the
	// node.
	Cover float64
}

// Tree is one regression tree, stored as a flat node slice.
type Tree struct {
	Nodes []Node
}

func (t *Tree) eval(x []float64) float64 {
	i := 0
	for {
		nd := &t.Nodes[i]
		if nd.Leaf {
			return nd.Value
		}
		v := math.NaN()
		if nd.Feature < len(x) {
			v = x[nd.Feature]
		}
		switch {
		case v <= nd.Threshold:
			i = nd.Left
		case v > nd.Threshold:
			i = nd.Right
		case nd.DefaultLeft: // NaN
			i = nd.Left
		default:
			i = nd.Right
		}
	}
}

// Model is a trained ensemble. It is safe for concurrent use by
// multiple goroutines once trained or loaded.
type Model struct {
	NumFeatures  int
	BaseScore    float64 // the raw prediction before any tree
	LearningRate float64 // informational: leaf values are pre-scaled
	Loss         Loss
	Monotone     []int // nil when unconstrained
	Trees        []Tree
	// BestIteration is the 0-based round with the lowest validation
	// metric, and BestScore that metric. BestIteration is -1 when
	// training had no validation set.
	BestIteration int
	BestScore     float64
}

// PredictRaw returns the raw prediction (the margin) for x: the base
// score plus every tree's leaf value. NaN, or a feature index beyond
// len(x), counts as missing.
func (m *Model) PredictRaw(x []float64) float64 {
	p := m.BaseScore
	for i := range m.Trees {
		p += m.Trees[i].eval(x)
	}
	return p
}

// Predict returns the prediction for x on the output scale: the raw
// prediction passed through Loss.Transform (a probability for
// Logistic).
func (m *Model) Predict(x []float64) float64 {
	return m.Loss.Transform(m.PredictRaw(x))
}

// PredictBatch returns Predict for every row of X.
func (m *Model) PredictBatch(X [][]float64) []float64 {
	out := make([]float64, len(X))
	for i, x := range X {
		out[i] = m.Predict(x)
	}
	return out
}

// ImportanceType selects what FeatureImportance sums.
type ImportanceType int

const (
	// TotalGain sums the loss reduction of every split on a feature.
	TotalGain ImportanceType = iota
	// SplitCount counts the splits on a feature.
	SplitCount
)

// FeatureImportance returns one unnormalised score per feature.
func (m *Model) FeatureImportance(kind ImportanceType) []float64 {
	imp := make([]float64, m.NumFeatures)
	for _, t := range m.Trees {
		for _, nd := range t.Nodes {
			if nd.Leaf {
				continue
			}
			if kind == SplitCount {
				imp[nd.Feature]++
			} else {
				imp[nd.Feature] += nd.Gain
			}
		}
	}
	return imp
}
