package xgoboost

import (
	"math"
	"testing"
)

// When missingness itself carries the signal, NaN rows should get
// their own value, not the median's.
func TestMissingInformative(t *testing.T) {
	methods(t, func(t *testing.T, method TreeMethod) {
		r := newRNG(21)
		X := uniformX(r, 1500, 2, 0, 1)
		y := make([]float64, len(X))
		for i, x := range X {
			if i%3 == 0 {
				x[0] = math.NaN()
				y[i] = 5
			} else {
				y[i] = x[0]
			}
		}
		p := DefaultParams()
		p.Method = method
		p.Rounds = 200
		m := mustTrain(t, X, y, nil, p)
		if v := m.Predict([]float64{math.NaN(), 0.5}); math.Abs(v-5) > 0.1 {
			t.Errorf("missing x0 predicts %v, want ~5", v)
		}
		if v := m.Predict([]float64{0.3, 0.5}); math.Abs(v-0.3) > 0.1 {
			t.Errorf("x0=0.3 predicts %v, want ~0.3", v)
		}
	})
}

// The learned default direction: missing rows that behave like the
// high values should be routed right, like the low values left.
func TestMissingDefaultDirection(t *testing.T) {
	for _, missingY := range []float64{0, 1} {
		r := newRNG(22)
		X := uniformX(r, 1000, 1, 0, 1)
		y := make([]float64, len(X))
		for i, x := range X {
			if x[0] > 0.5 {
				y[i] = 1
			}
			if i%4 == 0 {
				x[0] = math.NaN()
				y[i] = missingY
			}
		}
		p := DefaultParams()
		p.MaxDepth = 1 // one split: the default direction alone decides NaN
		p.Rounds = 100
		p.LearningRate = 0.3
		m := mustTrain(t, X, y, nil, p)
		if v := m.Predict([]float64{math.NaN()}); math.Abs(v-missingY) > 0.05 {
			t.Errorf("missing rows labelled %v: NaN predicts %v", missingY, v)
		}
		root := m.Trees[0].Nodes[0]
		if root.Leaf || root.DefaultLeft != (missingY == 0) {
			t.Errorf("missing rows labelled %v: root %+v", missingY, root)
		}
	}
}

// NaN at prediction time for a feature that was never missing in
// training, or a row too short, must still give a finite answer.
func TestMissingUnseen(t *testing.T) {
	r := newRNG(23)
	X := uniformX(r, 500, 3, 0, 1)
	y := label(X, func(x []float64) float64 { return x[0] + x[1] + x[2] })
	m := mustTrain(t, X, y, nil, DefaultParams())
	for _, x := range [][]float64{{math.NaN(), math.NaN(), math.NaN()}, {0.5}, nil} {
		if v := m.Predict(x); math.IsNaN(v) || math.IsInf(v, 0) {
			t.Errorf("Predict(%v) = %v", x, v)
		}
	}
	if m.Predict([]float64{0.5}) != m.Predict([]float64{0.5, math.NaN(), math.NaN()}) {
		t.Error("a short row should predict as if the tail were NaN")
	}
}

// A feature that is entirely missing must not break training.
func TestMissingWholeColumn(t *testing.T) {
	methods(t, func(t *testing.T, method TreeMethod) {
		r := newRNG(24)
		X := uniformX(r, 300, 2, 0, 1)
		for _, x := range X {
			x[1] = math.NaN()
		}
		y := label(X, func(x []float64) float64 { return x[0] })
		p := DefaultParams()
		p.Method = method
		m := mustTrain(t, X, y, nil, p)
		if imp := m.FeatureImportance(SplitCount); imp[1] != 0 {
			t.Errorf("all-NaN feature used in %v splits", imp[1])
		}
	})
}
