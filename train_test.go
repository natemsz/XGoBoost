package xgoboost

import (
	"bytes"
	"math"
	"testing"
)

func TestDeterministic(t *testing.T) {
	methods(t, func(t *testing.T, method TreeMethod) {
		r := newRNG(51)
		X := uniformX(r, 3000, 6, -1, 1)
		y := label(X, func(x []float64) float64 { return x[0]*x[1] + x[2] + 0.1*r.NormFloat64() })
		p := DefaultParams()
		p.Method = method
		p.Rounds = 30
		p.Subsample = 0.7
		p.ColSample = 0.5
		p.Monotone = []int{0, 0, 1, 0, 0, 0}
		p.Seed = 42

		p.Workers = 1
		a := marshal(t, mustTrain(t, X, y, nil, p))
		p.Workers = 8
		b := marshal(t, mustTrain(t, X, y, nil, p))
		c := marshal(t, mustTrain(t, X, y, nil, p))
		if !bytes.Equal(a, b) || !bytes.Equal(b, c) {
			t.Fatal("same seed, different models")
		}
		p.Seed = 43
		if bytes.Equal(a, marshal(t, mustTrain(t, X, y, nil, p))) {
			t.Fatal("different seeds should sample differently")
		}
	})
}

func TestEarlyStopping(t *testing.T) {
	r := newRNG(52)
	gen := func(n int) ([][]float64, []float64) {
		X := uniformX(r, n, 5, 0, 1)
		return X, label(X, func(x []float64) float64 { return x[0] + r.NormFloat64() })
	}
	X, y := gen(400)
	vX, vy := gen(400)
	p := DefaultParams()
	p.Rounds = 1000
	p.LearningRate = 0.3
	p.MaxDepth = 8
	p.MinChildWeight = 0
	p.Validation = &ValidationSet{X: vX, Y: vy}
	p.EarlyStoppingRounds = 15
	m := mustTrain(t, X, y, nil, p)
	if m.BestIteration < 0 || m.BestIteration > 200 {
		t.Fatalf("best iteration %d: noisy data should overfit early", m.BestIteration)
	}
	if len(m.Trees) != m.BestIteration+1 {
		t.Fatalf("%d trees kept, best iteration %d", len(m.Trees), m.BestIteration)
	}
	raw := make([]float64, len(vX))
	for i, x := range vX {
		raw[i] = m.PredictRaw(x)
	}
	if s := p.Loss.Metric(vy, raw, nil); math.Abs(s-m.BestScore) > 1e-9 {
		t.Errorf("kept model scores %v on validation, recorded best %v", s, m.BestScore)
	}

	// Without early stopping every round is kept but the best is still
	// reported.
	p.EarlyStoppingRounds = 0
	p.Rounds = 100
	full := mustTrain(t, X, y, nil, p)
	if len(full.Trees) != 100 || full.BestIteration != m.BestIteration {
		t.Errorf("%d trees, best %d (early-stopped best %d)", len(full.Trees), full.BestIteration, m.BestIteration)
	}
}

func TestWeights(t *testing.T) {
	// Two contradictory copies of every row; the heavy copy should win.
	r := newRNG(53)
	X0 := uniformX(r, 300, 2, 0, 1)
	var X [][]float64
	var y, w []float64
	for _, x := range X0 {
		X = append(X, x, x)
		y = append(y, 1, -1)
		w = append(w, 9, 1)
	}
	p := DefaultParams()
	p.Rounds = 200
	m := mustTrain(t, X, y, w, p)
	if v := m.Predict(X0[0]); math.Abs(v-0.8) > 0.05 {
		t.Errorf("weighted mean is 0.8, predicted %v", v)
	}
	// Zero-weight rows are ignored.
	w2 := make([]float64, len(w))
	for i := range w2 {
		if i%2 == 0 {
			w2[i] = 1
		}
	}
	m = mustTrain(t, X, y, w2, p)
	if v := m.Predict(X0[0]); math.Abs(v-1) > 0.05 {
		t.Errorf("zero-weighted rows moved the prediction to %v", v)
	}
}

func TestFeatureImportance(t *testing.T) {
	r := newRNG(54)
	X := uniformX(r, 2000, 4, 0, 1)
	y := label(X, func(x []float64) float64 { return 5*x[2] + x[0] + 0.01*r.NormFloat64() })
	m := mustTrain(t, X, y, nil, DefaultParams())
	g := m.FeatureImportance(TotalGain)
	if !(g[2] > g[0] && g[0] > g[1] && g[0] > g[3]) {
		t.Errorf("gain importance %v: want feature 2 first, then 0", g)
	}
	n := m.FeatureImportance(SplitCount)
	total := 0.0
	for _, v := range n {
		total += v
	}
	if total == 0 {
		t.Error("no splits counted")
	}
}

func TestParamsValidation(t *testing.T) {
	X := [][]float64{{1, 2}, {2, 3}, {3, 4}}
	y := []float64{1, 2, 3}
	type tc struct {
		X [][]float64
		y []float64
		w []float64
		p func(*Params)
	}
	nop := func(*Params) {}
	cases := map[string]tc{
		"no rows":           {nil, nil, nil, nop},
		"no features":       {[][]float64{{}}, []float64{1}, nil, nop},
		"ragged":            {[][]float64{{1, 2}, {1}}, []float64{1, 2}, nil, nop},
		"label count":       {X, []float64{1}, nil, nop},
		"NaN label":         {X, []float64{1, math.NaN(), 3}, nil, nop},
		"Inf label":         {X, []float64{1, math.Inf(1), 3}, nil, nop},
		"weight count":      {X, y, []float64{1}, nop},
		"negative weight":   {X, y, []float64{1, -1, 1}, nop},
		"NaN weight":        {X, y, []float64{1, math.NaN(), 1}, nop},
		"zero weights":      {X, y, []float64{0, 0, 0}, nop},
		"rounds":            {X, y, nil, func(p *Params) { p.Rounds = -1 }},
		"learning rate":     {X, y, nil, func(p *Params) { p.LearningRate = -0.1 }},
		"depth":             {X, y, nil, func(p *Params) { p.MaxDepth = -1 }},
		"lambda":            {X, y, nil, func(p *Params) { p.Lambda = math.NaN() }},
		"min child weight":  {X, y, nil, func(p *Params) { p.MinChildWeight = -1 }},
		"gamma":             {X, y, nil, func(p *Params) { p.MinSplitGain = -1 }},
		"monotone length":   {X, y, nil, func(p *Params) { p.Monotone = []int{1} }},
		"monotone value":    {X, y, nil, func(p *Params) { p.Monotone = []int{2, 0} }},
		"subsample":         {X, y, nil, func(p *Params) { p.Subsample = 1.5 }},
		"colsample":         {X, y, nil, func(p *Params) { p.ColSample = -0.5 }},
		"bins":              {X, y, nil, func(p *Params) { p.MaxBins = 1 }},
		"method":            {X, y, nil, func(p *Params) { p.Method = 9 }},
		"es without valid":  {X, y, nil, func(p *Params) { p.EarlyStoppingRounds = 5 }},
		"valid width":       {X, y, nil, func(p *Params) { p.Validation = &ValidationSet{X: [][]float64{{1}}, Y: []float64{1}} }},
		"valid empty":       {X, y, nil, func(p *Params) { p.Validation = &ValidationSet{} }},
		"valid label count": {X, y, nil, func(p *Params) { p.Validation = &ValidationSet{X: X, Y: y[:1]} }},
		"workers":           {X, y, nil, func(p *Params) { p.Workers = -2 }},
	}
	for name, c := range cases {
		p := DefaultParams()
		c.p(&p)
		if _, err := Train(c.X, c.y, c.w, p); err == nil {
			t.Errorf("%s: want an error", name)
		}
	}
	// The zero Params is valid.
	if _, err := Train(X, y, nil, Params{}); err != nil {
		t.Errorf("zero Params: %v", err)
	}
}

// Degenerate inputs: one row, constant labels, constant features.
func TestDegenerate(t *testing.T) {
	methods(t, func(t *testing.T, method TreeMethod) {
		p := DefaultParams()
		p.Method = method
		m := mustTrain(t, [][]float64{{1}}, []float64{3}, nil, p)
		if v := m.Predict([]float64{1}); math.Abs(v-3) > 1e-9 {
			t.Errorf("one row: %v", v)
		}
		X := [][]float64{{1, 5}, {1, 6}, {1, 7}, {1, 8}}
		m = mustTrain(t, X, []float64{2, 2, 2, 2}, nil, p)
		if v := m.Predict([]float64{1, 100}); math.Abs(v-2) > 1e-9 {
			t.Errorf("constant labels: %v", v)
		}
		m = mustTrain(t, [][]float64{{1}, {1}, {1}}, []float64{1, 2, 3}, nil, p)
		if v := m.Predict([]float64{1}); math.Abs(v-2) > 1e-6 {
			t.Errorf("constant feature: %v", v)
		}
	})
}
