package xgoboost

import (
	"math"
	"testing"
)

// checkMonotone sweeps each constrained feature over a grid (including
// ±Inf and values far outside the training range) from many base
// points, some with other features missing, and requires every raw
// prediction step to go the constrained way. No tolerance: each tree
// is exactly monotone and floating-point addition preserves order, so
// the ensemble must be too.
func checkMonotone(t *testing.T, m *Model, mono []int, lo, hi float64) {
	t.Helper()
	r := newRNG(99)
	grid := []float64{math.Inf(-1), -1e300, lo - 10}
	for k := 0; k <= 200; k++ {
		grid = append(grid, lo+(hi-lo)*float64(k)/200)
	}
	grid = append(grid, hi+10, 1e300, math.Inf(1))
	for j, c := range mono {
		if c == 0 {
			continue
		}
		for b := 0; b < 60; b++ {
			x := make([]float64, len(mono))
			for k := range x {
				x[k] = lo + (hi-lo)*r.Float64()
				if b%3 == 0 && k != j && r.Float64() < 0.3 {
					x[k] = math.NaN()
				}
			}
			prev, prevP := math.Inf(-c), math.Inf(-c)
			for _, v := range grid {
				x[j] = v
				raw, pr := m.PredictRaw(x), m.Predict(x)
				if (c > 0 && raw < prev) || (c < 0 && raw > prev) {
					t.Fatalf("feature %d (constraint %+d) not monotone at x=%v: raw %v after %v", j, c, v, raw, prev)
				}
				if (c > 0 && pr < prevP) || (c < 0 && pr > prevP) {
					t.Fatalf("feature %d: transformed prediction not monotone at x=%v: %v after %v", j, v, pr, prevP)
				}
				prev, prevP = raw, pr
			}
		}
	}
}

// Every loss and both split finders, on data whose truth agrees with
// the constraints in some features and fights them in others.
func TestMonotoneAllLosses(t *testing.T) {
	losses := map[string]Loss{
		"squared":  SquaredError{},
		"huber":    Huber{Delta: 0.3},
		"logistic": Logistic{},
		"quantile": Quantile{Alpha: 0.7},
	}
	mono := []int{+1, -1, +1, -1, 0}
	for name, loss := range losses {
		t.Run(name, func(t *testing.T) {
			methods(t, func(t *testing.T, method TreeMethod) {
				r := newRNG(11)
				X := uniformX(r, 1200, 5, -1, 1)
				y := label(X, func(x []float64) float64 {
					// Agrees on features 0 and 1, opposes on 2 and 3, and
					// interacts with the free feature 4.
					return x[0] - x[1] - math.Sin(4*x[2]) + x[3]*x[4] + 0.2*r.NormFloat64()
				})
				if loss.Name() == "logistic" {
					for i := range y {
						y[i] = math.Round(sigmoid(3 * y[i]))
					}
				}
				p := DefaultParams()
				p.Loss = loss
				p.Method = method
				p.Monotone = mono
				p.Rounds = 120
				p.LearningRate = 0.2
				p.Subsample = 0.8
				p.ColSample = 0.8
				p.MaxBins = 32
				m := mustTrain(t, X, y, nil, p)
				checkMonotone(t, m, mono, -1, 1)
			})
		})
	}
}

// Adversarial: the truth runs straight against the constraint. The
// model must stay monotone and so can learn (almost) nothing from the
// feature, while an unconstrained model learns the opposite trend.
func TestMonotoneAdversarial(t *testing.T) {
	methods(t, func(t *testing.T, method TreeMethod) {
		r := newRNG(12)
		X := uniformX(r, 1000, 2, 0, 1)
		y := label(X, func(x []float64) float64 { return -3*x[0] + 0.5*x[1] + 0.1*r.NormFloat64() })
		p := DefaultParams()
		p.Method = method
		p.Rounds = 200
		p.MinChildWeight = 0
		p.Lambda = 0

		free := mustTrain(t, X, y, nil, p)
		lo, hi := free.Predict([]float64{0.05, 0.5}), free.Predict([]float64{0.95, 0.5})
		if hi-lo > -2 {
			t.Fatalf("unconstrained model should learn the decreasing trend; got %v -> %v", lo, hi)
		}

		p.Monotone = []int{+1, 0}
		m := mustTrain(t, X, y, nil, p)
		checkMonotone(t, m, p.Monotone, 0, 1)
		lo, hi = m.Predict([]float64{0.05, 0.5}), m.Predict([]float64{0.95, 0.5})
		if hi-lo > 0.3 {
			t.Errorf("constrained model rises %v across a feature whose truth falls by 2.7", hi-lo)
		}

		// And the mirror image.
		for i := range y {
			y[i] = -y[i]
		}
		p.Monotone = []int{-1, 0}
		m = mustTrain(t, X, y, nil, p)
		checkMonotone(t, m, p.Monotone, 0, 1)
	})
}

// Adversarial interaction: the sign of x0's effect flips with x1, so a
// split on x0 that is ordered correctly in one branch is backwards in
// another. Checking each split locally is not enough; the envelope
// must carry the bound down through later splits on x1.
func TestMonotoneInteraction(t *testing.T) {
	methods(t, func(t *testing.T, method TreeMethod) {
		r := newRNG(13)
		X := uniformX(r, 1500, 3, -1, 1)
		y := label(X, func(x []float64) float64 { return 4*x[0]*x[1] + x[2] + 0.05*r.NormFloat64() })
		p := DefaultParams()
		p.Method = method
		p.Monotone = []int{+1, 0, +1}
		p.Rounds = 150
		p.MaxDepth = 8
		p.MinChildWeight = 0
		p.Lambda = 0
		p.LearningRate = 0.5
		m := mustTrain(t, X, y, nil, p)
		checkMonotone(t, m, p.Monotone, -1, 1)
	})
}

// Constraints must not stop the model learning a trend that agrees
// with them.
func TestMonotoneStillLearns(t *testing.T) {
	r := newRNG(14)
	f := func(x []float64) float64 { return 2*x[0] - math.Exp(x[1]) + math.Sin(3*x[2]) }
	X := uniformX(r, 2000, 3, -1, 1)
	y := label(X, func(x []float64) float64 { return f(x) + 0.05*r.NormFloat64() })
	p := DefaultParams()
	p.Monotone = []int{+1, -1, 0}
	p.Rounds = 300
	m := mustTrain(t, X, y, nil, p)
	checkMonotone(t, m, p.Monotone, -1, 1)
	Xt := uniformX(newRNG(15), 500, 3, -1, 1)
	if e := rmse(m.PredictBatch(Xt), label(Xt, f)); e > 0.15 {
		t.Errorf("test RMSE %v", e)
	}
}
