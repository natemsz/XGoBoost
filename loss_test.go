package xgoboost

import (
	"math"
	"testing"
)

// Each built-in loss's gradient and hessian against finite
// differences of its per-row loss.
func TestLossDerivatives(t *testing.T) {
	cases := []struct {
		name string
		loss Loss
		f    func(y, raw float64) float64 // the per-row loss
		ys   []float64
		raws []float64
		hess bool // check the hessian too (false where it is a surrogate)
	}{
		{"squared", SquaredError{}, func(y, r float64) float64 { return 0.5 * (r - y) * (r - y) },
			[]float64{-2, 0, 3.5}, []float64{-1, 0.3, 4}, true},
		{"logistic", Logistic{}, func(y, r float64) float64 { return softplus(r) - y*r },
			[]float64{0, 1, 0.3}, []float64{-3, -0.2, 0, 1.7, 5}, true},
		{"huber-inside", Huber{Delta: 2}, func(y, r float64) float64 { return 0.5 * (r - y) * (r - y) },
			[]float64{1}, []float64{0, 1.5, 2.2}, true},
		{"huber-outside", Huber{Delta: 0.5}, func(y, r float64) float64 {
			return 0.5 * (math.Abs(r-y) - 0.25)
		}, []float64{1}, []float64{-3, 3}, false},
		{"quantile", Quantile{Alpha: 0.8}, func(y, r float64) float64 {
			if y >= r {
				return 0.8 * (y - r)
			}
			return 0.2 * (r - y)
		}, []float64{1}, []float64{-2, 0.5, 1.5, 4}, false},
	}
	const eps = 1e-5
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			for _, y := range c.ys {
				for _, r := range c.raws {
					g, h := c.loss.GradHess(y, r)
					ng := (c.f(y, r+eps) - c.f(y, r-eps)) / (2 * eps)
					if math.Abs(g-ng) > 1e-5 {
						t.Errorf("y=%v raw=%v: grad %v, numeric %v", y, r, g, ng)
					}
					if h < 0 {
						t.Errorf("y=%v raw=%v: negative hessian %v", y, r, h)
					}
					if c.hess {
						nh := (c.f(y, r+eps) - 2*c.f(y, r) + c.f(y, r-eps)) / (eps * eps)
						if math.Abs(h-nh) > 1e-3 {
							t.Errorf("y=%v raw=%v: hess %v, numeric %v", y, r, h, nh)
						}
					}
				}
			}
		})
	}
}

func TestInitScores(t *testing.T) {
	y := []float64{1, 2, 3, 4, 10}
	w := []float64{1, 1, 1, 1, 1}
	if got := (SquaredError{}).InitScore(y, w); got != 4 {
		t.Errorf("squared init %v, want mean 4", got)
	}
	if got := (Huber{Delta: 1}).InitScore(y, w); got != 3 {
		t.Errorf("huber init %v, want median 3", got)
	}
	if got := (Quantile{Alpha: 0.8}).InitScore(y, w); got != 4 {
		t.Errorf("quantile init %v, want 4", got)
	}
	if got := (Quantile{Alpha: 0.5}).InitScore(y, []float64{0, 0, 0, 1, 5}); got != 10 {
		t.Errorf("weighted quantile init %v, want 10", got)
	}
	got := (Logistic{}).InitScore([]float64{0, 1, 1, 1}, nil)
	if math.Abs(sigmoid(got)-0.75) > 1e-12 {
		t.Errorf("logistic init %v maps to %v, want 0.75", got, sigmoid(got))
	}
}

func TestSquaredErrorFits(t *testing.T) {
	methods(t, func(t *testing.T, method TreeMethod) {
		r := newRNG(1)
		f := func(x []float64) float64 { return math.Sin(3*x[0]) + x[1]*x[1] - 0.5*x[2] }
		X := uniformX(r, 1500, 4, -1, 1)
		y := label(X, func(x []float64) float64 { return f(x) + 0.05*r.NormFloat64() })
		p := DefaultParams()
		p.Method = method
		p.Rounds = 300
		m := mustTrain(t, X, y, nil, p)
		Xt := uniformX(newRNG(2), 500, 4, -1, 1)
		if e := rmse(m.PredictBatch(Xt), label(Xt, f)); e > 0.12 {
			t.Errorf("test RMSE %v against the noiseless function", e)
		}
	})
}

// Huber should shrug off gross outliers that drag squared error.
func TestHuberRobustToOutliers(t *testing.T) {
	r := newRNG(3)
	f := func(x []float64) float64 { return 2 * x[0] }
	X := uniformX(r, 1000, 2, 0, 1)
	y := label(X, f)
	for i := range y {
		if i%10 == 0 {
			y[i] += 50 // 10% one-sided outliers
		}
	}
	p := DefaultParams()
	p.Rounds = 200
	p.MaxDepth = 3
	p.MinChildWeight = 30 // no leaf small enough to memorise single outliers
	sq := mustTrain(t, X, y, nil, p)
	p.Loss = Huber{Delta: 0.5}
	hu := mustTrain(t, X, y, nil, p)
	Xt := uniformX(newRNG(4), 400, 2, 0, 1)
	truth := label(Xt, f)
	eSq, eHu := rmse(sq.PredictBatch(Xt), truth), rmse(hu.PredictBatch(Xt), truth)
	if eHu > 0.5 || eHu > eSq/4 {
		t.Errorf("huber RMSE %v, squared %v: huber should be far closer to the clean signal", eHu, eSq)
	}
}

func TestLogisticClassifies(t *testing.T) {
	methods(t, func(t *testing.T, method TreeMethod) {
		r := newRNG(5)
		X := uniformX(r, 2000, 3, -1, 1)
		y := label(X, func(x []float64) float64 {
			if x[0]*x[0]+x[1]*x[1] < 0.5 {
				return 1
			}
			return 0
		})
		p := DefaultParams()
		p.Loss = Logistic{}
		p.Method = method
		p.Rounds = 150
		p.LearningRate = 0.3
		m := mustTrain(t, X, y, nil, p)
		correct := 0
		for i, x := range X {
			pr := m.Predict(x)
			if pr < 0 || pr > 1 {
				t.Fatalf("probability %v out of range", pr)
			}
			if (pr > 0.5) == (y[i] == 1) {
				correct++
			}
		}
		if acc := float64(correct) / float64(len(X)); acc < 0.95 {
			t.Errorf("training accuracy %v", acc)
		}
	})
}

// A quantile model's coverage should match its alpha.
func TestQuantileCoverage(t *testing.T) {
	for _, alpha := range []float64{0.1, 0.5, 0.9} {
		r := newRNG(6)
		X := uniformX(r, 4000, 2, 0, 1)
		y := label(X, func(x []float64) float64 { return 3*x[0] + (0.2+x[1])*r.NormFloat64() })
		p := DefaultParams()
		p.Loss = Quantile{Alpha: alpha}
		p.Rounds = 200
		p.MaxDepth = 4
		m := mustTrain(t, X, y, nil, p)
		Xt := uniformX(r, 4000, 2, 0, 1)
		yt := label(Xt, func(x []float64) float64 { return 3*x[0] + (0.2+x[1])*r.NormFloat64() })
		below := 0
		for i, x := range Xt {
			if yt[i] <= m.Predict(x) {
				below++
			}
		}
		if cov := float64(below) / float64(len(Xt)); math.Abs(cov-alpha) > 0.04 {
			t.Errorf("alpha %v: held-out coverage %v", alpha, cov)
		}
	}
}

func TestLossValidation(t *testing.T) {
	X := [][]float64{{1}, {2}, {3}}
	cases := map[string]struct {
		loss Loss
		y    []float64
	}{
		"logistic label 2":   {Logistic{}, []float64{0, 1, 2}},
		"huber zero delta":   {Huber{}, []float64{0, 1, 2}},
		"quantile alpha 1":   {Quantile{Alpha: 1}, []float64{0, 1, 2}},
		"quantile alpha NaN": {Quantile{Alpha: math.NaN()}, []float64{0, 1, 2}},
	}
	for name, c := range cases {
		p := DefaultParams()
		p.Loss = c.loss
		if _, err := Train(X, c.y, nil, p); err == nil {
			t.Errorf("%s: want an error", name)
		}
	}
}
