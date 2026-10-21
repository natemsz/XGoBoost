package xgoboost

import (
	"math"
	"testing"
)

// On data with at most MaxBins distinct values per feature, the
// histogram finder sees exactly the candidate splits exact greedy
// sees, so both must grow the same trees: same shapes, same features,
// same training predictions (thresholds may differ inside gaps that
// hold no training value at a node).
func TestHistMatchesExact(t *testing.T) {
	for _, loss := range []Loss{SquaredError{}, Logistic{}, Huber{Delta: 0.5}} {
		t.Run(loss.Name(), func(t *testing.T) {
			r := newRNG(31)
			X := uniformX(r, 400, 4, 0, 1)
			for i, x := range X {
				for j := range x {
					x[j] = math.Round(x[j]*40) / 40 // 41 levels
				}
				if i%7 == 0 {
					x[i%4] = math.NaN()
				}
			}
			y := label(X, func(x []float64) float64 {
				v := x[0] - 2*x[1]*x[2] + 0.1*r.NormFloat64()
				if math.IsNaN(v) {
					v = 0.3
				}
				return v
			})
			if loss.Name() == "logistic" {
				for i := range y {
					y[i] = math.Round(sigmoid(4 * y[i]))
				}
			}
			p := DefaultParams()
			p.Loss = loss
			p.Rounds = 30
			p.MaxDepth = 4
			p.Monotone = []int{+1, 0, 0, 0}
			p.Method = Hist
			h := mustTrain(t, X, y, nil, p)
			p.Method = Exact
			e := mustTrain(t, X, y, nil, p)

			if len(h.Trees) != len(e.Trees) {
				t.Fatalf("%d hist trees, %d exact", len(h.Trees), len(e.Trees))
			}
			for k := range h.Trees {
				hn, en := h.Trees[k].Nodes, e.Trees[k].Nodes
				if len(hn) != len(en) {
					t.Fatalf("tree %d: %d hist nodes, %d exact", k, len(hn), len(en))
				}
				for i := range hn {
					a, b := hn[i], en[i]
					// Two features can split a small node into the same
					// rows with the same gain; summation order then picks
					// one. Anything else is a real disagreement.
					tie := math.Abs(a.Gain-b.Gain) <= 1e-9*(1+math.Abs(b.Gain))
					if a.Leaf != b.Leaf || a.Left != b.Left || !tie ||
						(a.Feature != b.Feature || a.DefaultLeft != b.DefaultLeft) && a.Cover > 10 {
						t.Fatalf("tree %d node %d differs: hist %+v exact %+v", k, i, a, b)
					}
				}
			}
			for i, x := range X {
				a, b := h.PredictRaw(x), e.PredictRaw(x)
				if math.Abs(a-b) > 1e-9*(1+math.Abs(b)) {
					t.Fatalf("row %d: hist %v exact %v", i, a, b)
				}
			}
		})
	}
}

// With many distinct values the histogram is an approximation, but a
// good one.
func TestHistApproximatesExact(t *testing.T) {
	r := newRNG(32)
	f := func(x []float64) float64 { return math.Sin(5*x[0]) + x[1] }
	X := uniformX(r, 3000, 2, 0, 1)
	y := label(X, func(x []float64) float64 { return f(x) + 0.05*r.NormFloat64() })
	p := DefaultParams()
	p.Rounds = 200
	p.Method = Exact
	e := mustTrain(t, X, y, nil, p)
	p.Method = Hist
	p.MaxBins = 64
	h := mustTrain(t, X, y, nil, p)
	Xt := uniformX(newRNG(33), 1000, 2, 0, 1)
	truth := label(Xt, f)
	eh, ee := rmse(h.PredictBatch(Xt), truth), rmse(e.PredictBatch(Xt), truth)
	if eh > ee*1.25+0.01 {
		t.Errorf("hist RMSE %v vs exact %v", eh, ee)
	}
}

func TestComputeCuts(t *testing.T) {
	// Few distinct values: a cut in every gap.
	col := []float64{3, 1, 2, 2, math.NaN(), 1}
	w := []float64{1, 1, 1, 1, 1, 1}
	cuts := computeCuts(col, w, 256)
	if len(cuts) != 2 || cuts[0] != 1.5 || cuts[1] != 2.5 {
		t.Fatalf("cuts %v", cuts)
	}
	bins := binColumn(col, cuts)
	want := []uint16{2, 0, 1, 1, missingBin, 0}
	for i := range want {
		if bins[i] != want[i] {
			t.Fatalf("bins %v, want %v", bins, want)
		}
	}

	// Many distinct values: roughly equal-weight bins, never more than
	// maxBins.
	r := newRNG(34)
	col = make([]float64, 10000)
	w = make([]float64, len(col))
	for i := range col {
		col[i] = r.ExpFloat64()
		w[i] = 1
	}
	cuts = computeCuts(col, w, 16)
	if len(cuts) != 15 {
		t.Fatalf("%d cuts for 16 bins", len(cuts))
	}
	counts := make([]int, 16)
	for _, b := range binColumn(col, cuts) {
		counts[b]++
	}
	for b, c := range counts {
		if c < 500 || c > 750 {
			t.Errorf("bin %d holds %d of 10000 rows", b, c)
		}
	}

	// Infinities get usable thresholds.
	cuts = computeCuts([]float64{math.Inf(-1), 0, math.Inf(1)}, []float64{1, 1, 1}, 256)
	if len(cuts) != 2 || !math.IsInf(cuts[0], -1) || cuts[1] != 0 {
		t.Errorf("cuts with infinities %v", cuts)
	}
}

func TestMidpoint(t *testing.T) {
	inf := math.Inf(1)
	for _, c := range [][2]float64{
		{1, 2}, {-inf, 0}, {0, inf}, {-inf, inf}, {1, math.Nextafter(1, 2)},
		{-math.MaxFloat64, math.MaxFloat64}, {-0.0, 5e-324},
	} {
		a, b := c[0], c[1]
		m := midpoint(a, b)
		if !(a <= m && m < b) {
			t.Errorf("midpoint(%v, %v) = %v", a, b, m)
		}
	}
}
