package xgoboost

import (
	"bytes"
	"encoding/json"
	"math"
	"strings"
	"testing"
)

func TestSerialiseRoundTrip(t *testing.T) {
	losses := []Loss{SquaredError{}, Huber{Delta: 0.7}, Logistic{}, Quantile{Alpha: 0.25}}
	for _, loss := range losses {
		t.Run(loss.Name(), func(t *testing.T) {
			r := newRNG(41)
			X := uniformX(r, 600, 3, -1, 1)
			for i, x := range X {
				if i%5 == 0 {
					x[1] = math.NaN()
				}
				if i%11 == 0 {
					x[2] = math.Inf(-1) // forces a -Inf threshold somewhere
				}
			}
			y := label(X, func(x []float64) float64 { return x[0] + 0.3*r.NormFloat64() })
			if loss.Name() == "logistic" {
				for i := range y {
					y[i] = math.Round(sigmoid(3 * y[i]))
				}
			}
			p := DefaultParams()
			p.Loss = loss
			p.Rounds = 40
			p.Monotone = []int{1, 0, -1}
			p.Validation = &ValidationSet{X: X[:100], Y: y[:100]}
			m := mustTrain(t, X, y, nil, p)

			b := marshal(t, m)
			m2, err := Load(bytes.NewReader(b))
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(b, marshal(t, m2)) {
				t.Fatal("re-encoding a loaded model changed its bytes")
			}
			if m2.Loss != loss {
				t.Fatalf("loss %#v came back as %#v", loss, m2.Loss)
			}
			if m2.BestIteration != m.BestIteration || m2.BestScore != m.BestScore {
				t.Fatal("best iteration lost")
			}
			Xt := uniformX(newRNG(42), 300, 3, -2, 2)
			Xt = append(Xt, []float64{math.NaN(), math.NaN(), math.Inf(-1)}, []float64{math.Inf(1), 0, math.Inf(-1)})
			for _, x := range Xt {
				if a, b := m.Predict(x), m2.Predict(x); a != b && !(math.IsNaN(a) && math.IsNaN(b)) {
					t.Fatalf("prediction %v became %v after a round trip", a, b)
				}
			}
		})
	}
}

func TestSerialiseRejectsMalformed(t *testing.T) {
	r := newRNG(43)
	X := uniformX(r, 200, 2, 0, 1)
	m := mustTrain(t, X, label(X, func(x []float64) float64 { return x[0] }), nil, DefaultParams())
	good := string(marshal(t, m))
	// Find the first split node's "left":N and point it back at node 0.
	var raw map[string]any
	if err := json.Unmarshal([]byte(good), &raw); err != nil {
		t.Fatal(err)
	}
	mutate := func(f func(map[string]any)) string {
		var v map[string]any
		_ = json.Unmarshal([]byte(good), &v)
		f(v)
		b, _ := json.Marshal(v)
		return string(b)
	}
	root := func(v map[string]any) map[string]any {
		return v["trees"].([]any)[0].(map[string]any)["nodes"].([]any)[0].(map[string]any)
	}
	cases := map[string]string{
		"cycle":         mutate(func(v map[string]any) { root(v)["left"] = 0 }),
		"child range":   mutate(func(v map[string]any) { root(v)["right"] = 1e6 }),
		"feature range": mutate(func(v map[string]any) { root(v)["feature"] = 7 }),
		"unknown loss":  mutate(func(v map[string]any) { v["loss"] = map[string]any{"name": "nope"} }),
		"bad quantile": mutate(func(v map[string]any) {
			v["loss"] = map[string]any{"name": "quantile", "params": map[string]any{"alpha": 2}}
		}),
		"format":        mutate(func(v map[string]any) { v["format"] = "xgoboost/99" }),
		"unknown field": mutate(func(v map[string]any) { v["surprise"] = 1 }),
		"bad threshold": mutate(func(v map[string]any) { root(v)["threshold"] = "sideways" }),
		"empty tree":    mutate(func(v map[string]any) { v["trees"] = []any{map[string]any{"nodes": []any{}}} }),
		"not json":      "{",
	}
	for name, s := range cases {
		if _, err := Load(strings.NewReader(s)); err == nil {
			t.Errorf("%s: loaded without error", name)
		}
	}
}

// cubeLoss is a custom loss: ¼(raw-y)⁴ with a floor on curvature.
type cubeLoss struct{ Floor float64 }

func (cubeLoss) Name() string                     { return "test_quartic" }
func (cubeLoss) InitScore(y, w []float64) float64 { return weightedMean(y, w) }
func (l cubeLoss) GradHess(y, raw float64) (float64, float64) {
	r := raw - y
	return r * r * r, math.Max(3*r*r, l.Floor)
}
func (cubeLoss) Transform(raw float64) float64      { return raw }
func (cubeLoss) Metric(y, raw, w []float64) float64 { return SquaredError{}.Metric(y, raw, w) }
func (l cubeLoss) LossParams() map[string]float64   { return map[string]float64{"floor": l.Floor} }

func TestCustomLoss(t *testing.T) {
	RegisterLoss("test_quartic", func(p map[string]float64) (Loss, error) {
		return cubeLoss{Floor: p["floor"]}, nil
	})
	r := newRNG(44)
	X := uniformX(r, 500, 2, 0, 1)
	y := label(X, func(x []float64) float64 { return 2 * x[0] })
	p := DefaultParams()
	p.Loss = cubeLoss{Floor: 0.1}
	p.Rounds = 200
	m := mustTrain(t, X, y, nil, p)
	if e := rmse(m.PredictBatch(X), y); e > 0.1 {
		t.Errorf("custom loss RMSE %v", e)
	}
	m2, err := Load(bytes.NewReader(marshal(t, m)))
	if err != nil {
		t.Fatal(err)
	}
	if m2.Loss != (cubeLoss{Floor: 0.1}) {
		t.Errorf("custom loss came back as %#v", m2.Loss)
	}
	defer func() {
		if recover() == nil {
			t.Error("registering a name twice should panic")
		}
	}()
	RegisterLoss("test_quartic", func(map[string]float64) (Loss, error) { return nil, nil })
}
