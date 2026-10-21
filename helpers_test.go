package xgoboost

import (
	"bytes"
	"math"
	"math/rand/v2"
	"testing"
)

func newRNG(seed uint64) *rand.Rand { return rand.New(rand.NewPCG(seed, 7)) }

// uniformX returns n rows of nf features drawn from U(lo, hi).
func uniformX(r *rand.Rand, n, nf int, lo, hi float64) [][]float64 {
	X := make([][]float64, n)
	for i := range X {
		X[i] = make([]float64, nf)
		for j := range X[i] {
			X[i][j] = lo + (hi-lo)*r.Float64()
		}
	}
	return X
}

func label(X [][]float64, f func(x []float64) float64) []float64 {
	y := make([]float64, len(X))
	for i, x := range X {
		y[i] = f(x)
	}
	return y
}

func mustTrain(t testing.TB, X [][]float64, y, w []float64, p Params) *Model {
	t.Helper()
	m, err := Train(X, y, w, p)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func rmse(a, b []float64) float64 {
	s := 0.0
	for i := range a {
		s += (a[i] - b[i]) * (a[i] - b[i])
	}
	return math.Sqrt(s / float64(len(a)))
}

func marshal(t testing.TB, m *Model) []byte {
	t.Helper()
	var buf bytes.Buffer
	if err := m.Save(&buf); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

// methods runs f under both split finders.
func methods(t *testing.T, f func(t *testing.T, method TreeMethod)) {
	for _, mm := range []struct {
		name string
		m    TreeMethod
	}{{"hist", Hist}, {"exact", Exact}} {
		t.Run(mm.name, func(t *testing.T) { f(t, mm.m) })
	}
}
