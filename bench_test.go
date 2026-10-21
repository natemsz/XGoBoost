package xgoboost

import (
	"math"
	"testing"
)

func benchData(n, nf int) ([][]float64, []float64) {
	r := newRNG(61)
	X := uniformX(r, n, nf, -1, 1)
	y := label(X, func(x []float64) float64 {
		return math.Sin(3*x[0]) + x[1]*x[2] - x[3] + 0.1*r.NormFloat64()
	})
	return X, y
}

func benchTrain(b *testing.B, n, nf int, method TreeMethod, mono bool) {
	X, y := benchData(n, nf)
	p := DefaultParams()
	p.Method = method
	p.Rounds = 50
	if mono {
		p.Monotone = make([]int, nf)
		p.Monotone[0], p.Monotone[3] = +1, -1
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := Train(X, y, nil, p); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkTrainHist100k(b *testing.B)    { benchTrain(b, 100_000, 20, Hist, false) }
func BenchmarkTrainHist20k(b *testing.B)     { benchTrain(b, 20_000, 20, Hist, false) }
func BenchmarkTrainHist20kMono(b *testing.B) { benchTrain(b, 20_000, 20, Hist, true) }
func BenchmarkTrainExact20k(b *testing.B)    { benchTrain(b, 20_000, 20, Exact, false) }
func BenchmarkTrainExact2k(b *testing.B)     { benchTrain(b, 2_000, 10, Exact, false) }

func BenchmarkPredictBatch(b *testing.B) {
	X, y := benchData(20_000, 20)
	p := DefaultParams()
	p.Rounds = 100
	m, err := Train(X, y, nil, p)
	if err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.PredictBatch(X)
	}
	b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(b.N*len(X)), "ns/row")
}
