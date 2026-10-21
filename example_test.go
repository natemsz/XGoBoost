package xgoboost_test

import (
	"fmt"
	"math"
	"math/rand/v2"

	xgoboost "github.com/natemsz/xgoboost"
)

// Price rises with quality and falls with wear, whatever the noise
// says locally; the constraints make the model promise the same.
func Example() {
	r := rand.New(rand.NewPCG(1, 2))
	var X [][]float64
	var y []float64
	for i := 0; i < 2000; i++ {
		quality, wear, colour := r.Float64(), r.Float64(), r.Float64()
		if i%10 == 0 {
			wear = math.NaN() // unknown
		}
		price := 10*quality - 4*wear + math.Sin(6*colour) + r.NormFloat64()
		if math.IsNaN(wear) {
			price = 10*quality - 2 + math.Sin(6*colour) + r.NormFloat64()
		}
		X = append(X, []float64{quality, wear, colour})
		y = append(y, price)
	}

	p := xgoboost.DefaultParams()
	p.Monotone = []int{+1, -1, 0} // quality up, wear down, colour free
	p.Rounds = 200
	m, err := xgoboost.Train(X, y, nil, p)
	if err != nil {
		panic(err)
	}

	better := m.Predict([]float64{0.9, 0.2, 0.5})
	worse := m.Predict([]float64{0.8, 0.2, 0.5})
	fmt.Println("better quality never prices lower:", better >= worse)
	fmt.Printf("unknown wear: %.0f\n", m.Predict([]float64{0.5, math.NaN(), 0.5}))
	// Output:
	// better quality never prices lower: true
	// unknown wear: 3
}
