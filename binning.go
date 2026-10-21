package xgoboost

import (
	"math"
	"slices"
	"sort"
)

// missingBin marks a NaN in a binned column.
const missingBin = math.MaxUint16

// computeCuts returns ascending cut points for one feature column. Bin
// b holds values in (cuts[b-1], cuts[b]], so "bin <= b" is exactly
// "x <= cuts[b]" and a histogram split can be stored as an ordinary
// threshold. When the column has at most maxBins distinct values every
// gap between neighbours gets a cut (the histogram then sees exactly
// the splits exact greedy sees); otherwise cuts sit at weighted
// quantiles.
func computeCuts(col, w []float64, maxBins int) []float64 {
	type vw struct{ v, w float64 }
	vals := make([]vw, 0, len(col))
	for i, v := range col {
		if !math.IsNaN(v) {
			vals = append(vals, vw{v, w[i]})
		}
	}
	if len(vals) == 0 {
		return nil
	}
	slices.SortFunc(vals, func(a, b vw) int {
		switch {
		case a.v < b.v:
			return -1
		case a.v > b.v:
			return 1
		}
		return 0
	})
	// Collapse to distinct values with cumulative weight.
	distinct := []float64{vals[0].v}
	cum := []float64{0}
	total := 0.0
	for _, p := range vals {
		if p.v != distinct[len(distinct)-1] {
			distinct = append(distinct, p.v)
			cum = append(cum, total)
		}
		total += p.w
		cum[len(cum)-1] = total
	}
	if len(distinct) <= maxBins {
		cuts := make([]float64, len(distinct)-1)
		for i := range cuts {
			cuts[i] = midpoint(distinct[i], distinct[i+1])
		}
		return cuts
	}
	if total <= 0 { // all weights zero: fall back to equal counts
		for i := range cum {
			cum[i] = float64(i + 1)
		}
		total = float64(len(cum))
	}
	cuts := make([]float64, 0, maxBins-1)
	last := -1
	for k := 1; k < maxBins; k++ {
		target := total * float64(k) / float64(maxBins)
		i := sort.SearchFloat64s(cum, target) // first i with cum[i] >= target
		if i <= last {
			i = last + 1
		}
		if i >= len(distinct)-1 {
			break
		}
		cuts = append(cuts, midpoint(distinct[i], distinct[i+1]))
		last = i
	}
	return cuts
}

// binColumn maps a column to bin indices under cuts.
func binColumn(col, cuts []float64) []uint16 {
	out := make([]uint16, len(col))
	for i, v := range col {
		if math.IsNaN(v) {
			out[i] = missingBin
		} else {
			out[i] = uint16(sort.SearchFloat64s(cuts, v))
		}
	}
	return out
}

// midpoint returns a threshold t with a <= t < b for a < b: their
// midpoint when it is representable strictly below b, else a (which
// also handles infinities: a finite a below +Inf gives a; -Inf below
// anything gives -Inf).
func midpoint(a, b float64) float64 {
	m := a/2 + b/2
	if math.IsNaN(m) || m < a || m >= b {
		return a
	}
	return m
}
