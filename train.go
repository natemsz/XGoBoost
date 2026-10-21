package xgoboost

import (
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"sync"
	"sync/atomic"
)

// Train fits a boosted ensemble to rows X (n rows of equal length;
// NaN marks a missing value, ±Inf is allowed), labels y and row
// weights w (nil means unit weights; weights must be finite and >= 0
// with a positive sum).
func Train(X [][]float64, y, w []float64, p Params) (*Model, error) {
	t, err := newTrainer(X, y, w, p)
	if err != nil {
		return nil, err
	}
	return t.run(), nil
}

// bucket accumulates one histogram bin, or one distinct value in exact
// mode.
type bucket struct {
	g, h float64
	n    int
}

func (b *bucket) add(g, h float64) { b.g += g; b.h += h; b.n++ }

type split struct {
	feature        int
	bin            int // hist mode: left iff bin <= this
	thr            float64
	defaultLeft    bool
	gl, hl, gr, hr float64
	vl, vr         float64
	gain           float64
}

type leafRec struct {
	id     int
	rows   []int
	lo, hi float64
}

type trainer struct {
	p     Params
	renew LeafRenewer
	X     [][]float64
	y, w  []float64
	n, nf int
	mono  []int

	valid *ValidationSet

	// Hist mode: bins[j][i] is row i's bin for feature j; a histogram
	// is one flat []bucket with feature j's bins at off[j] and its
	// missing bucket right after them. Bins k and k+1 of feature j
	// split at cuts[j][k].
	hist bool
	bins [][]uint16
	cuts [][]float64
	nbin []int
	off  []int
	hlen int
	pool [][]bucket

	// Exact mode: column-major copy of X, and every row index sorted
	// by each feature once (ascending, ties by row, NaN last). Each
	// tree filters order[j] to its sampled rows into sorted[j]; a
	// node at offset start with m rows owns sorted[j][start:start+m]
	// for every feature, and a split stably partitions those segments
	// between its children, so no node ever sorts.
	cols   [][]float64
	order  [][]int
	sorted [][]int
	bkbuf  [][]bucket
	thrbuf [][]float64

	// Per round.
	g, h     []float64
	raw      []float64
	feats    []int
	nodes    []Node
	leaves   []leafRec
	scratch  []int
	goLeft   []bool
	inSample []bool
}

func newTrainer(X [][]float64, y, w []float64, p Params) (*trainer, error) {
	n := len(X)
	if n == 0 {
		return nil, errors.New("xgoboost: no training rows")
	}
	nf := len(X[0])
	if nf == 0 {
		return nil, errors.New("xgoboost: rows have no features")
	}
	if err := checkData("training", X, y, w, nf); err != nil {
		return nil, err
	}
	if w == nil {
		w = make([]float64, n)
		for i := range w {
			w[i] = 1
		}
	}
	p, err := p.resolve(n, nf)
	if err != nil {
		return nil, err
	}
	if v, ok := p.Loss.(LossValidator); ok {
		if err := v.Validate(y); err != nil {
			return nil, err
		}
	}
	t := &trainer{p: p, X: X, y: y, w: w, n: n, nf: nf, mono: p.Monotone}
	if t.mono == nil {
		t.mono = make([]int, nf)
	}
	t.renew, _ = p.Loss.(LeafRenewer)
	if vs := p.Validation; vs != nil {
		if len(vs.X) == 0 {
			return nil, errors.New("xgoboost: empty validation set")
		}
		if err := checkData("validation", vs.X, vs.Y, vs.W, nf); err != nil {
			return nil, err
		}
		if v, ok := p.Loss.(LossValidator); ok {
			if err := v.Validate(vs.Y); err != nil {
				return nil, fmt.Errorf("validation set: %w", err)
			}
		}
		t.valid = vs
	}

	t.hist = p.Method == Hist
	if t.hist {
		t.bins = make([][]uint16, nf)
		t.cuts = make([][]float64, nf)
		t.nbin = make([]int, nf)
		t.off = make([]int, nf)
		t.parallel(nf, n*nf, func(j int) {
			col := t.column(j)
			t.cuts[j] = computeCuts(col, w, p.MaxBins)
			t.bins[j] = binColumn(col, t.cuts[j])
			t.nbin[j] = len(t.cuts[j]) + 1
		})
		for j := range t.off {
			t.off[j] = t.hlen
			t.hlen += t.nbin[j] + 1
		}
	} else {
		t.cols = make([][]float64, nf)
		t.order = make([][]int, nf)
		t.sorted = make([][]int, nf)
		t.bkbuf = make([][]bucket, nf)
		t.thrbuf = make([][]float64, nf)
		t.parallel(nf, n*nf, func(j int) {
			col := t.column(j)
			ord := make([]int, n)
			for i := range ord {
				ord[i] = i
			}
			slices.SortFunc(ord, func(a, b int) int {
				na, nb := math.IsNaN(col[a]), math.IsNaN(col[b])
				switch {
				case na != nb:
					if na {
						return 1
					}
					return -1
				case !na && col[a] < col[b]:
					return -1
				case !na && col[a] > col[b]:
					return 1
				}
				return a - b
			})
			t.cols[j], t.order[j] = col, ord
			t.sorted[j] = make([]int, 0, n)
		})
		t.goLeft = make([]bool, n)
	}
	t.inSample = make([]bool, n)
	t.g = make([]float64, n)
	t.h = make([]float64, n)
	t.raw = make([]float64, n)
	t.scratch = make([]int, n)
	return t, nil
}

func checkData(what string, X [][]float64, y, w []float64, nf int) error {
	n := len(X)
	for i, row := range X {
		if len(row) != nf {
			return fmt.Errorf("xgoboost: %s row %d has %d features, want %d", what, i, len(row), nf)
		}
	}
	if len(y) != n {
		return fmt.Errorf("xgoboost: %s has %d labels for %d rows", what, len(y), n)
	}
	for i, v := range y {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return fmt.Errorf("xgoboost: %s label %d is %v", what, i, v)
		}
	}
	if w == nil {
		return nil
	}
	if len(w) != n {
		return fmt.Errorf("xgoboost: %s has %d weights for %d rows", what, len(w), n)
	}
	sum := 0.0
	for i, v := range w {
		if !(v >= 0) || math.IsInf(v, 0) {
			return fmt.Errorf("xgoboost: %s weight %d is %v", what, i, v)
		}
		sum += v
	}
	if !(sum > 0) || math.IsInf(sum, 0) {
		return fmt.Errorf("xgoboost: %s weights sum to %v", what, sum)
	}
	return nil
}

func (t *trainer) column(j int) []float64 {
	col := make([]float64, t.n)
	for i, row := range t.X {
		col[i] = row[j]
	}
	return col
}

func (t *trainer) run() *Model {
	p := t.p
	m := &Model{
		NumFeatures:   t.nf,
		BaseScore:     p.Loss.InitScore(t.y, t.w),
		LearningRate:  p.LearningRate,
		Loss:          p.Loss,
		Monotone:      p.Monotone,
		BestIteration: -1,
	}
	for i := range t.raw {
		t.raw[i] = m.BaseScore
	}
	var vraw, vw []float64
	if t.valid != nil {
		vraw = make([]float64, len(t.valid.X))
		for i := range vraw {
			vraw[i] = m.BaseScore
		}
		vw = t.valid.W
		if vw == nil {
			vw = make([]float64, len(vraw))
			for i := range vw {
				vw[i] = 1
			}
		}
	}
	rng := rand.New(rand.NewPCG(p.Seed, 0x9e3779b97f4a7c15))

	for round := 0; round < p.Rounds; round++ {
		for i := range t.g {
			g, h := p.Loss.GradHess(t.y[i], t.raw[i])
			t.g[i], t.h[i] = t.w[i]*g, t.w[i]*h
		}
		rows := t.sampleRows(rng)
		t.feats = t.sampleFeats(rng)
		tree := t.buildTree(rows) // also updates raw for the sampled rows
		m.Trees = append(m.Trees, tree)
		if len(rows) < t.n {
			for _, r := range rows {
				t.inSample[r] = true
			}
			for i, x := range t.X {
				if !t.inSample[i] {
					t.raw[i] += tree.eval(x)
				}
				t.inSample[i] = false
			}
		}
		if t.valid == nil {
			continue
		}
		for i, x := range t.valid.X {
			vraw[i] += tree.eval(x)
		}
		s := p.Loss.Metric(t.valid.Y, vraw, vw)
		if m.BestIteration < 0 || s < m.BestScore {
			m.BestIteration, m.BestScore = round, s
		} else if p.EarlyStoppingRounds > 0 && round-m.BestIteration >= p.EarlyStoppingRounds {
			break
		}
	}
	if p.EarlyStoppingRounds > 0 && m.BestIteration >= 0 {
		m.Trees = m.Trees[:m.BestIteration+1]
	}
	return m
}

// sampleRows returns this round's rows in ascending order (a fresh
// slice: growing a tree reorders it).
func (t *trainer) sampleRows(rng *rand.Rand) []int {
	if t.p.Subsample >= 1 {
		rows := make([]int, t.n)
		for i := range rows {
			rows[i] = i
		}
		return rows
	}
	k := max(1, int(math.Round(t.p.Subsample*float64(t.n))))
	rows := rng.Perm(t.n)[:k]
	slices.Sort(rows)
	return rows
}

func (t *trainer) sampleFeats(rng *rand.Rand) []int {
	if t.p.ColSample >= 1 {
		f := make([]int, t.nf)
		for j := range f {
			f[j] = j
		}
		return f
	}
	k := max(1, int(math.Round(t.p.ColSample*float64(t.nf))))
	f := rng.Perm(t.nf)[:k]
	slices.Sort(f)
	return f
}

func (t *trainer) buildTree(rows []int) Tree {
	t.nodes = make([]Node, 0, 2*min(len(rows), 1<<min(t.p.MaxDepth, 16)))
	t.leaves = t.leaves[:0]
	G, H := 0.0, 0.0
	for _, r := range rows {
		G += t.g[r]
		H += t.h[r]
	}
	var hist []bucket
	if t.hist {
		hist = t.getHist()
		t.buildHist(hist, rows)
	} else {
		sampled := len(rows) < t.n
		if sampled {
			for _, r := range rows {
				t.inSample[r] = true
			}
		}
		for _, j := range t.feats {
			s := t.sorted[j][:0]
			for _, r := range t.order[j] {
				if !sampled || t.inSample[r] {
					s = append(s, r)
				}
			}
			t.sorted[j] = s
		}
		if sampled {
			for _, r := range rows {
				t.inSample[r] = false
			}
		}
	}
	t.grow(rows, 0, hist, G, H, 0, math.Inf(-1), math.Inf(1))
	if hist != nil {
		t.putHist(hist)
	}
	if t.renew != nil {
		for _, lf := range t.leaves {
			res := make([]float64, len(lf.rows))
			ws := make([]float64, len(lf.rows))
			for k, r := range lf.rows {
				res[k] = t.y[r] - t.raw[r]
				ws[k] = t.w[r]
			}
			v := t.renew.RenewLeaf(res, ws)
			if !math.IsNaN(v) && !math.IsInf(v, 0) {
				t.nodes[lf.id].Value = clamp(v, lf.lo, lf.hi)
			}
		}
	}
	for i := range t.nodes {
		if t.nodes[i].Leaf {
			t.nodes[i].Value *= t.p.LearningRate
		}
	}
	// The grower knows each sampled row's leaf: no need to evaluate.
	for _, lf := range t.leaves {
		v := t.nodes[lf.id].Value
		for _, r := range lf.rows {
			t.raw[r] += v
		}
	}
	return Tree{Nodes: t.nodes}
}

// grow grows the subtree over rows (whose gradient and hessian sums
// are G and H) and returns its root's index. lo and hi bound every
// leaf value below it: the monotone envelope. In hist mode hist is the
// rows' histogram, which grow may overwrite; in exact mode start is
// the node's offset into the sorted feature lists.
func (t *trainer) grow(rows []int, start int, hist []bucket, G, H float64, depth int, lo, hi float64) int {
	p := t.p
	id := len(t.nodes)
	v := clamp(newton(G, H, p.Lambda), lo, hi)
	t.nodes = append(t.nodes, Node{Leaf: true, Value: v, Cover: H})
	leaf := func() int {
		t.leaves = append(t.leaves, leafRec{id: id, rows: rows, lo: lo, hi: hi})
		return id
	}
	if depth >= p.MaxDepth || len(rows) < 2 || H < 2*p.MinChildWeight {
		return leaf()
	}
	s := t.bestSplit(rows, start, hist, G, H, score(G, H, v, p.Lambda), lo, hi)
	if s.feature < 0 {
		return leaf()
	}

	// Stable partitions: left rows to the front, right rows after,
	// each keeping its order (ascending row index for rows; feature
	// order for the exact-mode sorted segments).
	// Decisions read the contiguous bin or value column, not row-major
	// X; for bins, "bin <= s.bin" is the same test as "x <= s.thr".
	var nl int
	if t.hist {
		col, bin := t.bins[s.feature], uint16(s.bin)
		nl = t.partition(rows, func(r int) bool {
			b := col[r]
			if b == missingBin {
				return s.defaultLeft
			}
			return b <= bin
		})
	} else {
		col := t.cols[s.feature]
		for _, r := range rows {
			x := col[r]
			t.goLeft[r] = x <= s.thr || (math.IsNaN(x) && s.defaultLeft)
		}
		nl = t.partition(rows, func(r int) bool { return t.goLeft[r] })
	}
	if !t.hist && depth+1 < p.MaxDepth {
		for _, j := range t.feats {
			t.partition(t.sorted[j][start:start+len(rows)], func(r int) bool { return t.goLeft[r] })
		}
	}
	left, right := rows[:nl], rows[nl:]

	// Histogram subtraction: build the smaller child's, derive the
	// larger's from the parent's in place.
	var lh, rh, small []bucket
	if hist != nil && depth+1 < p.MaxDepth {
		small = t.getHist()
		if len(left) <= len(right) {
			t.buildHist(small, left)
			t.subHist(hist, small)
			lh, rh = small, hist
		} else {
			t.buildHist(small, right)
			t.subHist(hist, small)
			lh, rh = hist, small
		}
	}

	// The envelope: a +1 split caps the left subtree and floors the
	// right one at the midpoint of the children's values; -1 the
	// reverse. Unconstrained splits pass the parent's bounds down.
	llo, lhi, rlo, rhi := lo, hi, lo, hi
	mid := (s.vl + s.vr) / 2
	switch t.mono[s.feature] {
	case +1:
		lhi, rlo = mid, mid
	case -1:
		llo, rhi = mid, mid
	}
	l := t.grow(left, start, lh, s.gl, s.hl, depth+1, llo, lhi)
	r := t.grow(right, start+nl, rh, s.gr, s.hr, depth+1, rlo, rhi)
	if small != nil {
		t.putHist(small)
	}
	t.nodes[id] = Node{
		Feature: s.feature, Threshold: s.thr, DefaultLeft: s.defaultLeft,
		Left: l, Right: r, Gain: s.gain, Cover: H,
	}
	return id
}

// bestSplit searches the round's features in parallel and keeps the
// best split, ties going to the lower feature index.
func (t *trainer) bestSplit(rows []int, start int, hist []bucket, G, H, parentScore, lo, hi float64) split {
	res := make([]split, len(t.feats))
	t.parallel(len(t.feats), len(rows)*len(t.feats), func(k int) {
		j := t.feats[k]
		if t.hist {
			seg := hist[t.off[j] : t.off[j]+t.nbin[j]+1]
			res[k] = t.scan(j, seg[:t.nbin[j]], seg[t.nbin[j]], nil, len(rows), G, H, parentScore, lo, hi)
		} else {
			bk, thr, miss := t.exactBuckets(j, t.sorted[j][start:start+len(rows)])
			res[k] = t.scan(j, bk, miss, thr, len(rows), G, H, parentScore, lo, hi)
		}
	})
	best := split{feature: -1, gain: t.p.MinSplitGain}
	for _, s := range res {
		if s.feature >= 0 && s.gain > best.gain {
			best = s
		}
	}
	return best
}

// exactBuckets groups a node's rows, given in feature j's sorted order
// (NaN last), into one bucket per distinct value with the thresholds
// between consecutive buckets. The slices are per-feature buffers,
// reused from node to node.
func (t *trainer) exactBuckets(j int, seg []int) (bk []bucket, thr []float64, miss bucket) {
	col := t.cols[j]
	bk, thr = t.bkbuf[j][:0], t.thrbuf[j][:0]
	i := 0
	for i < len(seg) && !math.IsNaN(col[seg[i]]) {
		x := col[seg[i]]
		var b bucket
		for ; i < len(seg) && col[seg[i]] == x; i++ {
			b.add(t.g[seg[i]], t.h[seg[i]])
		}
		bk = append(bk, b)
		if i < len(seg) && !math.IsNaN(col[seg[i]]) {
			thr = append(thr, midpoint(x, col[seg[i]]))
		}
	}
	for ; i < len(seg); i++ {
		miss.add(t.g[seg[i]], t.h[seg[i]])
	}
	t.bkbuf[j], t.thrbuf[j] = bk, thr
	return bk, thr, miss
}

// partition stably moves the elements of s for which left holds to
// the front and returns how many there are.
func (t *trainer) partition(s []int, left func(r int) bool) int {
	nl, nr := 0, 0
	for _, r := range s {
		if left(r) {
			s[nl] = r
			nl++
		} else {
			t.scratch[nr] = r
			nr++
		}
	}
	copy(s[nl:], t.scratch[:nr])
	return nl
}

// scan finds feature j's best split over ordered buckets bk plus a
// missing-value bucket. thr gives exact-mode thresholds; nil means
// histogram cuts. Each candidate is tried with the missing rows on
// either side; when there are none, unseen missing values later
// follow the child with more cover.
func (t *trainer) scan(j int, bk []bucket, miss bucket, thr []float64, n int, G, H, parentScore, lo, hi float64) split {
	best := split{feature: -1, gain: math.Inf(-1)}
	lam, mcw, c := t.p.Lambda, t.p.MinChildWeight, t.mono[j]
	var gl, hl float64
	nl := 0
	for k := 0; k+1 < len(bk); k++ {
		gl += bk[k].g
		hl += bk[k].h
		nl += bk[k].n
		if bk[k].n == 0 {
			continue // same partition as the previous candidate
		}
		if n-nl-miss.n == 0 {
			break // no non-missing rows remain on the right
		}
		gr, hr := G-gl-miss.g, H-hl-miss.h
		for d := 0; d < 2; d++ {
			missLeft := d == 1
			if missLeft && miss.n == 0 {
				break
			}
			GL, HL, GR, HR := gl, hl, gr+miss.g, hr+miss.h
			if missLeft {
				GL, HL, GR, HR = gl+miss.g, hl+miss.h, gr, hr
			}
			if HL < mcw || HR < mcw {
				continue
			}
			vl := clamp(newton(GL, HL, lam), lo, hi)
			vr := clamp(newton(GR, HR, lam), lo, hi)
			if (c > 0 && vl > vr) || (c < 0 && vl < vr) {
				continue
			}
			gain := 0.5 * (score(GL, HL, vl, lam) + score(GR, HR, vr, lam) - parentScore)
			if gain > best.gain {
				dl := missLeft
				if miss.n == 0 {
					dl = HL > HR
				}
				th := 0.0
				if thr != nil {
					th = thr[k]
				} else {
					th = t.cuts[j][k]
				}
				best = split{feature: j, bin: k, thr: th, defaultLeft: dl,
					gl: GL, hl: HL, gr: GR, hr: HR, vl: vl, vr: vr, gain: gain}
			}
		}
	}
	return best
}

func (t *trainer) buildHist(hist []bucket, rows []int) {
	t.parallel(len(t.feats), len(rows)*len(t.feats), func(k int) {
		j := t.feats[k]
		nb := t.nbin[j]
		seg := hist[t.off[j] : t.off[j]+nb+1]
		clear(seg)
		col := t.bins[j]
		for _, r := range rows {
			b := int(col[r])
			if b == missingBin {
				b = nb
			}
			seg[b].add(t.g[r], t.h[r])
		}
	})
}

// subHist turns parent into parent - child over the active features.
func (t *trainer) subHist(parent, child []bucket) {
	for _, j := range t.feats {
		for b := t.off[j]; b <= t.off[j]+t.nbin[j]; b++ {
			parent[b].g -= child[b].g
			parent[b].h -= child[b].h
			parent[b].n -= child[b].n
		}
	}
}

func (t *trainer) getHist() []bucket {
	if k := len(t.pool); k > 0 {
		h := t.pool[k-1]
		t.pool = t.pool[:k-1]
		return h
	}
	return make([]bucket, t.hlen)
}

func (t *trainer) putHist(h []bucket) { t.pool = append(t.pool, h) }

// parallel calls fn(0..n-1) on up to Workers goroutines, or inline
// when work is too small to be worth it. Callers write results by
// index, so the outcome never depends on scheduling.
func (t *trainer) parallel(n, work int, fn func(k int)) {
	w := min(t.p.Workers, n)
	if w <= 1 || work < 1<<15 {
		for k := 0; k < n; k++ {
			fn(k)
		}
		return
	}
	var next atomic.Int64
	var wg sync.WaitGroup
	wg.Add(w)
	for i := 0; i < w; i++ {
		go func() {
			defer wg.Done()
			for {
				k := int(next.Add(1) - 1)
				if k >= n {
					return
				}
				fn(k)
			}
		}()
	}
	wg.Wait()
}

// newton is the leaf value minimising the second-order objective.
func newton(G, H, lambda float64) float64 {
	if H+lambda <= 0 {
		return 0
	}
	return -G / (H + lambda)
}

// score is twice the objective reduction a leaf of value v achieves
// over value 0: -(2Gv + (H+λ)v²). At the unconstrained optimum it is
// G²/(H+λ); for a clamped v it is the reduction actually achieved.
func score(G, H, v, lambda float64) float64 {
	return -(2*G*v + (H+lambda)*v*v)
}

// clamp bounds v to [lo, hi]. Plain comparisons: v is never NaN here,
// and math.Min/Max's NaN and signed-zero handling costs a third of
// split search.
func clamp(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
