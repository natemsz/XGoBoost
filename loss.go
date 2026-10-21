package xgoboost

import (
	"fmt"
	"math"
	"slices"
	"sync"
)

// Loss is a training objective. Boosting works on raw predictions (the
// margin); Transform maps a raw prediction to the output scale.
//
// Train always passes non-nil weight slices. Implementations used
// elsewhere should treat a nil w as unit weights.
type Loss interface {
	// Name identifies the loss in serialised models. A custom loss must
	// be registered under this name with RegisterLoss before a model
	// using it can be loaded.
	Name() string
	// InitScore is the constant raw prediction boosting starts from.
	InitScore(y, w []float64) float64
	// GradHess returns the first and second derivative of one row's
	// loss with respect to its raw prediction. hess must be >= 0.
	GradHess(y, raw float64) (grad, hess float64)
	// Transform maps a raw prediction to the output scale. It must be
	// non-decreasing, so that monotone constraints carry over to
	// Model.Predict.
	Transform(raw float64) float64
	// Metric scores raw predictions against labels; lower is better.
	// It drives early stopping.
	Metric(y, raw, w []float64) float64
}

// LeafRenewer is implemented by losses whose Newton step is a poor
// leaf value (the quantile loss has zero curvature). After a tree is
// grown, each leaf's value is replaced by RenewLeaf over the residuals
// y - raw of the training rows in it, then clamped to the leaf's
// monotone envelope, so constraints still hold.
type LeafRenewer interface {
	RenewLeaf(residual, w []float64) float64
}

// LossValidator is implemented by losses that restrict their labels or
// their own parameters. Train calls Validate before fitting.
type LossValidator interface {
	Validate(y []float64) error
}

// ParamLoss is implemented by losses with parameters that must survive
// serialisation. The map is passed back to the loss's LossDecoder.
type ParamLoss interface {
	LossParams() map[string]float64
}

// LossDecoder rebuilds a loss from its serialised parameters.
type LossDecoder func(params map[string]float64) (Loss, error)

var (
	lossMu       sync.RWMutex
	lossDecoders = map[string]LossDecoder{
		"squared_error": func(map[string]float64) (Loss, error) { return SquaredError{}, nil },
		"logistic":      func(map[string]float64) (Loss, error) { return Logistic{}, nil },
		"huber": func(p map[string]float64) (Loss, error) {
			l := Huber{Delta: p["delta"]}
			return l, l.Validate(nil)
		},
		"quantile": func(p map[string]float64) (Loss, error) {
			l := Quantile{Alpha: p["alpha"]}
			return l, l.Validate(nil)
		},
	}
)

// RegisterLoss makes a custom loss loadable from serialised models. It
// panics if name is empty, dec is nil, or name is already registered.
func RegisterLoss(name string, dec LossDecoder) {
	lossMu.Lock()
	defer lossMu.Unlock()
	if name == "" || dec == nil {
		panic("xgoboost: RegisterLoss with empty name or nil decoder")
	}
	if _, dup := lossDecoders[name]; dup {
		panic("xgoboost: RegisterLoss called twice for " + name)
	}
	lossDecoders[name] = dec
}

func decodeLoss(name string, params map[string]float64) (Loss, error) {
	lossMu.RLock()
	dec, ok := lossDecoders[name]
	lossMu.RUnlock()
	if !ok {
		return nil, fmt.Errorf("xgoboost: unknown loss %q (register it with RegisterLoss)", name)
	}
	return dec(params)
}

// ---------------------------------------------------------------------

// SquaredError is ½(raw - y)², the default. Its metric is RMSE.
type SquaredError struct{}

func (SquaredError) Name() string                           { return "squared_error" }
func (SquaredError) InitScore(y, w []float64) float64       { return weightedMean(y, w) }
func (SquaredError) GradHess(y, raw float64) (g, h float64) { return raw - y, 1 }
func (SquaredError) Transform(raw float64) float64          { return raw }

func (SquaredError) Metric(y, raw, w []float64) float64 {
	s, sw := 0.0, 0.0
	for i := range y {
		r := raw[i] - y[i]
		s += wt(w, i) * r * r
		sw += wt(w, i)
	}
	return math.Sqrt(s / sw)
}

// Huber is squared error within Delta of the label and absolute error
// beyond it, so outliers pull the fit with bounded force. Its gradient
// is the residual clipped to ±Delta and its hessian is taken as 1 (the
// true curvature, 0 outside the band, would make Newton steps
// unbounded). The metric is the mean Huber loss.
type Huber struct {
	Delta float64
}

func (l Huber) Name() string                     { return "huber" }
func (l Huber) InitScore(y, w []float64) float64 { return weightedQuantile(y, w, 0.5) }
func (l Huber) Transform(raw float64) float64    { return raw }
func (l Huber) LossParams() map[string]float64   { return map[string]float64{"delta": l.Delta} }

func (l Huber) Validate([]float64) error {
	if !(l.Delta > 0) || math.IsInf(l.Delta, 0) {
		return fmt.Errorf("xgoboost: Huber.Delta must be positive and finite, got %v", l.Delta)
	}
	return nil
}

func (l Huber) GradHess(y, raw float64) (g, h float64) {
	return math.Max(-l.Delta, math.Min(l.Delta, raw-y)), 1
}

func (l Huber) Metric(y, raw, w []float64) float64 {
	s, sw := 0.0, 0.0
	for i := range y {
		r := math.Abs(raw[i] - y[i])
		v := 0.5 * r * r
		if r > l.Delta {
			v = l.Delta * (r - 0.5*l.Delta)
		}
		s += wt(w, i) * v
		sw += wt(w, i)
	}
	return s / sw
}

// Logistic is binary cross-entropy on labels in [0, 1]; raw predictions
// are log-odds and Transform is the logistic function. The metric is
// the mean log loss.
type Logistic struct{}

func (Logistic) Name() string                  { return "logistic" }
func (Logistic) Transform(raw float64) float64 { return sigmoid(raw) }

func (Logistic) Validate(y []float64) error {
	for i, v := range y {
		if v < 0 || v > 1 {
			return fmt.Errorf("xgoboost: Logistic needs labels in [0, 1], row %d is %v", i, v)
		}
	}
	return nil
}

func (Logistic) InitScore(y, w []float64) float64 {
	p := math.Min(math.Max(weightedMean(y, w), 1e-6), 1-1e-6)
	return math.Log(p / (1 - p))
}

func (Logistic) GradHess(y, raw float64) (g, h float64) {
	p := sigmoid(raw)
	return p - y, math.Max(p*(1-p), 1e-16)
}

func (Logistic) Metric(y, raw, w []float64) float64 {
	s, sw := 0.0, 0.0
	for i := range y {
		s += wt(w, i) * (softplus(raw[i]) - y[i]*raw[i])
		sw += wt(w, i)
	}
	return s / sw
}

// Quantile is the pinball loss for the Alpha quantile of y given x. Its
// curvature is zero, so trees are grown on gradients with unit
// hessians and each leaf is then renewed to the Alpha quantile of its
// rows' residuals. The metric is the mean pinball loss.
type Quantile struct {
	Alpha float64
}

func (l Quantile) Name() string                   { return "quantile" }
func (l Quantile) Transform(raw float64) float64  { return raw }
func (l Quantile) LossParams() map[string]float64 { return map[string]float64{"alpha": l.Alpha} }

func (l Quantile) Validate([]float64) error {
	if !(l.Alpha > 0 && l.Alpha < 1) {
		return fmt.Errorf("xgoboost: Quantile.Alpha must be in (0, 1), got %v", l.Alpha)
	}
	return nil
}

func (l Quantile) InitScore(y, w []float64) float64 { return weightedQuantile(y, w, l.Alpha) }

func (l Quantile) GradHess(y, raw float64) (g, h float64) {
	if raw < y {
		return -l.Alpha, 1
	}
	return 1 - l.Alpha, 1
}

func (l Quantile) RenewLeaf(residual, w []float64) float64 {
	return weightedQuantile(residual, w, l.Alpha)
}

func (l Quantile) Metric(y, raw, w []float64) float64 {
	s, sw := 0.0, 0.0
	for i := range y {
		r := y[i] - raw[i]
		v := l.Alpha * r
		if r < 0 {
			v = (l.Alpha - 1) * r
		}
		s += wt(w, i) * v
		sw += wt(w, i)
	}
	return s / sw
}

// ---------------------------------------------------------------------

func wt(w []float64, i int) float64 {
	if w == nil {
		return 1
	}
	return w[i]
}

func weightedMean(y, w []float64) float64 {
	s, sw := 0.0, 0.0
	for i, v := range y {
		s += wt(w, i) * v
		sw += wt(w, i)
	}
	if sw <= 0 {
		return 0
	}
	return s / sw
}

// weightedQuantile is the smallest v[i] at which the cumulative weight
// of the sorted values reaches q of the total. Zero-weight values are
// ignored unless every weight is zero.
func weightedQuantile(v, w []float64, q float64) float64 {
	if len(v) == 0 {
		return 0
	}
	idx := make([]int, 0, len(v))
	total := 0.0
	for i := range v {
		if wt(w, i) > 0 {
			idx = append(idx, i)
			total += wt(w, i)
		}
	}
	if len(idx) == 0 {
		w = nil
		for i := range v {
			idx = append(idx, i)
		}
		total = float64(len(v))
	}
	slices.SortStableFunc(idx, func(a, b int) int {
		switch {
		case v[a] < v[b]:
			return -1
		case v[a] > v[b]:
			return 1
		}
		return 0
	})
	target, cum := q*total, 0.0
	for _, i := range idx {
		cum += wt(w, i)
		if cum >= target {
			return v[i]
		}
	}
	return v[idx[len(idx)-1]]
}

func sigmoid(x float64) float64 {
	if x >= 0 {
		return 1 / (1 + math.Exp(-x))
	}
	e := math.Exp(x)
	return e / (1 + e)
}

// softplus is log(1 + e^x) without overflow.
func softplus(x float64) float64 {
	if x > 0 {
		return x + math.Log1p(math.Exp(-x))
	}
	return math.Log1p(math.Exp(x))
}
