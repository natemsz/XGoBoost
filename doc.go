// Package xgoboost trains gradient-boosted decision trees with
// per-feature monotone constraints, in pure Go with no dependencies
// outside the standard library.
//
// Training is second order: every round computes each row's gradient
// and hessian under a pluggable [Loss], grows one regression tree whose
// leaves take the Newton value -G/(H+λ), and adds it, scaled by the
// learning rate, to the ensemble. Splits are scored by the usual
// second-order gain and found either by exact greedy enumeration or on
// quantile-binned feature histograms with histogram subtraction.
//
// Monotone constraints use the XGBoost envelope construction: a split
// on a constrained feature is kept only when its children's values are
// ordered the right way, and each child's subtree is then bounded at
// the midpoint between them. Every tree, and therefore the ensemble, is
// monotone in each constrained feature; see the README for the proof.
//
// Missing values are NaN. Each split learns a default direction for
// them from the training data.
//
// A minimal regression:
//
//	p := xgoboost.DefaultParams()
//	p.Monotone = []int{+1, 0, -1}
//	m, err := xgoboost.Train(X, y, nil, p)
//	if err != nil { ... }
//	yhat := m.Predict(x)
package xgoboost
