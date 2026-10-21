package xgoboost

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"strconv"
)

// formatVersion tags serialised models; bump it on incompatible
// changes.
const formatVersion = "xgoboost/1"

type modelJSON struct {
	Format        string     `json:"format"`
	NumFeatures   int        `json:"num_features"`
	BaseScore     float64    `json:"base_score"`
	LearningRate  float64    `json:"learning_rate"`
	Loss          lossJSON   `json:"loss"`
	Monotone      []int      `json:"monotone,omitempty"`
	BestIteration int        `json:"best_iteration"`
	BestScore     float64    `json:"best_score"`
	Trees         []treeJSON `json:"trees"`
}

type lossJSON struct {
	Name   string             `json:"name"`
	Params map[string]float64 `json:"params,omitempty"`
}

type treeJSON struct {
	Nodes []nodeJSON `json:"nodes"`
}

// nodeJSON keeps leaves and splits compact. Threshold is a raw message
// because a split below -Inf-valued data has threshold -Inf, which
// JSON numbers cannot carry; infinities are written as strings.
type nodeJSON struct {
	Leaf        bool            `json:"leaf,omitempty"`
	Value       float64         `json:"value,omitempty"`
	Feature     int             `json:"feature,omitempty"`
	Threshold   json.RawMessage `json:"threshold,omitempty"`
	DefaultLeft bool            `json:"default_left,omitempty"`
	Left        int             `json:"left,omitempty"`
	Right       int             `json:"right,omitempty"`
	Gain        float64         `json:"gain,omitempty"`
	Cover       float64         `json:"cover"`
}

// MarshalJSON encodes the model. Floats round-trip exactly, so a
// reloaded model predicts bit-for-bit the same.
func (m *Model) MarshalJSON() ([]byte, error) {
	if m.Loss == nil {
		return nil, errors.New("xgoboost: model has no loss")
	}
	out := modelJSON{
		Format:        formatVersion,
		NumFeatures:   m.NumFeatures,
		BaseScore:     m.BaseScore,
		LearningRate:  m.LearningRate,
		Loss:          lossJSON{Name: m.Loss.Name()},
		Monotone:      m.Monotone,
		BestIteration: m.BestIteration,
		BestScore:     m.BestScore,
		Trees:         make([]treeJSON, len(m.Trees)),
	}
	if pl, ok := m.Loss.(ParamLoss); ok {
		out.Loss.Params = pl.LossParams()
	}
	if math.IsNaN(out.BestScore) || math.IsInf(out.BestScore, 0) {
		out.BestScore = 0
	}
	for i, t := range m.Trees {
		nodes := make([]nodeJSON, len(t.Nodes))
		for k, nd := range t.Nodes {
			nj := nodeJSON{Leaf: nd.Leaf, Value: nd.Value, Cover: nd.Cover}
			if !nd.Leaf {
				nj.Feature, nj.DefaultLeft = nd.Feature, nd.DefaultLeft
				nj.Left, nj.Right, nj.Gain = nd.Left, nd.Right, nd.Gain
				nj.Threshold = encodeFloat(nd.Threshold)
			}
			nodes[k] = nj
		}
		out.Trees[i] = treeJSON{Nodes: nodes}
	}
	return json.Marshal(out)
}

// UnmarshalJSON decodes and validates a model. Losses other than the
// built-ins must be registered with RegisterLoss first.
func (m *Model) UnmarshalJSON(b []byte) error {
	var in modelJSON
	dec := json.NewDecoder(bytes.NewReader(b))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&in); err != nil {
		return fmt.Errorf("xgoboost: decoding model: %w", err)
	}
	if in.Format != formatVersion {
		return fmt.Errorf("xgoboost: unsupported model format %q", in.Format)
	}
	if in.NumFeatures <= 0 {
		return fmt.Errorf("xgoboost: model has %d features", in.NumFeatures)
	}
	if in.Monotone != nil && len(in.Monotone) != in.NumFeatures {
		return fmt.Errorf("xgoboost: model has %d monotone entries for %d features", len(in.Monotone), in.NumFeatures)
	}
	loss, err := decodeLoss(in.Loss.Name, in.Loss.Params)
	if err != nil {
		return err
	}
	trees := make([]Tree, len(in.Trees))
	for i, tj := range in.Trees {
		if len(tj.Nodes) == 0 {
			return fmt.Errorf("xgoboost: tree %d is empty", i)
		}
		nodes := make([]Node, len(tj.Nodes))
		for k, nj := range tj.Nodes {
			nd := Node{Leaf: nj.Leaf, Value: nj.Value, Cover: nj.Cover}
			if math.IsNaN(nd.Value) || math.IsInf(nd.Value, 0) {
				return fmt.Errorf("xgoboost: tree %d node %d has value %v", i, k, nd.Value)
			}
			if !nj.Leaf {
				// Children must come after their parent: that rules out
				// cycles, so prediction always terminates.
				if nj.Feature < 0 || nj.Feature >= in.NumFeatures ||
					nj.Left <= k || nj.Left >= len(tj.Nodes) ||
					nj.Right <= k || nj.Right >= len(tj.Nodes) {
					return fmt.Errorf("xgoboost: tree %d node %d is malformed", i, k)
				}
				thr, err := decodeFloat(nj.Threshold)
				if err != nil {
					return fmt.Errorf("xgoboost: tree %d node %d: %w", i, k, err)
				}
				nd.Feature, nd.Threshold, nd.DefaultLeft = nj.Feature, thr, nj.DefaultLeft
				nd.Left, nd.Right, nd.Gain = nj.Left, nj.Right, nj.Gain
			}
			nodes[k] = nd
		}
		trees[i] = Tree{Nodes: nodes}
	}
	*m = Model{
		NumFeatures:   in.NumFeatures,
		BaseScore:     in.BaseScore,
		LearningRate:  in.LearningRate,
		Loss:          loss,
		Monotone:      in.Monotone,
		Trees:         trees,
		BestIteration: in.BestIteration,
		BestScore:     in.BestScore,
	}
	return nil
}

// Save writes the model as JSON.
func (m *Model) Save(w io.Writer) error {
	b, err := m.MarshalJSON()
	if err != nil {
		return err
	}
	_, err = w.Write(b)
	return err
}

// Load reads a model written by Save.
func Load(r io.Reader) (*Model, error) {
	b, err := io.ReadAll(r)
	if err != nil {
		return nil, err
	}
	m := new(Model)
	if err := m.UnmarshalJSON(b); err != nil {
		return nil, err
	}
	return m, nil
}

func encodeFloat(v float64) json.RawMessage {
	switch {
	case math.IsInf(v, 1):
		return json.RawMessage(`"+Inf"`)
	case math.IsInf(v, -1):
		return json.RawMessage(`"-Inf"`)
	}
	return json.RawMessage(strconv.FormatFloat(v, 'g', -1, 64))
}

func decodeFloat(raw json.RawMessage) (float64, error) {
	if len(raw) == 0 {
		return 0, nil // omitted: zero
	}
	var s string
	if raw[0] == '"' {
		if err := json.Unmarshal(raw, &s); err != nil {
			return 0, err
		}
		switch s {
		case "+Inf":
			return math.Inf(1), nil
		case "-Inf":
			return math.Inf(-1), nil
		}
		return 0, fmt.Errorf("bad threshold %q", s)
	}
	var v float64
	err := json.Unmarshal(raw, &v)
	return v, err
}
