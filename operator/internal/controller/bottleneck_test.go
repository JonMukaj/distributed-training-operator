package controller

import "testing"

func TestClassifyBottleneck(t *testing.T) {
	cases := []struct {
		name    string
		metrics map[string]string
		want    string
		wantOK  bool
	}{
		{
			name:    "not instrumented → no classification",
			metrics: map[string]string{"samplesPerSecond": "10"},
			want:    "", wantOK: false,
		},
		{
			name:    "compute dominates → compute-bound",
			metrics: map[string]string{"allreduceFraction": "0.15", "dataloadFraction": "0.10", "computeFraction": "0.75"},
			want:    "compute-bound", wantOK: true,
		},
		{
			name:    "allreduce over dataload below compute threshold → communication-bound",
			metrics: map[string]string{"allreduceFraction": "0.40", "dataloadFraction": "0.20", "computeFraction": "0.40"},
			want:    "communication-bound", wantOK: true,
		},
		{
			name:    "dataload dominates → data-bound",
			metrics: map[string]string{"allreduceFraction": "0.20", "dataloadFraction": "0.45", "computeFraction": "0.35"},
			want:    "data-bound", wantOK: true,
		},
		{
			name:    "compute exactly at threshold → compute-bound",
			metrics: map[string]string{"allreduceFraction": "0.30", "dataloadFraction": "0.10", "computeFraction": "0.60"},
			want:    "compute-bound", wantOK: true,
		},
		{
			name:    "unparseable fraction → no classification",
			metrics: map[string]string{"allreduceFraction": "x", "dataloadFraction": "0.1", "computeFraction": "0.5"},
			want:    "", wantOK: false,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := classifyBottleneck(tc.metrics)
			if got != tc.want || ok != tc.wantOK {
				t.Errorf("classifyBottleneck() = (%q, %v), want (%q, %v)", got, ok, tc.want, tc.wantOK)
			}
		})
	}
}
