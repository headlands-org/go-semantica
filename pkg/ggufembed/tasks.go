package ggufembed

import (
	"errors"
	"fmt"
	"strings"
)

// Task defines the embedding use case. The underlying prompt strings follow the
// recommendations from the EmbeddingGemma model card:
// https://ai.google.dev/gemma/docs/embeddinggemma/model_card
type Task int

const (
	// TaskNone leaves the content untouched; no task prefix is added.
	TaskNone Task = -1
)

const (
	// TaskSearchQuery is optimised for retrieval queries (“task: search result | query:”).
	TaskSearchQuery Task = iota
	// TaskSearchDocument is optimised for indexing documents (“title: … | text:”).
	TaskSearchDocument
	// TaskQuestionAnswering is optimised for question queries.
	TaskQuestionAnswering
	// TaskFactVerification is optimised for statements that need evidence.
	TaskFactVerification
	// TaskClassification is optimised for downstream classification.
	TaskClassification
	// TaskClustering is optimised for grouping similar texts.
	TaskClustering
	// TaskSemanticSimilarity is optimised for sentence-level similarity comparisons.
	TaskSemanticSimilarity
	// TaskCodeRetrieval is optimised for natural-language-to-code retrieval.
	TaskCodeRetrieval
)

// String returns a human-readable name.
func (t Task) String() string {
	switch t {
	case TaskNone:
		return "none"
	case TaskSearchQuery:
		return "search_query"
	case TaskSearchDocument:
		return "search_document"
	case TaskQuestionAnswering:
		return "question_answering"
	case TaskFactVerification:
		return "fact_verification"
	case TaskClassification:
		return "classification"
	case TaskClustering:
		return "clustering"
	case TaskSemanticSimilarity:
		return "semantic_similarity"
	case TaskCodeRetrieval:
		return "code_retrieval"
	default:
		return fmt.Sprintf("task(%d)", int(t))
	}
}

// EmbedInput describes a single embedding request.
type EmbedInput struct {
	Task    Task
	Content string

	// Title is optional metadata for TaskSearchDocument. If left empty the
	// generated prompt uses "none".
	Title string

	// CustomTaskDescription overrides the default description in the prompt for
	// query-style tasks. Leave empty to use the recommended strings from the
	// model card.
	CustomTaskDescription string

	// Dim specifies the requested embedding dimension. Supported values align
	// with EmbeddingGemma's Matryoshka Representation Learning tiers:
	//   768, 512, 256, 128. A value of 0 selects DefaultEmbedDim (512).
	Dim int
}

const (
	// DefaultEmbedDim is the embedding dimension used when callers do not
	// specify a value. This matches the recommended truncation tier for most
	// retrieval workloads.
	DefaultEmbedDim = 512
)

var supportedDims = []int{768, 512, 256, 128}

var supportedDimSet = map[int]struct{}{
	768: {},
	512: {},
	256: {},
	128: {},
}

// ResolveDim validates dim and applies the default when zero.
func ResolveDim(dim int) (int, error) {
	if dim == 0 {
		return DefaultEmbedDim, nil
	}
	if _, ok := supportedDimSet[dim]; !ok {
		return 0, fmt.Errorf("unsupported embed dim %d (allowed: 768, 512, 256, 128)", dim)
	}
	return dim, nil
}

// SupportedDims returns the list of valid embedding dimensions in descending order.
func SupportedDims() []int {
	cp := make([]int, len(supportedDims))
	copy(cp, supportedDims)
	return cp
}

// prompt builds the prompt string expected by EmbeddingGemma.
func (in EmbedInput) prompt() (string, error) {
	content := strings.TrimSpace(in.Content)
	if content == "" {
		return "", errors.New("content must not be empty")
	}

	if in.Task == TaskNone {
		return content, nil
	}

	if in.Task == TaskSearchDocument {
		title := strings.TrimSpace(in.Title)
		if title == "" {
			title = "none"
		}
		return fmt.Sprintf("title: %s | text: %s", title, content), nil
	}

	desc := strings.TrimSpace(in.CustomTaskDescription)
	if desc == "" {
		var ok bool
		desc, ok = defaultTaskDescriptions[in.Task]
		if !ok {
			return "", fmt.Errorf("unsupported task %v (missing default description)", in.Task)
		}
	}

	return fmt.Sprintf("task: %s | query: %s", desc, content), nil
}

var defaultTaskDescriptions = map[Task]string{
	TaskSearchQuery:        "search result",
	TaskQuestionAnswering:  "question answering",
	TaskFactVerification:   "fact checking",
	TaskClassification:     "classification",
	TaskClustering:         "clustering",
	TaskSemanticSimilarity: "sentence similarity",
	TaskCodeRetrieval:      "code retrieval",
}

// BuildPrompt returns the task-specific prompt string used internally. It is
// exported to help callers log or inspect the string before sending it to the
// model.
func BuildPrompt(input EmbedInput) (string, error) {
	return input.prompt()
}
