package tokenizer

import "container/heap"

// bpeBigram represents a candidate bigram merge in BPE tokenization.
// It stores the merge score and indices of the symbols to merge.
type bpeBigram struct {
	score float32 // Merge score (higher is better)
	left  int     // Index of left symbol
	right int     // Index of right symbol
	text  string  // Merged text (for validation)
}

// bpePriorityQueue implements a max-heap for BPE bigrams.
// Bigrams with higher scores have higher priority.
// On tie, prefer rightmost position (higher left index).
type bpePriorityQueue []*bpeBigram

func (pq bpePriorityQueue) Len() int { return len(pq) }

func (pq bpePriorityQueue) Less(i, j int) bool {
	// Max-heap: higher score = higher priority
	if pq[i].score != pq[j].score {
		return pq[i].score > pq[j].score
	}
	// Tie-breaker: prefer rightmost position (higher left index)
	return pq[i].left > pq[j].left
}

func (pq bpePriorityQueue) Swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
}

func (pq *bpePriorityQueue) Push(x interface{}) {
	*pq = append(*pq, x.(*bpeBigram))
}

func (pq *bpePriorityQueue) Pop() interface{} {
	old := *pq
	n := len(old)
	item := old[n-1]
	old[n-1] = nil // avoid memory leak
	*pq = old[0 : n-1]
	return item
}

// newBPEPriorityQueue creates a new priority queue with initial capacity.
func newBPEPriorityQueue(capacity int) *bpePriorityQueue {
	pq := make(bpePriorityQueue, 0, capacity)
	heap.Init(&pq)
	return &pq
}

// push adds a bigram to the queue.
func (pq *bpePriorityQueue) push(b *bpeBigram) {
	heap.Push(pq, b)
}

// pop removes and returns the highest-priority bigram.
func (pq *bpePriorityQueue) pop() *bpeBigram {
	if pq.Len() == 0 {
		return nil
	}
	return heap.Pop(pq).(*bpeBigram)
}
