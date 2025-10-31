package runtime

import (
	"runtime"
	"sync/atomic"
	"testing"
	"time"
)

// BenchmarkWorkerPoolDispatch measures the dispatch overhead of the worker pool
func BenchmarkWorkerPoolDispatch(b *testing.B) {
	workerCount := runtime.GOMAXPROCS(0)
	pool := newWorkerPool(workerCount)
	if pool == nil {
		b.Skip("Worker pool requires GOMAXPROCS > 1")
	}
	defer pool.Close()

	// Minimal work per task to measure dispatch overhead
	noop := func() {
		_ = 1 + 1
	}

	b.Run("SingleTask", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			pool.Run(noop)
		}
	})

	b.Run("TwoTasks", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			pool.Run(noop, noop)
		}
	})

	b.Run("FourTasks", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			pool.Run(noop, noop, noop, noop)
		}
	})

	b.Run("EightTasks", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			pool.Run(noop, noop, noop, noop, noop, noop, noop, noop)
		}
	})
}

// BenchmarkRunTasksThreshold validates threshold behavior
func BenchmarkRunTasksThreshold(b *testing.B) {
	workerCount := runtime.GOMAXPROCS(0)
	m := &Model{
		workers: newWorkerPool(workerCount),
	}
	if m.workers == nil {
		b.Skip("Worker pool requires GOMAXPROCS > 1")
	}
	defer m.workers.Close()

	noop := func() {
		_ = 1 + 1
	}

	b.Run("TwoTasks_Threshold2_Parallel", func(b *testing.B) {
		tasks := []func(){noop, noop}
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			m.runTasksThreshold(tasks, 2)
		}
	})

	b.Run("TwoTasks_Threshold4_Serial", func(b *testing.B) {
		tasks := []func(){noop, noop}
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			m.runTasksThreshold(tasks, 4)
		}
	})

	b.Run("FourTasks_Threshold4_Parallel", func(b *testing.B) {
		tasks := []func(){noop, noop, noop, noop}
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			m.runTasksThreshold(tasks, 4)
		}
	})

	b.Run("FourTasks_Threshold8_Serial", func(b *testing.B) {
		tasks := []func(){noop, noop, noop, noop}
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			m.runTasksThreshold(tasks, 8)
		}
	})
}

// BenchmarkWorkerPoolThroughput measures sustained throughput
func BenchmarkWorkerPoolThroughput(b *testing.B) {
	workerCount := runtime.GOMAXPROCS(0)
	pool := newWorkerPool(workerCount)
	if pool == nil {
		b.Skip("Worker pool requires GOMAXPROCS > 1")
	}
	defer pool.Close()

	// Simulate real work (small matmul-like operation)
	work := func() {
		sum := 0
		for i := 0; i < 1000; i++ {
			sum += i
		}
		_ = sum
	}

	b.Run("Batch8x100", func(b *testing.B) {
		tasks := make([]func(), 8)
		for i := range tasks {
			tasks[i] = work
		}
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			for j := 0; j < 100; j++ {
				pool.Run(tasks...)
			}
		}
	})
}

// TestWorkerPoolThreadSafety verifies concurrent submission safety
func TestWorkerPoolThreadSafety(t *testing.T) {
	workerCount := runtime.GOMAXPROCS(0)
	if workerCount <= 1 {
		t.Skip("Test requires GOMAXPROCS > 1")
	}

	pool := newWorkerPool(workerCount)
	defer pool.Close()

	const numSubmitters = 8
	const numSubmissions = 100

	var completed atomic.Int64

	task := func() {
		completed.Add(1)
	}

	// Launch multiple goroutines submitting tasks concurrently
	done := make(chan bool, numSubmitters)
	for i := 0; i < numSubmitters; i++ {
		go func() {
			for j := 0; j < numSubmissions; j++ {
				pool.Run(task, task, task, task)
			}
			done <- true
		}()
	}

	// Wait for all submitters
	timeout := time.After(5 * time.Second)
	for i := 0; i < numSubmitters; i++ {
		select {
		case <-done:
		case <-timeout:
			t.Fatal("Test timed out - possible deadlock")
		}
	}

	expected := int64(numSubmitters * numSubmissions * 4)
	actual := completed.Load()
	if actual != expected {
		t.Errorf("Expected %d completed tasks, got %d", expected, actual)
	}
}

// TestRunTasksThresholdBehavior verifies threshold logic
func TestRunTasksThresholdBehavior(t *testing.T) {
	workerCount := runtime.GOMAXPROCS(0)
	if workerCount <= 1 {
		t.Skip("Test requires GOMAXPROCS > 1")
	}

	m := &Model{
		workers: newWorkerPool(workerCount),
	}
	defer m.workers.Close()

	var executed atomic.Int64
	task := func() {
		executed.Add(1)
	}

	tests := []struct {
		name              string
		taskCount         int
		threshold         int
		shouldRunParallel bool
	}{
		{"Below threshold", 2, 4, false},
		{"At threshold", 4, 4, true},
		{"Above threshold", 8, 4, true},
		{"Zero tasks", 0, 4, false},
		{"Single task", 1, 1, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			executed.Store(0)
			tasks := make([]func(), tt.taskCount)
			for i := range tasks {
				tasks[i] = task
			}

			m.runTasksThreshold(tasks, tt.threshold)

			if got := executed.Load(); got != int64(tt.taskCount) {
				t.Errorf("Expected %d tasks executed, got %d", tt.taskCount, got)
			}
		})
	}
}

// TestWorkerPoolNilTasks verifies nil task handling
func TestWorkerPoolNilTasks(t *testing.T) {
	workerCount := runtime.GOMAXPROCS(0)
	if workerCount <= 1 {
		t.Skip("Test requires GOMAXPROCS > 1")
	}

	pool := newWorkerPool(workerCount)
	defer pool.Close()

	var executed atomic.Int64
	task := func() {
		executed.Add(1)
	}

	// Mix of nil and real tasks
	pool.Run(task, nil, task, nil, task)

	if got := executed.Load(); got != 3 {
		t.Errorf("Expected 3 tasks executed (nil tasks skipped), got %d", got)
	}
}
