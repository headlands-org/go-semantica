//go:build !linux && !darwin && !freebsd

package main

import "time"

// cpuTimeNow returns zero on unsupported platforms.
func cpuTimeNow() time.Duration {
	return 0
}
