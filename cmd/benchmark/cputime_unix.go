//go:build linux || darwin || freebsd

package main

import (
	"time"

	"golang.org/x/sys/unix"
)

// cpuTimeNow returns the combined user+system CPU time consumed by the current
// process. The value represents total compute time across all cores.
func cpuTimeNow() time.Duration {
	var ru unix.Rusage
	if err := unix.Getrusage(unix.RUSAGE_SELF, &ru); err != nil {
		return 0
	}

	user := time.Duration(ru.Utime.Sec)*time.Second + time.Duration(ru.Utime.Usec)*time.Microsecond
	sys := time.Duration(ru.Stime.Sec)*time.Second + time.Duration(ru.Stime.Usec)*time.Microsecond
	return user + sys
}
