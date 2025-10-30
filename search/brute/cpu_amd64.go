//go:build amd64

package brute

import "golang.org/x/sys/cpu"

var hasAVX2 = cpu.X86.HasAVX2
