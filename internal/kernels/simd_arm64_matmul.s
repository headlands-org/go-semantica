//go:build arm64 && !noasm

#include "textflag.h"

// func matmulInnerLoopAsm(inputRow *int8, weightData *byte, scales *float32, numBlocks int) float32
TEXT ·matmulInnerLoopAsm(SB), NOSPLIT, $0-36
	// Load arguments
	MOVD	inputRow+0(FP), R0      // R0 = input pointer
	MOVD	weightData+8(FP), R1    // R1 = weight pointer (raw GGUF layout)
	MOVD	scales+16(FP), R2       // R2 = scales pointer (float32)
	MOVD	numBlocks+24(FP), R3    // R3 = number of 32-element blocks

	// Skip the float16 scale that prefixes the first block
	ADD	$2, R1

	// Accumulate the final float32 sum in F31
	FMOVS	$0.0, F31

	CBZ	R3, done

loop:
	// Reset SIMD accumulator (V0 holds 4x int32 partial sums)
	VEOR	V0.B16, V0.B16, V0.B16

	// First 16 int8 values
	VLD1.P	16(R0), [V1.B16]
	VLD1.P	16(R1), [V2.B16]
	WORD	$0x4e829420          // sdot v0.4s, v1.16b, v2.16b

	// Second 16 int8 values
	VLD1.P	16(R0), [V3.B16]
	VLD1.P	16(R1), [V4.B16]
	WORD	$0x4e849460          // sdot v0.4s, v3.16b, v4.16b

	// Advance weight pointer past the float16 scale for the next block
	ADD	$2, R1

	// Horizontal reduction of V0's four int32 lanes
	WORD	$0x4ea0bc00          // addp v0.4s, v0.4s, v0.4s
	WORD	$0x4ea0bc00          // addp v0.4s, v0.4s, v0.4s (repeat to finish reduction)
	VMOV	V0.S[0], R4          // Move int32 sum into general-purpose register

	// Convert to float32 and apply scale
	WORD	$0x1e220081          // scvtf s1, w4
	FMOVS	(R2), F2
	ADD	$4, R2
	FMULS	F1, F2, F1
	FADDS	F1, F31, F31

	// Loop control
	SUB	$1, R3
	CBNZ	R3, loop

done:
	FMOVS	F31, ret+32(FP)
	RET

