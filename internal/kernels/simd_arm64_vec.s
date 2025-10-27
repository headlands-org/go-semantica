//go:build arm64 && !noasm

#include "textflag.h"

// func vecMulF32SIMD(dst, a, b *float32, n int)
TEXT ·vecMulF32SIMD(SB), NOSPLIT, $0-32
	MOVD	dst+0(FP), R0
	MOVD	a+8(FP), R1
	MOVD	b+16(FP), R2
	MOVD	n+24(FP), R3

	CBZ	R3, vecmul_done

	AND	$3, R3, R5          // remainder = n & 3
	LSR	$2, R3, R4         // vector blocks = n / 4
	CBZ	R4, vecmul_scalar

vecmul_loop:
	VLD1.P	16(R1), [V0.S4]
	VLD1.P	16(R2), [V1.S4]
	WORD	$0x6e21dc00        // fmul v0.4s, v0.4s, v1.4s
	VST1.P	[V0.S4], 16(R0)
	SUB	$1, R4
	CBNZ	R4, vecmul_loop

vecmul_scalar:
	CBZ	R5, vecmul_done

vecmul_remainder:
	FMOVS	(R1), F0
	FMOVS	(R2), F1
	FMULS	F0, F1, F0
	FMOVS	F0, (R0)
	ADD	$4, R1
	ADD	$4, R2
	ADD	$4, R0
	SUB	$1, R5
	CBNZ	R5, vecmul_remainder

vecmul_done:
	RET

// func vecAddF32SIMD(dst, a, b *float32, n int)
TEXT ·vecAddF32SIMD(SB), NOSPLIT, $0-32
	MOVD	dst+0(FP), R0
	MOVD	a+8(FP), R1
	MOVD	b+16(FP), R2
	MOVD	n+24(FP), R3

	CBZ	R3, vecadd_done

	AND	$3, R3, R5          // remainder = n & 3
	LSR	$2, R3, R4         // vector blocks = n / 4
	CBZ	R4, vecadd_scalar

vecadd_loop:
	VLD1.P	16(R1), [V0.S4]
	VLD1.P	16(R2), [V1.S4]
	WORD	$0x4e21d400        // fadd v0.4s, v0.4s, v1.4s
	VST1.P	[V0.S4], 16(R0)
	SUB	$1, R4
	CBNZ	R4, vecadd_loop

vecadd_scalar:
	CBZ	R5, vecadd_done

vecadd_remainder:
	FMOVS	(R1), F0
	FMOVS	(R2), F1
	FADDS	F0, F1, F0
	FMOVS	F0, (R0)
	ADD	$4, R1
	ADD	$4, R2
	ADD	$4, R0
	SUB	$1, R5
	CBNZ	R5, vecadd_remainder

vecadd_done:
	RET

// func vecScaleF32SIMD(dst, a *float32, scale float32, n int)
TEXT ·vecScaleF32SIMD(SB), NOSPLIT, $0-32
	MOVD	dst+0(FP), R0
	MOVD	a+8(FP), R1
	FMOVS	scale+16(FP), F1
	MOVD	n+24(FP), R2

	CBZ	R2, vecscale_done

	AND	$3, R2, R4          // remainder = n & 3
	LSR	$2, R2, R3         // vector blocks = n / 4
	VDUP	V1.S[0], V1.S4    // broadcast scale into V1
	CBZ	R3, vecscale_scalar

vecscale_loop:
	VLD1.P	16(R1), [V0.S4]
	WORD	$0x6e21dc00        // fmul v0.4s, v0.4s, v1.4s
	VST1.P	[V0.S4], 16(R0)
	SUB	$1, R3
	CBNZ	R3, vecscale_loop

vecscale_scalar:
	CBZ	R4, vecscale_done

vecscale_remainder:
	FMOVS	(R1), F0
	FMULS	F0, F1, F0
	FMOVS	F0, (R0)
	ADD	$4, R1
	ADD	$4, R0
	SUB	$1, R4
	CBNZ	R4, vecscale_remainder

vecscale_done:
	RET
