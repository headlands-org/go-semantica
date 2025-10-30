//go:build amd64

#include "textflag.h"

DATA ·inv127+0(SB)/4, $0.0078740157
GLOBL ·inv127(SB), RODATA, $4

DATA ·inv32767+0(SB)/4, $0.000030518509
GLOBL ·inv32767(SB), RODATA, $4

// func dotInt8AVX2(vec *int8, query *float32, length int) float32
TEXT ·dotInt8AVX2(SB), NOSPLIT, $0-28
    MOVQ vec+0(FP), AX
    MOVQ query+8(FP), BX
    MOVQ length+16(FP), CX
    VXORPS Y4, Y4, Y4
    VMOVSS ·inv127(SB), X0
    VBROADCASTSS X0, Y5
    CMPQ CX, $8
    JL dotInt8Reduce

loopInt8:
    VMOVQ (AX), X1
    VPMOVSXBD X1, Y0
    VCVTDQ2PS Y0, Y0
    VMOVUPS (BX), Y1
    VMULPS Y5, Y0, Y0
    VMULPS Y1, Y0, Y0
    VADDPS Y0, Y4, Y4
    ADDQ $8, AX
    ADDQ $32, BX
    SUBQ $8, CX
    CMPQ CX, $8
    JGE loopInt8

 dotInt8Reduce:
    VEXTRACTF128 $1, Y4, X1
    VADDPS X1, X4, X4
    VHADDPS X4, X4, X4
    VHADDPS X4, X4, X4
    MOVSS X4, ret+24(FP)
    VZEROUPPER
    RET

// func dotInt16AVX2(vec *int16, query *float32, length int) float32
TEXT ·dotInt16AVX2(SB), NOSPLIT, $0-28
    MOVQ vec+0(FP), AX
    MOVQ query+8(FP), BX
    MOVQ length+16(FP), CX
    VXORPS Y4, Y4, Y4
    VMOVSS ·inv32767(SB), X0
    VBROADCASTSS X0, Y5
    CMPQ CX, $8
    JL dotInt16Reduce

loopInt16:
    VMOVDQU (AX), X1
    VPMOVSXWD X1, Y0
    VCVTDQ2PS Y0, Y0
    VMOVUPS (BX), Y1
    VMULPS Y5, Y0, Y0
    VMULPS Y1, Y0, Y0
    VADDPS Y0, Y4, Y4
    ADDQ $16, AX
    ADDQ $32, BX
    SUBQ $8, CX
    CMPQ CX, $8
    JGE loopInt16

 dotInt16Reduce:
    VEXTRACTF128 $1, Y4, X1
    VADDPS X1, X4, X4
    VHADDPS X4, X4, X4
    VHADDPS X4, X4, X4
    MOVSS X4, ret+24(FP)
    VZEROUPPER
    RET

