// +build !noasm

#include "textflag.h"

// func dotProductAVX2Asm(a, b *float32, n int) float32
TEXT ·dotProductAVX2Asm(SB), NOSPLIT, $0-28
    // Load pointers and length
    MOVQ    a+0(FP), SI          // SI = a pointer
    MOVQ    b+8(FP), DI          // DI = b pointer
    MOVQ    n+16(FP), CX         // CX = n (length)

    // Initialize accumulator to zero (8 floats in YMM0)
    VXORPS  Y0, Y0, Y0

    // Check if we have at least 8 elements
    CMPQ    CX, $8
    JL      tail

    // Process 8 floats at a time using AVX2
    MOVQ    CX, BX
    SHRQ    $3, BX               // BX = len / 8 (number of full vectors)

loop:
    VMOVUPS (SI), Y1             // Load 8 floats from a
    VMOVUPS (DI), Y2             // Load 8 floats from b

    // Multiply and accumulate: Y0 += Y1 * Y2
    VFMADD231PS Y1, Y2, Y0       // Y0 = Y0 + (Y1 * Y2)

    ADDQ    $32, SI              // Advance a pointer (8 * 4 bytes)
    ADDQ    $32, DI              // Advance b pointer
    DECQ    BX
    JNZ     loop

    // Horizontal sum of Y0 into XMM0
    // Y0 = [a7, a6, a5, a4, a3, a2, a1, a0]
    VEXTRACTF128 $1, Y0, X1      // X1 = upper half [a7, a6, a5, a4]
    VADDPS  X0, X1, X0           // X0 = [a7+a3, a6+a2, a5+a1, a4+a0]

    VHADDPS X0, X0, X0           // X0 = [*, *, a7+a6+a5+a4+a3+a2+a1+a0, *]
    VHADDPS X0, X0, X0           // X0 = sum in all lanes

    // Handle tail elements (< 8 remaining)
tail:
    ANDQ    $7, CX               // CX = len % 8
    JZ      done

tail_loop:
    VMOVSS  (SI), X1
    VMOVSS  (DI), X2
    VMULSS  X1, X2, X1
    VADDSS  X0, X1, X0

    ADDQ    $4, SI
    ADDQ    $4, DI
    DECQ    CX
    JNZ     tail_loop

done:
    // Store result
    VMOVSS  X0, ret+24(FP)       // Return offset adjusted for new signature
    VZEROUPPER                   // Clear upper YMM state for compatibility
    RET
