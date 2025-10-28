// +build !noasm

#include "textflag.h"

// Precomputed constants for AVX512 VNNI inner loop
DATA ·vnniSignMask+0(SB)/8, $0x8080808080808080
DATA ·vnniSignMask+8(SB)/8, $0x8080808080808080
DATA ·vnniSignMask+16(SB)/8, $0x8080808080808080
DATA ·vnniSignMask+24(SB)/8, $0x8080808080808080
GLOBL ·vnniSignMask(SB), RODATA|NOPTR, $32

DATA ·vnniOnes+0(SB)/8, $0x0101010101010101
DATA ·vnniOnes+8(SB)/8, $0x0101010101010101
DATA ·vnniOnes+16(SB)/8, $0x0101010101010101
DATA ·vnniOnes+24(SB)/8, $0x0101010101010101
GLOBL ·vnniOnes(SB), RODATA|NOPTR, $32

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
// func dotProductINT8VNNI(a, b *int8, n int) int32
// Uses AVX512 to process 16 int8 values per iteration
// For signed×signed int8: sign-extend to int16, then VPMADDWD
TEXT ·dotProductINT8VNNI(SB), NOSPLIT, $0-28
    // Load pointers and length
    MOVQ    a+0(FP), SI          // SI = a pointer
    MOVQ    b+8(FP), DI          // DI = b pointer
    MOVQ    n+16(FP), CX         // CX = n (length)

    // Initialize int32 accumulator to zero
    VPXOR   X0, X0, X0           // XMM0 = 4 × int32 zeros

    // Check if we have at least 16 elements
    CMPQ    CX, $16
    JL      tail

    // Process 16 int8s at a time
    MOVQ    CX, BX
    SHRQ    $4, BX               // BX = len / 16

loop:
    // Load 16 int8 values from a and b
    VMOVDQU8    (SI), X1         // X1 = 16 × int8 from a
    VMOVDQU8    (DI), X2         // X2 = 16 × int8 from b

    // Sign-extend int8 to int16 (16 int8 -> 16 int16 in YMM)
    VPMOVSXBW   X1, Y3           // Y3 = 16 × int16 from a (sign-extended)
    VPMOVSXBW   X2, Y4           // Y4 = 16 × int16 from b (sign-extended)

    // Multiply int16 pairs and add: (a0*b0 + a1*b1), (a2*b2 + a3*b3), ...
    // Result: 8 × int32
    VPMADDWD    Y3, Y4, Y5       // Y5 = 8 × int32 products

    // Accumulate into our int32 accumulator
    // Extract low and high halves and add
    VEXTRACTI128    $1, Y5, X6   // X6 = upper 4 int32s
    VPADDD          X5, X0, X0   // X0 += lower 4 int32s
    VPADDD          X6, X0, X0   // X0 += upper 4 int32s

    ADDQ    $16, SI              // Advance a pointer (16 bytes)
    ADDQ    $16, DI              // Advance b pointer
    DECQ    BX
    JNZ     loop

    // Horizontal sum of X0 (4 int32s) into one int32
    // X0 = [d3, d2, d1, d0]
    VPSHUFD     $0xEE, X0, X1    // X1 = [d3, d2, d3, d2]
    VPADDD      X0, X1, X0       // X0 = [d3+d3, d2+d2, d1+d3, d0+d2]
    VPSHUFD     $0x55, X0, X1    // X1 = [d2+d2, d1+d3, d2+d2, d1+d3]
    VPADDD      X0, X1, X0       // X0 = sum in lowest lane

tail:
    // Handle remaining elements (< 16)
    ANDQ    $15, CX
    JZ      done_int8

    // Scalar accumulator in EAX
    VMOVD   X0, AX               // Extract accumulated sum to EAX

tail_loop:
    // Load one int8 from a and b
    MOVBQSX (SI), R8             // Sign-extend int8 to int64
    MOVBQSX (DI), R9

    // Multiply and accumulate
    IMULQ   R9, R8               // R8 = a * b (int64)
    ADDQ    R8, AX               // AX += R8

    INCQ    SI
    INCQ    DI
    DECQ    CX
    JNZ     tail_loop

    // Store result
    MOVL    AX, ret+24(FP)
    VZEROUPPER
    RET

done_int8:
    // Store result from XMM0
    VMOVD   X0, ret+24(FP)
    VZEROUPPER
    RET

// func matmulInnerLoopAsm(inputRow *int8, weightData *byte, scales *float32, numBlocks int) float32
// Fully-optimized inner loop with vectorized FMA accumulation
// Processes all blocks for one output element, keeping accumulators in YMM registers
TEXT ·matmulInnerLoopAsm(SB), NOSPLIT, $0-36
    MOVQ    inputRow+0(FP), SI      // SI = input pointer
    MOVQ    weightData+8(FP), DI    // DI = weight data pointer
    MOVQ    scales+16(FP), R8       // R8 = scales pointer
    MOVQ    numBlocks+24(FP), CX    // CX = number of blocks

    // Initialize float32 accumulator to zero
    VXORPS  Y0, Y0, Y0              // Y0 = 8 × float32 accumulator

    // Process each block
    XORQ    BX, BX                  // BX = block index

block_loop:
    CMPQ    BX, CX
    JGE     done_matmul

    // Load scale for this block and broadcast to all 8 lanes
    MOVQ    BX, R9
    SHLQ    $2, R9                  // R9 = blockIdx * 4
    VBROADCASTSS (R8)(R9*1), Y1     // Y1 = scale (broadcast to 8 floats)

    // Calculate input pointer: inputRow + blockIdx * 32
    MOVQ    BX, R10
    SHLQ    $5, R10                 // R10 = blockIdx * 32
    LEAQ    (SI)(R10*1), R11        // R11 = input ptr for this block

    // Calculate weight pointer: weightData + blockIdx * 34 + 2
    MOVQ    BX, R12
    IMULQ   $34, R12                // R12 = blockIdx * 34
    LEAQ    2(DI)(R12*1), R13       // R13 = weight ptr (skip f16 scale)

    // Load 32 int8 values from input and weight
    VMOVDQU (R11), Y2               // Y2 = 32 × int8 from input
    VMOVDQU (R13), Y3               // Y3 = 32 × int8 from weight

    // Sign-extend int8 to int16 (process 16 at a time)
    // Lower 16 bytes
    VPMOVSXBW X2, Y4                // Y4 = lower 16 int8 → int16
    VPMOVSXBW X3, Y5                // Y5 = lower 16 int8 → int16

    // Multiply int16 pairs and add adjacent → 8 int32
    VPMADDWD Y4, Y5, Y6             // Y6 = 8 × int32 sums (lower half)

    // Upper 16 bytes
    VEXTRACTI128 $1, Y2, X7         // X7 = upper 16 bytes of input
    VEXTRACTI128 $1, Y3, X8         // X8 = upper 16 bytes of weight
    VPMOVSXBW X7, Y7                // Y7 = upper 16 int8 → int16
    VPMOVSXBW X8, Y8                // Y8 = upper 16 int8 → int16
    VPMADDWD Y7, Y8, Y9             // Y9 = 8 × int32 sums (upper half)

    // Add lower and upper halves
    VPADDD Y6, Y9, Y10              // Y10 = total 8 × int32 sums

    // Convert int32 to float32
    VCVTDQ2PS Y10, Y11              // Y11 = 8 × float32

    // Vectorized FMA: acc += dotProduct * scale
    VFMADD231PS Y11, Y1, Y0         // Y0 = Y0 + (Y11 * Y1)

    // Next block
    INCQ    BX
    JMP     block_loop

done_matmul:
    // Horizontal sum of Y0 (8 float32 values)
    VEXTRACTF128 $1, Y0, X1         // X1 = upper 4 floats
    VADDPS X0, X1, X0               // X0 = lower + upper
    VHADDPS X0, X0, X0              // Horizontal add
    VHADDPS X0, X0, X0              // Horizontal add again

    // Return result
    VMOVSS X0, ret+32(FP)
    VZEROUPPER
    RET

// func matmulInnerLoopVNNIAsm(inputRow *int8, weightData *byte, scales *float32, numBlocks int) float32
// Uses AVX512 VNNI (VPDPBUSD) to fuse byte dot products and minimize conversions.
TEXT ·matmulInnerLoopVNNIAsm(SB), NOSPLIT, $0-36
    MOVQ    inputRow+0(FP), SI      // SI = input pointer
    MOVQ    weightData+8(FP), DI    // DI = weight data pointer
    MOVQ    scales+16(FP), R8       // R8 = scales pointer
    MOVQ    numBlocks+24(FP), CX    // CX = number of blocks

    // Load sign-bit mask (0x80) and ones constants for unsigned conversion / sum
    LEAQ    ·vnniSignMask(SB), R9
    VMOVDQU8 (R9), Y8               // Y8 = 0x80 mask
    LEAQ    ·vnniOnes(SB), R10
    VMOVDQU8 (R10), Y9              // Y9 = vector of uint8(1)

    VXORPS  Y0, Y0, Y0              // Y0 = float32 accumulator

    XORQ    BX, BX                  // BX = block index

vnni_block_loop:
    CMPQ    BX, CX
    JGE     vnni_done

    // Broadcast scale
    MOVQ    BX, R11
    SHLQ    $2, R11
    VBROADCASTSS (R8)(R11*1), Y1    // Y1 = scale

    // Compute input pointer: inputRow + blockIdx * 32
    MOVQ    BX, R12
    SHLQ    $5, R12
    LEAQ    (SI)(R12*1), R13        // R13 = input ptr

    // Compute weight pointer: weightData + blockIdx * 34 + 2
    MOVQ    BX, R14
    IMULQ   $34, R14
    LEAQ    2(DI)(R14*1), R15       // R15 = weight ptr (skip scale)

    VMOVDQU8 (R13), Y2              // Y2 = 32 × int8 input
    VMOVDQU8 (R15), Y3              // Y3 = 32 × int8 weight

    VPXORD  Y8, Y3, Y4              // Y4 = weight unsigned (xor with 0x80)

    VPXORD  Y5, Y5, Y5              // Y5 = 0 (int32 accum) for dot
    VPDPBUSD Y5, Y4, Y2             // Y5 = sum((w+128) * input)

    VPXORD  Y6, Y6, Y6              // Y6 = 0 accumulator for input sums
    VPDPBUSD Y6, Y9, Y2             // Y6 = sum(input) per lane

    VPSLLD  $7, Y6, Y6              // Multiply by 128
    VPSUBD  Y6, Y5, Y5              // Correct for offset: dot -= 128 * sum(input)

    VCVTDQ2PS Y5, Y11               // Convert to float
    VFMADD231PS Y11, Y1, Y0         // Accumulate scaled contribution

    INCQ    BX
    JMP     vnni_block_loop

vnni_done:
    VEXTRACTF128 $1, Y0, X1
    VADDPS X0, X1, X0
    VHADDPS X0, X0, X0
    VHADDPS X0, X0, X0

    VMOVSS X0, ret+32(FP)
    VZEROUPPER
    RET

// func vecMulF32SIMD(dst, a, b *float32, n int)
// Element-wise multiply: dst[i] = a[i] * b[i]
// Processes 8 floats per iteration using AVX2
TEXT ·vecMulF32SIMD(SB), NOSPLIT, $0-32
    MOVQ dst+0(FP), AX      // AX = dst pointer
    MOVQ a+8(FP), SI        // SI = a pointer
    MOVQ b+16(FP), DI       // DI = b pointer
    MOVQ n+24(FP), CX       // CX = n (length)

    // Check if we have at least 8 elements
    CMPQ CX, $8
    JL tail_mul

    // Process 8 floats at a time
    MOVQ CX, BX
    SHRQ $3, BX             // BX = n / 8

loop_mul:
    VMOVUPS (SI), Y0        // Load 8 floats from a
    VMOVUPS (DI), Y1        // Load 8 floats from b
    VMULPS Y0, Y1, Y2       // Multiply: Y2 = Y0 * Y1
    VMOVUPS Y2, (AX)        // Store result

    ADDQ $32, SI            // Advance a pointer (8 * 4 bytes)
    ADDQ $32, DI            // Advance b pointer
    ADDQ $32, AX            // Advance dst pointer
    DECQ BX
    JNZ loop_mul

tail_mul:
    // Handle remaining elements (< 8)
    ANDQ $7, CX
    JZ done_mul

tail_loop_mul:
    VMOVSS (SI), X0
    VMOVSS (DI), X1
    VMULSS X0, X1, X0
    VMOVSS X0, (AX)

    ADDQ $4, SI
    ADDQ $4, DI
    ADDQ $4, AX
    DECQ CX
    JNZ tail_loop_mul

done_mul:
    VZEROUPPER
    RET

// func vecAddF32SIMD(dst, a, b *float32, n int)
// Element-wise add: dst[i] = a[i] + b[i]
// Processes 8 floats per iteration using AVX2
TEXT ·vecAddF32SIMD(SB), NOSPLIT, $0-32
    MOVQ dst+0(FP), AX      // AX = dst pointer
    MOVQ a+8(FP), SI        // SI = a pointer
    MOVQ b+16(FP), DI       // DI = b pointer
    MOVQ n+24(FP), CX       // CX = n (length)

    // Check if we have at least 8 elements
    CMPQ CX, $8
    JL tail_add

    // Process 8 floats at a time
    MOVQ CX, BX
    SHRQ $3, BX             // BX = n / 8

loop_add:
    VMOVUPS (SI), Y0        // Load 8 floats from a
    VMOVUPS (DI), Y1        // Load 8 floats from b
    VADDPS Y0, Y1, Y2       // Add: Y2 = Y0 + Y1
    VMOVUPS Y2, (AX)        // Store result

    ADDQ $32, SI            // Advance a pointer (8 * 4 bytes)
    ADDQ $32, DI            // Advance b pointer
    ADDQ $32, AX            // Advance dst pointer
    DECQ BX
    JNZ loop_add

tail_add:
    // Handle remaining elements (< 8)
    ANDQ $7, CX
    JZ done_add

tail_loop_add:
    VMOVSS (SI), X0
    VMOVSS (DI), X1
    VADDSS X0, X1, X0
    VMOVSS X0, (AX)

    ADDQ $4, SI
    ADDQ $4, DI
    ADDQ $4, AX
    DECQ CX
    JNZ tail_loop_add

done_add:
    VZEROUPPER
    RET

// func vecScaleF32SIMD(dst, a *float32, scale float32, n int)
// Scale vector: dst[i] = a[i] * scale
// Processes 8 floats per iteration using AVX2
TEXT ·vecScaleF32SIMD(SB), NOSPLIT, $0-32
    MOVQ dst+0(FP), AX      // AX = dst pointer
    MOVQ a+8(FP), SI        // SI = a pointer
    VMOVSS scale+16(FP), X0 // X0 = scale (scalar)
    MOVQ n+24(FP), CX       // CX = n (length)

    // Broadcast scale to all 8 lanes of Y1
    VBROADCASTSS X0, Y1

    // Check if we have at least 8 elements
    CMPQ CX, $8
    JL tail_scale

    // Process 8 floats at a time
    MOVQ CX, BX
    SHRQ $3, BX             // BX = n / 8

loop_scale:
    VMOVUPS (SI), Y0        // Load 8 floats from a
    VMULPS Y0, Y1, Y2       // Multiply: Y2 = Y0 * Y1 (scale)
    VMOVUPS Y2, (AX)        // Store result

    ADDQ $32, SI            // Advance a pointer (8 * 4 bytes)
    ADDQ $32, AX            // Advance dst pointer
    DECQ BX
    JNZ loop_scale

tail_scale:
    // Handle remaining elements (< 8)
    ANDQ $7, CX
    JZ done_scale

    // Use scalar version of scale (in X1)
    VMOVSS scale+16(FP), X1

tail_loop_scale:
    VMOVSS (SI), X0
    VMULSS X0, X1, X0
    VMOVSS X0, (AX)

    ADDQ $4, SI
    ADDQ $4, AX
    DECQ CX
    JNZ tail_loop_scale

done_scale:
    VZEROUPPER
    RET
