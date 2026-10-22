// xbf3_gemv_chunk — XBF3 packed-weight GEMV chunk kernel for AIE2P (XDNA2) compute tile.
//
// XBF3 layout (docs/DESIGN.md §3): row-major (out,in) matrix; per 256-weight
// superblock: [2B little-endian BF16 scale][96B packed weights]; 8 weights per
// 3-byte little-endian group, value i at bits [3i,3i+3), 3-bit two's complement.
//
// v1 (correctness-first): scalar unpack + fp32 MAC. No intrinsics needed.
//
// One call processes one ObjectFifo chunk = CHUNK_ROWS consecutive output rows.
// IN is fixed at 512 (one expert down_proj reduction dim): 2 superblocks/row.

#include <stdint.h>

#define XBF3_IN 512
#define XBF3_SB 256
#define XBF3_ROW_SBS (XBF3_IN / XBF3_SB)   // 2
#define XBF3_SB_BYTES (2 + XBF3_SB * 3 / 8) // 98
#ifdef CHUNK_ROWS_C
#define CHUNK_ROWS CHUNK_ROWS_C
#else
#define CHUNK_ROWS 16
#endif

extern "C"
void xbf3_gemv_chunk(const uint8_t* __restrict w, const float* __restrict x,
                     float* __restrict y) {
  for (int r = 0; r < CHUNK_ROWS; r++) {
    const uint8_t* rowp = w + r * (XBF3_ROW_SBS * XBF3_SB_BYTES);
    float acc = 0.0f;
    for (int sb = 0; sb < XBF3_ROW_SBS; sb++, rowp += XBF3_SB_BYTES) {
      // BF16 scale, little-endian 2B -> fp32 by placing into high half-word
      uint32_t sbits = (uint32_t)rowp[0] | ((uint32_t)rowp[1] << 8);
      uint32_t fbits = sbits << 16;
      float scale;
      __builtin_memcpy(&scale, &fbits, sizeof(float));

      const uint8_t* pk = rowp + 2;
      const float* xs = x + sb * XBF3_SB;
      float sacc = 0.0f;
      for (int j = 0; j < 96; j += 3) {
        uint32_t v = (uint32_t)pk[j] | ((uint32_t)pk[j + 1] << 8) |
                     ((uint32_t)pk[j + 2] << 16);
#pragma unroll
        for (int k = 0; k < 8; k++) {
          int q = (int)((v >> (3 * k)) & 7u);
          if (q >= 4) q -= 8;  // 3-bit two's complement
          sacc += (float)q * xs[(j / 3) * 8 + k];
        }
      }
      acc += scale * sacc;
    }
    y[r] = acc;
  }
}

// Pipe-only variant: consume wts without computing (isolates DMA ceiling).
extern "C"
void xbf3_discard_chunk(const uint8_t* __restrict w, const float* __restrict x,
                        float* __restrict y) {
  y[0] = (float)w[0];  // 触摸首字节，防止被优化掉
}

