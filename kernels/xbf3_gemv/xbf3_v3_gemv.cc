// ---- v3 (multi-column, act streamed as chunk 0) ----
// 静态激活缓冲（定义在此 TU，load_act 里 extern 引用）
float xbuf_v3[512];

extern "C"
void xbf3_gemv_chunk3(const uint8_t* __restrict w, float* __restrict y) {
  for (int r = 0; r < CHUNK_ROWS; r++) {
    const uint8_t* rowp = w + r * (XBF3_ROW_SBS * XBF3_SB_BYTES);
    float acc = 0.0f;
    for (int sb = 0; sb < XBF3_ROW_SBS; sb++, rowp += XBF3_SB_BYTES) {
      uint32_t sbits = (uint32_t)rowp[0] | ((uint32_t)rowp[1] << 8);
      uint32_t fbits = sbits << 16;
      float scale;
      __builtin_memcpy(&scale, &fbits, sizeof(float));
      const uint8_t* pk = rowp + 2;
      const float* xs = xbuf_v3 + sb * XBF3_SB;
      float sacc = 0.0f;
      for (int j = 0; j < 96; j += 3) {
        uint32_t v = (uint32_t)pk[j] | ((uint32_t)pk[j + 1] << 8) |
                     ((uint32_t)pk[j + 2] << 16);
#pragma unroll
        for (int k = 0; k < 8; k++) {
          int q = (int)((v >> (3 * k)) & 7u);
          if (q >= 4) q -= 8;
          sacc += (float)q * xs[(j / 3) * 8 + k];
        }
      }
      acc += scale * sacc;
    }
    y[r] = acc;
  }
}

extern "C"
void xbf3_discard_chunk3(const uint8_t* __restrict w, float* __restrict y) {
  y[0] = (float)w[0];
}
