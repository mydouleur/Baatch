// gemv_avx512.cpp — CPU XBF3 GEMV（AVX-512, Zen5）M3'-1。
// 变体 A (lut)：vpermb 聚合 24-bit 组 → vpsrlvd/vpandd 提取 3-bit → vpermps LUT → fmadd
// 变体 B (bp)：位平面 vpandd + vptestmb 掩码 → mask_add 选择性累加（平面权重 1/2/-4 预折进 xp）
//
// 数值对照 tools/quant/xbf3.py 参考（gen_data.py 生成）。
#include <immintrin.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <fstream>

#ifdef _WIN32
#include <windows.h>
#endif

#define SB 256
#define ROW_BYTES 98

static std::vector<char> read_file(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) { fprintf(stderr, "cannot open %s\n", p.c_str()); exit(1); }
  f.seekg(0, std::ios::end);
  size_t n = f.tellg(); f.seekg(0);
  std::vector<char> d(n); f.read(d.data(), n); return d;
}

// ---------------- 变体 A：LUT（vpermb + vpermps） ----------------
// 每 sb: [2B scale][96B packed]；96B = 2 块 × 48B；块 = 16 组 × 3B。
// vpermb 后 lane j = 组 j（低 3B）。提取 k：srli 3k & 7 → vpermps LUT → fmadd(xp[k][sb][16b+j])
static const __m512i VPERMB_IDX = [] {
  alignas(64) uint8_t idx[64];
  for (int j = 0; j < 16; j++) { idx[4*j+0] = 3*j; idx[4*j+1] = 3*j+1; idx[4*j+2] = 3*j+2; idx[4*j+3] = 0; }
  return _mm512_load_si512(idx);
}();

static inline float bf16_to_f32(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }

void gemv_lut_sb(const uint8_t* wsb, float scale, const float* xp_lut_sb,
                 __m512& acc0, __m512& acc1) {
  // xp_lut_sb: [k][2 blocks][16 lanes] = 128 floats per sb
  const __m512 lut = _mm512_setr_ps(0.f, 1.f, 2.f, 3.f, -4.f, -3.f, -2.f, -1.f,
                                    0, 0, 0, 0, 0, 0, 0, 0);
  const __m512i seven = _mm512_set1_epi32(7);
  for (int b = 0; b < 2; b++) {
    __m512i z = _mm512_maskz_loadu_epi8(0x0000FFFFFFFFFFFFULL, wsb + 2 + 48 * b);
    __m512i g = _mm512_permutexvar_epi8(VPERMB_IDX, z);
    __m512& acc = b ? acc1 : acc0;
    #pragma unroll
    for (int k = 0; k < 8; k++) {
      __m512i v = _mm512_and_si512(_mm512_srli_epi32(g, 3 * k), seven);
      __m512 qf = _mm512_permutexvar_ps(v, lut);
      __m512 xv = _mm512_load_ps(xp_lut_sb + (k * 2 + b) * 16);
      acc = _mm512_fmadd_ps(qf, xv, acc);
    }
  }
}

// ---------------- 变体 B：位平面 mask 选择 ----------------
// 每 sb: [2B scale][P0 32B][P1][P2]；plane p byte b bit k = value[8b+k] 的 bit p
// y += Σ_p Σ_k maskadd(xp_bp[p][k][sb][lane], bit 已置)
void gemv_bp_sb(const uint8_t* wsb, float scale, const float* xp_bp_sb,
                __m512& acc_lo, __m512& acc_hi) {
  // xp_bp_sb: [p][k][32 floats] — 96 floats per (p,k)... 布局: [p][k][2 个 zmm]
  const uint8_t* P0 = wsb + 2;
  const uint8_t* P1 = wsb + 34;
  const uint8_t* P2 = wsb + 66;
  __m256i p0 = _mm256_loadu_si256((const __m256i*)P0);
  __m256i p1 = _mm256_loadu_si256((const __m256i*)P1);
  __m256i p2 = _mm256_loadu_si256((const __m256i*)P2);
  #pragma unroll
  for (int k = 0; k < 8; k++) {
    __m256i mk = _mm256_set1_epi8((char)(1 << k));
    __mmask32 m0 = _mm256_test_epi8_mask(_mm256_and_si256(p0, mk), _mm256_and_si256(p0, mk));
    __mmask32 m1 = _mm256_test_epi8_mask(_mm256_and_si256(p1, mk), _mm256_and_si256(p1, mk));
    __mmask32 m2 = _mm256_test_epi8_mask(_mm256_and_si256(p2, mk), _mm256_and_si256(p2, mk));
    const float* x0 = xp_bp_sb + k * 64;                    // p0
    const float* x1 = xp_bp_sb + 8 * 64 + k * 64;           // p1
    const float* x2 = xp_bp_sb + 16 * 64 + k * 64;          // p2
    acc_lo = _mm512_mask_add_ps(acc_lo, (__mmask16)(m0 & 0xFFFF), acc_lo, _mm512_loadu_ps(x0));
    acc_hi = _mm512_mask_add_ps(acc_hi, (__mmask16)(m0 >> 16),  acc_hi, _mm512_loadu_ps(x0 + 16));
    acc_lo = _mm512_mask_add_ps(acc_lo, (__mmask16)(m1 & 0xFFFF), acc_lo, _mm512_loadu_ps(x1));
    acc_hi = _mm512_mask_add_ps(acc_hi, (__mmask16)(m1 >> 16),  acc_hi, _mm512_loadu_ps(x1 + 16));
    acc_lo = _mm512_mask_add_ps(acc_lo, (__mmask16)(m2 & 0xFFFF), acc_lo, _mm512_loadu_ps(x2));
    acc_hi = _mm512_mask_add_ps(acc_hi, (__mmask16)(m2 >> 16),  acc_hi, _mm512_loadu_ps(x2 + 16));
  }
}

// ---------------- GEMV 主体（行循环） ----------------
template <int VARIANT>
void gemv_rows(const uint8_t* w, int row0, int nrows, int n_in,
               const float* xp, float* y) {
  const int sb_per_row = n_in / SB;
  for (int r = 0; r < nrows; r++) {
    const uint8_t* rowp = w + (size_t)(row0 + r) * sb_per_row * ROW_BYTES;
    __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
    for (int sb = 0; sb < sb_per_row; sb++) {
      const uint8_t* wsb = rowp + sb * ROW_BYTES;
      float scale = bf16_to_f32(wsb[0] | (wsb[1] << 8));
      __m512 sb0 = _mm512_setzero_ps(), sb1 = _mm512_setzero_ps();
      if (VARIANT == 0) gemv_lut_sb(wsb, scale, xp + sb * 256, sb0, sb1);
      else              gemv_bp_sb(wsb, scale, xp + sb * 1536 / 2, sb0, sb1);
      __m512 sc = _mm512_set1_ps(scale);
      acc0 = _mm512_fmadd_ps(sb0, sc, acc0);
      acc1 = _mm512_fmadd_ps(sb1, sc, acc1);
    }
    y[row0 + r] = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
  }
}

// ---------------- harness ----------------
struct Args {
  std::string variant = "lut";
  std::string dir = "data_cpu";
  int threads = 1;
  int iters = 20;
  int experts = 64;
  int n_in = 512;
  int n_out = 2048;
  bool verify_only = false;
};

int main(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; i++) {
    std::string s = argv[i];
    if (s == "--variant" && i + 1 < argc) a.variant = argv[++i];
    else if (s == "--threads") a.threads = atoi(argv[++i]);
    else if (s == "--iters") a.iters = atoi(argv[++i]);
    else if (s == "--dir") a.dir = argv[++i];
    else if (s == "--experts") a.experts = atoi(argv[++i]);
    else if (s == "--verify-only") a.verify_only = true;
  }
  int E = a.experts, NIN = a.n_in, NOUT = a.n_out;
  auto wts = read_file(a.dir + (a.variant == "lut" ? "\\wts_lut.bin" : "\\wts_bp.bin"));
  auto xp  = read_file(a.dir + (a.variant == "lut" ? "\\xp_lut.bin" : "\\xp_bp.bin"));
  auto yr  = read_file(a.dir + "\\y_ref.bin");
  const uint8_t* W = (const uint8_t*)wts.data();
  const float* XP = (const float*)xp.data();
  const float* YR = (const float*)yr.data();
  std::vector<float> Y(E * NOUT, 0.0f);
  size_t w_per_exp = (size_t)NOUT * (NIN / SB) * ROW_BYTES;
  size_t total_w = w_per_exp * E;

  auto run_slice = [&](int e0, int e1, float* yout) {
    for (int e = e0; e < e1; e++)
      if (a.variant == "lut")
        gemv_rows<0>(W + e * w_per_exp, 0, NOUT, NIN, XP, yout + e * NOUT);
      else
        gemv_rows<1>(W + e * w_per_exp, 0, NOUT, NIN, XP, yout + e * NOUT);
  };

  // verify（单线程全量）
  {
    run_slice(0, E, Y.data());
    int bad = 0; float max_rel = 0;
    for (size_t i = 0; i < (size_t)E * NOUT; i++) {
      float d = fabsf(Y[i] - YR[i]);
      float rel = d / (fabsf(YR[i]) + 1e-6f);
      if (rel > max_rel) max_rel = rel;
      if (rel > 1e-3f && d > 1e-3f) bad++;
    }
    printf("verify[%s]: %s (bad=%d max_rel=%.2e)\n",
           a.variant.c_str(), bad ? "FAIL" : "PASS", bad, max_rel);
    if (bad) return 1;
  }
  if (a.verify_only) return 0;

  // 带宽测量：每线程连续专家段
  std::vector<std::thread> ths;
  std::vector<double> tbest(a.threads, 1e9);
  std::atomic<int> ready{0};
  std::atomic<bool> go{false};
  for (int t = 0; t < a.threads; t++) {
    ths.emplace_back([&, t] {
#ifdef _WIN32
      // 绑核：偶数逻辑核（SMT 第一线程）——Zen5 大核优先（见 README 注记）
      SetThreadAffinityMask(GetCurrentThread(), 1ull << (2 * t));
#endif
      int per = E / a.threads;
      int e0 = t * per, e1 = (t == a.threads - 1) ? E : e0 + per;
      ready.fetch_add(1);
      while (!go.load()) _mm_pause();
      for (int it = 0; it < a.iters; it++) {
        auto t0 = std::chrono::steady_clock::now();
        run_slice(e0, e1, Y.data());
        auto t1 = std::chrono::steady_clock::now();
        double sec = std::chrono::duration<double>(t1 - t0).count();
        double gbps = (total_w / a.threads) / sec / 1e9;
        if (sec < tbest[t]) tbest[t] = sec;
        if (it == 0 && t == 0)
          printf("  thread0 iter0: %.3f ms (%.2f GB/s per thread)\n", sec * 1e3, gbps);
      }
    });
  }
  auto tall0 = std::chrono::steady_clock::now();
  while (ready.load() < a.threads) _mm_pause();
  go.store(true);
  for (auto& th : ths) th.join();
  auto tall1 = std::chrono::steady_clock::now();
  double wall = std::chrono::duration<double>(tall1 - tall0).count();
  double agg = total_w * (double)a.iters / wall / 1e9;
  double per_thread_best = (total_w / a.threads) / *std::min_element(tbest.begin(), tbest.end()) / 1e9;
  printf("RESULT[%s] threads=%d iters=%d  total=%.1fMB  wall=%.1fms  agg=%.2f GB/s  per-thread-best=%.2f GB/s\n",
         a.variant.c_str(), a.threads, a.iters, total_w / 1048576.0, wall * 1e3, agg, per_thread_best);
  return 0;
}
