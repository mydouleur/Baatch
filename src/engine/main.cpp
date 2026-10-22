// main.cpp — Baatch 引擎 stage 2/3 驱动
//   --test-layer N --vec <dir>   单层对拍（x_in.bin/y_gdn.bin/y_full.bin）
//   --decode --ids "1,2,3" --n 8 全模型贪心 decode（阶段 3）
#include "baa.h"
#include "model.h"
#include <chrono>
#include <fstream>

static std::vector<float> read_f32(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  f.seekg(0, std::ios::end); size_t n = f.tellg(); f.seekg(0);
  std::vector<float> d(n / 4); f.read((char*)d.data(), n);
  return d;
}

static void compare(const char* tag, const std::vector<float>& y, const std::vector<float>& ref) {
  double num = 0, den = 0, maxrel = 0;
  for (size_t i = 0; i < y.size(); i++) {
    double d = y[i] - ref[i];
    num += d * d; den += (double)ref[i] * ref[i];
    double rel = fabs(d) / (fabs(ref[i]) + 1e-4);
    maxrel = std::max(maxrel, rel);
  }
  printf("[%s] rel_l2=%.4f  max_rel=%.4f  %s\n", tag, sqrt(num / (den + 1e-12)), maxrel,
         sqrt(num / (den + 1e-12)) < 0.02 ? "PASS(<2%)" : "FAIL");
}

int main(int argc, char** argv) {
  std::string baa = "models/ornith35.baa";
  std::string vec;
  int test_layer = -1;
  for (int i = 1; i < argc; i++) {
    std::string s = argv[i];
    if (s == "--baa") baa = argv[++i];
    else if (s == "--test-layer") test_layer = atoi(argv[++i]);
    else if (s == "--vec") vec = argv[++i];
  }
  Model m;
  m.open(baa);
  fprintf(stderr, "baa loaded: %zu tensors\n", m.baa.names.size());

  if (test_layer >= 0) {
    int L = test_layer;
    auto x = read_f32(vec + "\\x_in.bin");
    auto yg_ref = read_f32(vec + "\\y_gdn.bin");
    auto yf_ref = read_f32(vec + "\\y_full.bin");
    GdnState gst;
    std::vector<float> y1(Cfg::H), y2(Cfg::H);
    auto t0 = std::chrono::steady_clock::now();
    gdn_decode(m, L, gst, x.data(), y1.data());
    moe(m, L, y1.data(), y2.data());
    auto t1 = std::chrono::steady_clock::now();
    compare("gdn", y1, yg_ref);
    compare("full", y2, yf_ref);
    printf("layer time: %.1f ms\n", std::chrono::duration<double>(t1 - t0).count() * 1e3);
    return 0;
  }
  fprintf(stderr, "nothing to do\n");
  return 1;
}
