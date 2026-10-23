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

int main(int argc, char** argv) { try {
  std::string baa = "models/ornith35.baa";
  std::string vec;
  int test_layer = -1;
  int n_gen = 0;
  std::string ids_str;
  std::string probe_dir;
  for (int i = 1; i < argc; i++) {
    std::string s = argv[i];
    if (s == "--baa") baa = argv[++i];
    else if (s == "--test-layer") test_layer = atoi(argv[++i]);
    else if (s == "--vec") vec = argv[++i];
    else if (s == "--test-attn") test_layer = 1000 + atoi(argv[++i]);
    else if (s == "--ids") ids_str = argv[++i];
    else if (s == "--n") n_gen = atoi(argv[++i]);
    else if (s == "--probe") probe_dir = argv[++i];
  }
  Model m;
  m.open(baa);
  fprintf(stderr, "baa loaded: %zu tensors\n", m.baa.names.size());

  if (test_layer >= 0 && test_layer < 1000) {
    int L = test_layer;
    auto xall = read_f32(vec + "\\x_in.bin");
    auto yg_ref = read_f32(vec + "\\y_gdn.bin");
    auto yf_ref = read_f32(vec + "\\y_full.bin");
    int ntok = (int)(xall.size() / Cfg::H);
    GdnState gst;
    std::vector<float> y1(Cfg::H), y2(Cfg::H);
    auto t0 = std::chrono::steady_clock::now();
    for (int t = 0; t < ntok; t++) {
      gdn_decode(m, L, gst, xall.data() + t * Cfg::H, y1.data());
      std::vector<float> ref_t(yg_ref.begin() + t * Cfg::H, yg_ref.begin() + (t + 1) * Cfg::H);
      compare((std::string("gdn_t") + std::to_string(t)).c_str(), y1, ref_t);
    }
    moe(m, L, y1.data(), y2.data());
    auto t1 = std::chrono::steady_clock::now();
    compare("full", y2, yf_ref);
    printf("layer time: %.1f ms (last token)\n", std::chrono::duration<double>(t1 - t0).count() * 1e3);
    return 0;
  }
  // ---------------- probe：整 prompt 逐层 dump 最后 token 的 hidden（对拍 HF） ----------------
  if (!probe_dir.empty() && !ids_str.empty()) {
    std::vector<int> pids;
    { size_t p = 0; while (p <= ids_str.size()) {
        size_t c = ids_str.find(',', p);
        pids.push_back(atoi(ids_str.substr(p, c == std::string::npos ? ids_str.size() : c - p).c_str()));
        if (c == std::string::npos) break; p = c + 1; } }
    const BaaTensor& emb = m.T("embed_tokens");
    std::vector<GdnState> gst(Cfg::NLAYERS);
    std::vector<AttnState> ast(Cfg::NLAYERS);
    std::vector<float> x(Cfg::H), x2(Cfg::H);
    int npos = 0;
    for (int tok : pids) {
      { const uint16_t* e = emb.bf16() + (size_t)tok * Cfg::H;
        for (int i = 0; i < Cfg::H; i++) x[i] = bf16_to_f32(e[i]); }
      bool last = (npos == (int)pids.size() - 1);
      for (int l = 0; l < Cfg::NLAYERS; l++) {
        if (m.is_gdn(l)) gdn_decode(m, l, gst[l], x.data(), x2.data());
        else attn_decode(m, l, ast[l], npos, x.data(), x2.data());
        moe(m, l, x2.data(), x.data());
        if (last) {
          char fn[256]; sprintf(fn, "%s/ylayer_%02d.bin", probe_dir.c_str(), l);
          FILE* f = fopen(fn, "wb"); fwrite(x.data(), 4, Cfg::H, f); fclose(f);
        }
      }
      npos++;
    }
    fprintf(stderr, "probe done: %d tokens dumped for last token\n", npos);
    return 0;
  }
  // ---------------- 全注意力层对拍（--test-attn N） ----------------
  if (test_layer >= 1000) {
    int L = test_layer - 1000;
    auto xall = read_f32(vec + "\\x_in.bin");
    auto ya_ref = read_f32(vec + "\\y_attn.bin");
    auto yf_ref = read_f32(vec + "\\y_full.bin");
    int ntok = (int)(xall.size() / Cfg::H);
    AttnState ast;
    std::vector<float> y1(Cfg::H), y2(Cfg::H);
    for (int t = 0; t < ntok; t++) {
      attn_decode(m, L, ast, t, xall.data() + t * Cfg::H, y1.data());
      std::vector<float> ref_t(ya_ref.begin() + t * Cfg::H, ya_ref.begin() + (t + 1) * Cfg::H);
      compare((std::string("attn_t") + std::to_string(t)).c_str(), y1, ref_t);
    }
    moe(m, L, y1.data(), y2.data());
    compare("full", y2, yf_ref);
    return 0;
  }
  // ---------------- 阶段 3：全模型贪心 decode ----------------
  if (!ids_str.empty()) {
    std::vector<int> ids;
    { size_t p = 0; while (p <= ids_str.size()) {
        size_t c = ids_str.find(',', p);
        ids.push_back(atoi(ids_str.substr(p, c == std::string::npos ? ids_str.size() : c - p).c_str()));
        if (c == std::string::npos) break; p = c + 1; } }
    const BaaTensor& emb = m.T("embed_tokens");
    const BaaTensor& lmh = m.T("lm_head.weight");
    const BaaTensor& fnorm = m.T("norm");
    fprintf(stderr, "ckpt1: embed=%llu lm_head=%llu norm ok\n", emb.size, lmh.size);
    std::vector<GdnState> gst(Cfg::NLAYERS);
    std::vector<AttnState> ast(Cfg::NLAYERS);
    fprintf(stderr, "ckpt2: states alloc ok\n");
    std::vector<float> x(Cfg::H), x2(Cfg::H), logits(Cfg::VOCAB);
    int next = -1;
    double t_gen0 = 0; bool first = true;
    int pos = 0;
    std::vector<int> out_ids;
    for (int step = 0; step < (int)ids.size() + n_gen; step++) {
      int tok = (step < (int)ids.size()) ? ids[step] : next;
      // embed
      { const uint16_t* e = emb.bf16() + (size_t)tok * Cfg::H;
        for (int i = 0; i < Cfg::H; i++) x[i] = bf16_to_f32(e[i]); }
      auto t0 = std::chrono::steady_clock::now();
      for (int l = 0; l < Cfg::NLAYERS; l++) {
        if (m.is_gdn(l)) gdn_decode(m, l, gst[l], x.data(), x2.data());
        else attn_decode(m, l, ast[l], pos, x.data(), x2.data());
        moe(m, l, x2.data(), x.data());
      }
      rmsnorm(x.data(), fnorm.bf16(), Cfg::H, x.data());
      m.mv(lmh, x.data(), logits.data());
      next = (int)(std::max_element(logits.begin(), logits.end()) - logits.begin());
      auto t1 = std::chrono::steady_clock::now();
      double ms = std::chrono::duration<double>(t1 - t0).count() * 1e3;
      if (step >= (int)ids.size()) { out_ids.push_back(next); }
      printf("step %3d tok %6d → %6d  (%.1f ms%s)\n", step, tok, next, ms,
             step < (int)ids.size() ? " prefill" : "");
      pos++;
    }
    printf("OUT_IDS:");
    for (int t : out_ids) printf("%d,", t);
    printf("\n");
    return 0;
  }
  fprintf(stderr, "nothing to do\n");
  return 1;
} catch (const std::exception& e) { fprintf(stderr, "EXC: %s\n", e.what()); return 2; } }
