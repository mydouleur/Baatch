// main.cpp — Baatch 引擎 stage 2/3 驱动
//   --test-layer N --vec <dir>   单层对拍（x_in.bin/y_gdn.bin/y_full.bin）
//   --decode --ids "1,2,3" --n 8 全模型贪心 decode（阶段 3）
#include "baa.h"
#include "model.h"
#include "prefill.h"
#include "moe_prefill.h"
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
  std::string npu_dir;
  bool use_prefill = false, npu_experts = false;
  for (int i = 1; i < argc; i++) {
    std::string s = argv[i];
    if (s == "--baa") baa = argv[++i];
    else if (s == "--test-layer") test_layer = atoi(argv[++i]);
    else if (s == "--vec") vec = argv[++i];
    else if (s == "--test-attn") test_layer = 1000 + atoi(argv[++i]);
    else if (s == "--ids") ids_str = argv[++i];
    else if (s == "--n") n_gen = atoi(argv[++i]);
    else if (s == "--probe") probe_dir = argv[++i];
    else if (s == "--npu-dir") npu_dir = argv[++i];
    else if (s == "--prefill") use_prefill = true;
    else if (s == "--npu-xbf4-experts") npu_experts = true;
  }
  Model m;
  m.open(baa);
  NpuBackend npu;
  if (!npu_dir.empty()) { npu.init(npu_dir, npu_experts); m.npu = &npu; fprintf(stderr, "npu backend on: %s\n", npu_dir.c_str()); }
  fprintf(stderr, "baa loaded: %zu tensors\n", m.baa.names.size());

  if (test_layer >= 0 && test_layer < 1000) {
    int L = test_layer;
    auto xall = read_f32(vec + "\\x_in.bin");
    auto yg_ref = read_f32(vec + "\\y_gdn.bin");
    auto yf_ref = read_f32(vec + "\\y_full.bin");
    int ntok = (int)(xall.size() / Cfg::H);
    GdnState gst;
    std::vector<float> y1((size_t)ntok * Cfg::H), y2(Cfg::H);
    auto t0 = std::chrono::steady_clock::now();
    if (use_prefill) gdn_prefill(m, L, gst, xall.data(), ntok, y1.data());
    for (int t = 0; t < ntok; t++) {
      if (!use_prefill) gdn_decode(m, L, gst, xall.data() + t * Cfg::H, y1.data() + t * Cfg::H);
      std::vector<float> ref_t(yg_ref.begin() + t * Cfg::H, yg_ref.begin() + (t + 1) * Cfg::H);
      compare((std::string("gdn_t") + std::to_string(t)).c_str(),
              std::vector<float>(y1.begin() + t * Cfg::H, y1.begin() + (t + 1) * Cfg::H), ref_t);
    }
    moe(m, L, y1.data() + (ntok - 1) * Cfg::H, y2.data());
    auto t1 = std::chrono::steady_clock::now();
    { FILE* f = fopen("y_cpp_gdn.bin", "wb"); fwrite(y1.data() + (ntok - 1) * Cfg::H, 4, Cfg::H, f); fclose(f); }
    { FILE* f = fopen("y_cpp_full.bin", "wb"); fwrite(y2.data(), 4, Cfg::H, f); fclose(f); }
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
      { if (emb.scheme == 3)
          dequant_xbf4_row(emb.data + (size_t)tok * (Cfg::H / 256) * 130, Cfg::H, x.data());
        else if (emb.scheme == 5) {
          const float* e = emb.f32() + (size_t)tok * Cfg::H;
          memcpy(x.data(), e, Cfg::H * 4);
        } else {
          const uint16_t* e = emb.bf16() + (size_t)tok * Cfg::H;
          for (int i = 0; i < Cfg::H; i++) x[i] = bf16_to_f32(e[i]);
        } }
      bool last = (npos == 0);   // 首 token（零状态，直接可比）
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
    std::vector<float> y1((size_t)ntok * Cfg::H), y2(Cfg::H);
    if (use_prefill) attn_prefill(m, L, ast, 0, xall.data(), ntok, y1.data());
    for (int t = 0; t < ntok; t++) {
      if (!use_prefill) attn_decode(m, L, ast, t, xall.data() + t * Cfg::H, y1.data() + t * Cfg::H);
      std::vector<float> ref_t(ya_ref.begin() + t * Cfg::H, ya_ref.begin() + (t + 1) * Cfg::H);
      compare((std::string("attn_t") + std::to_string(t)).c_str(),
              std::vector<float>(y1.begin() + t * Cfg::H, y1.begin() + (t + 1) * Cfg::H), ref_t);
    }
    moe(m, L, y1.data() + (ntok - 1) * Cfg::H, y2.data());
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
    g_prof.enabled = getenv("PROF") != nullptr;
    g_prof.reset();
    npu.reset_stats();
    const BaaTensor& emb = m.T("embed_tokens");
    const BaaTensor& lmh = m.T("lm_head.weight");
    const BaaTensor& fnorm = m.T("norm");
    fprintf(stderr, "ckpt1: embed=%llu lm_head=%llu norm ok\n", emb.size, lmh.size);
    std::vector<GdnState> gst(Cfg::NLAYERS);
    std::vector<AttnState> ast(Cfg::NLAYERS);
    std::vector<std::unique_ptr<MoeLayerPlan>> moe_plans;
    for (int l = 0; l < Cfg::NLAYERS; l++) moe_plans.push_back(std::make_unique<MoeLayerPlan>(m, l));
    std::unique_ptr<MoeScheduler> moe_scheduler;
    const char* scheduled = getenv("MOE_SCHEDULED");
    if (!scheduled || *scheduled != '0') moe_scheduler = std::make_unique<MoeScheduler>(m);
    fprintf(stderr, "ckpt2: states alloc ok\n");
    std::vector<float> logits(Cfg::VOCAB);
    auto embed = [&](int tok, float* x) {
      if (emb.scheme == 3)
        dequant_xbf4_row(emb.data + (size_t)tok * (Cfg::H / 256) * 130, Cfg::H, x);
      else if (emb.scheme == 5)
        memcpy(x, emb.f32() + (size_t)tok * Cfg::H, Cfg::H * 4);
      else {
        const uint16_t* e = emb.bf16() + (size_t)tok * Cfg::H;
        for (int i = 0; i < Cfg::H; i++) x[i] = bf16_to_f32(e[i]);
      }
    };
    std::vector<float> dx(Cfg::H), dx2(Cfg::H);
    auto decode_one = [&](int tok, int pos, double& ms) {
      embed(tok, dx.data());
      auto t0 = std::chrono::steady_clock::now();
      for (int l = 0; l < Cfg::NLAYERS; l++) {
        if (m.is_gdn(l)) gdn_decode(m, l, gst[l], dx.data(), dx2.data());
        else attn_decode(m, l, ast[l], pos, dx.data(), dx2.data());
        auto mt = prof_now();
        if (moe_scheduler) moe_scheduler->run(*moe_plans[l], dx2.data(), dx.data());
        else moe(m, l, dx2.data(), dx.data());
        prof_add(g_prof.moe[l], mt);
      }
      auto lh = prof_now();
      m.nrm(fnorm, dx.data(), Cfg::H, dx.data(), true);
      if (env_flag_default("CPU_LM_HEAD")) cpu_mv_pretiled_fast(lmh, dx.data(), logits.data(), &m.pool);
      else m.mv(lmh, dx.data(), logits.data());
      int result = (int)(std::max_element(logits.begin(), logits.end()) - logits.begin());
      prof_add(g_prof.lm_head, lh);
      ms = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() * 1e3;
      return result;
    };

    int next = -1, pos = 0;
    std::vector<int> out_ids;
    if (use_prefill && ids.size() > 1) {
      int T = ids.size();
      std::vector<float> x((size_t)T * Cfg::H), x2((size_t)T * Cfg::H);
      MoePrefillWorkspace prefill_workspace(T);
      std::exception_ptr warm_error;
      std::thread warmer([&] {
        SetThreadAffinityMask(GetCurrentThread(), 1ull << 0);
        try {
          for (int l = 0; l < Cfg::NLAYERS; l++)
            cpu_prepare_pretiled(m.T(LN(l, m.is_gdn(l) ? "linear_attn.out_proj.weight" : "self_attn.o_proj.weight")));
          cpu_prepare_pretiled(lmh);
        } catch (...) {
          warm_error = std::current_exception();
        }
      });
      for (int t = 0; t < T; t++) embed(ids[t], x.data() + (size_t)t * Cfg::H);
      auto t0 = std::chrono::steady_clock::now();
      for (int l = 0; l < Cfg::NLAYERS; l++) {
        if (m.is_gdn(l)) gdn_prefill(m, l, gst[l], x.data(), T, x2.data());
        else attn_prefill(m, l, ast[l], 0, x.data(), T, x2.data());
        auto mt = prof_now();
        moe_prefill(m, *moe_plans[l], prefill_workspace, x2.data(), x.data());
        prof_add(g_prof.moe[l], mt);
      }
      warmer.join();
      if (warm_error) std::rethrow_exception(warm_error);
      float* last = x.data() + (size_t)(T - 1) * Cfg::H;
      m.nrm(fnorm, last, Cfg::H, last, true);
      m.mv(lmh, last, logits.data());
      next = (int)(std::max_element(logits.begin(), logits.end()) - logits.begin());
      double ms = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() * 1e3;
      printf("prefill %d tokens (%.1f ms) -> %d\n", T, ms, next);
      pos = T;
      for (int step = 0; step < n_gen; step++) {
        int tok = next;
        next = decode_one(tok, pos++, ms);
        out_ids.push_back(next);
        printf("step %3d tok %6d → %6d  (%.1f ms)\n", T + step, tok, next, ms);
      }
    } else {
      for (int step = 0; step < (int)ids.size() + n_gen; step++) {
        int tok = (step < (int)ids.size()) ? ids[step] : next;
        double ms;
        next = decode_one(tok, pos++, ms);
        if (step >= (int)ids.size()) out_ids.push_back(next);
        printf("step %3d tok %6d → %6d  (%.1f ms%s)\n", step, tok, next, ms,
               step < (int)ids.size() ? " prefill" : "");
      }
    }
    printf("OUT_IDS:");
    for (int t : out_ids) printf("%d,", t);
    printf("\n");
    if (g_prof.enabled) {
      double tp = 0, ts = 0, to = 0, tm = 0;
      size_t kv_bytes = 0, gdn_bytes = 0;
      for (int l = 0; l < Cfg::NLAYERS; l++) {
        bool gdn = m.is_gdn(l);
        double p = gdn ? g_prof.gdn_proj[l] : g_prof.attn_proj[l];
        double s = gdn ? g_prof.gdn_state[l] : g_prof.attn_state[l];
        double o = gdn ? g_prof.gdn_out[l] : g_prof.attn_out[l];
        printf("[prof] layer=%02d proj_ms=%.3f state_ms=%.3f out_ms=%.3f moe_ms=%.3f\n",
               l, p, s, o, g_prof.moe[l]);
        tp += p; ts += s; to += o; tm += g_prof.moe[l];
        kv_bytes += (ast[l].k.size() + ast[l].v.size()) * sizeof(float);
        gdn_bytes += (gst[l].ssm.size() + gst[l].conv.size()) * sizeof(float);
      }
      printf("[prof] totals proj_ms=%.3f state_ms=%.3f out_ms=%.3f moe_ms=%.3f lm_head_ms=%.3f "
             "kv_MB=%.3f gdn_MB=%.3f\n", tp, ts, to, tm, g_prof.lm_head,
             kv_bytes / 1e6, gdn_bytes / 1e6);
      npu.print_stats();
    }
    return 0;
  }
  fprintf(stderr, "nothing to do\n");
  return 1;
} catch (const std::exception& e) { fprintf(stderr, "EXC: %s\n", e.what()); return 2; } }
