// baa.h — .baa 容器 loader（mmap 零拷贝）+ 张量访问（BAA_FORMAT.md v0.1）
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <stdexcept>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

struct BaaTensor {
  uint8_t scheme;      // 0=BF16 1=INT8 2=XBF3
  uint8_t ndim;
  uint64_t shape[4];
  const uint8_t* data; // mmap 指针
  uint64_t size;       // packed_size
  // BF16 访问
  const uint16_t* bf16() const { return (const uint16_t*)data; }
  const float* f32() const { return (const float*)data; }
};

struct Baa {
  std::string arch;
  std::vector<std::string> names;          // 排序
  std::vector<BaaTensor> tensors;
  const uint8_t* base = nullptr;
  size_t file_size = 0;
  std::string meta_json;

#ifdef _WIN32
  HANDLE hFile = INVALID_HANDLE_VALUE;
  HANDLE hMap = nullptr;
#endif

  void open(const std::string& path) {
#ifdef _WIN32
    hFile = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                        OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (hFile == INVALID_HANDLE_VALUE) throw std::runtime_error("open fail: " + path);
    LARGE_INTEGER sz; GetFileSizeEx(hFile, &sz); file_size = (size_t)sz.QuadPart;
    hMap = CreateFileMappingA(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!hMap) throw std::runtime_error("CreateFileMapping fail");
    base = (const uint8_t*)MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
    if (!base) throw std::runtime_error("MapViewOfFile fail");
#else
    throw std::runtime_error("only Windows mmap implemented");
#endif
    if (memcmp(base, "BAA1", 4) != 0) throw std::runtime_error("bad magic");
    uint32_t ver; memcpy(&ver, base + 4, 4);
    if (ver != 1) throw std::runtime_error("bad version");
    arch = (const char*)(base + 8);
    uint64_t tc, dir_off, meta_off, meta_len, data_off;
    memcpy(&tc, base + 40, 8); memcpy(&dir_off, base + 48, 8);
    memcpy(&meta_off, base + 56, 8); memcpy(&meta_len, base + 64, 8);
    memcpy(&data_off, base + 72, 8);
    meta_json.assign((const char*)(base + meta_off), meta_len);
    const uint8_t* dir = base + dir_off;
    const char* pool = (const char*)(dir + 96 * tc);
    for (uint64_t i = 0; i < tc; i++) {
      const uint8_t* e = dir + 96 * i;
      uint32_t noff; memcpy(&noff, e, 4);
      BaaTensor t;
      t.scheme = e[4]; t.ndim = e[6];
      memcpy(t.shape, e + 8, 32);
      uint64_t doff, psz;
      memcpy(&doff, e + 40, 8); memcpy(&psz, e + 48, 8);
      t.data = base + data_off + doff; t.size = psz;
      names.push_back(pool + noff);
      tensors.push_back(t);
    }
  }

  const BaaTensor& at(const std::string& name) const {
    // 目录按规范化名排序（writer 保证）；保险起见二分失败时线性兜底
    size_t lo = 0, hi = names.size();
    while (lo < hi) {
      size_t mid = (lo + hi) / 2;
      if (names[mid] < name) lo = mid + 1; else hi = mid;
    }
    if (lo < names.size() && names[lo] == name) return tensors[lo];
    for (size_t i = 0; i < names.size(); i++)
      if (names[i] == name) return tensors[i];
    throw std::runtime_error("tensor not found: " + name);
  }
  bool has(const std::string& name) const {
    for (size_t i = 0; i < names.size(); i++) if (names[i] == name) return true;
    return false;
  }
};
