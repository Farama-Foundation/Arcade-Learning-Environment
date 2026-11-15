#ifndef ALE_VECTOR_AREA_RESIZE_HPP_
#define ALE_VECTOR_AREA_RESIZE_HPP_

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <tuple>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>

// SSE2 is part of the x86-64 baseline. GCC/Clang announce it with __SSE2__;
// MSVC does not, so it is inferred from the target.
#if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
  #define ALE_AREA_RESIZE_SSE2 1
  #include <emmintrin.h>
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
  #include <arm_neon.h>
#endif

namespace ale::vector {

/**
 * Maxpool two uint8_t frames in place: dst[i] = max(dst[i], src[i]).
 */
inline void maxpool_frames(uint8_t* dst, const uint8_t* src, int size) {
  int i = 0;
#if defined(ALE_AREA_RESIZE_SSE2)
  for (; i + 16 <= size; i += 16) {
    const __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(dst + i));
    const __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i), _mm_max_epu8(a, b));
  }
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
  for (; i + 16 <= size; i += 16) {
    vst1q_u8(dst + i, vmaxq_u8(vld1q_u8(dst + i), vld1q_u8(src + i)));
  }
#endif
  for (; i < size; ++i) {
    dst[i] = std::max(dst[i], src[i]);
  }
}

/**
 * A bit-exact, fused replacement for `maxpool_frames` + `cv::resize(INTER_AREA)`
 * on single-channel uint8 frames.
 *
 * This is not an approximation of the area filter: the decimation tables, the
 * order of the float accumulations and the final rounding all mirror OpenCV's
 * general area path (`imgproc/src/resize.cpp`: `computeResizeAreaTab` +
 * `ResizeArea_Invoker`). The gain comes from doing it in one traversal: each
 * source row is maxpooled, widened to float and reduced to a row of horizontal
 * box sums exactly once, where OpenCV re-reads a source row for every output row
 * that overlaps it (252 row visits instead of 210 at 210x160 -> 84x84) and
 * accumulates taps one scalar at a time.
 *
 * `create()` checks this rather than assuming it: it runs the kernel and
 * `cv::resize` against each other on synthetic frames and returns no plan if
 * they ever differ, so a build whose floating-point behaviour differs (FMA
 * contraction, another OpenCV) falls back to `cv::resize` instead of changing
 * observations.
 */
class AreaResizePlan {
 public:
  /// Per-env scratch, owned by the caller so the kernel never allocates.
  struct Scratch {
    std::vector<float> gf;      ///< one source row widened to float (padded)
    std::vector<float> rowbuf;  ///< horizontal box sums of that row
    std::vector<float> sum;     ///< vertical accumulator for the current dst row
  };

  /// Shared, immutable plan for one geometry; nullptr if fusion is not usable.
  /// Safe to call from the parallel env constructors: the cache is guarded by a
  /// mutex and the plans are never modified after `create()`.
  static std::shared_ptr<const AreaResizePlan> get(int src_w, int src_h, int dst_w, int dst_h) {
    static std::mutex mutex;
    static std::map<std::tuple<int, int, int, int>, std::shared_ptr<const AreaResizePlan>> cache;

    const auto key = std::make_tuple(src_w, src_h, dst_w, dst_h);
    std::lock_guard<std::mutex> guard(mutex);
    auto it = cache.find(key);
    if (it == cache.end()) {
      it = cache.emplace(key, create(src_w, src_h, dst_w, dst_h)).first;
    }
    return it->second;
  }

  void init_scratch(Scratch& scratch) const {
    // gf is read 4 floats at a time at every tap offset and written 16 at a
    // time, so it carries zero padding at the end; the padding is only ever
    // multiplied by a zero weight.
    scratch.gf.assign(((src_w_ + 15) / 16) * 16 + 8, 0.0f);
    scratch.rowbuf.assign(dst_w_, 0.0f);
    scratch.sum.assign(dst_w_, 0.0f);
  }

  /**
   * Maxpool `frame1` into `frame0` (skipped when `frame1` is null) and write the
   * area-resized result to `dst`.
   *
   * `frame0` is modified in place, exactly as `maxpool_frames` does: the next
   * step's early-break paths observe the maxpooled frame, so the write-back is
   * part of the contract.
   */
  void run(uint8_t* frame0, const uint8_t* frame1, uint8_t* dst, Scratch& scratch) const {
    const int sw = src_w_, dw = dst_w_;
    float* gf = scratch.gf.data();
    float* rowbuf = scratch.rowbuf.data();
    float* sum = scratch.sum.data();

    int last_sy = -1;
    for (int dy = 0; dy < dst_h_; ++dy) {
      for (int j = y_ofs_[dy]; j < y_ofs_[dy + 1]; ++j) {
        const int sy = y_si_[j];
        if (sy != last_sy) {
          uint8_t* row = frame0 + static_cast<size_t>(sy) * sw;
          if (frame1 != nullptr) {
            maxpool_frames(row, frame1 + static_cast<size_t>(sy) * sw, sw);
          }
          widen_row(row, sw, gf);
          horizontal_pass(gf, rowbuf);
          last_sy = sy;
        }
        const float beta = y_alpha_[j];
        if (j == y_ofs_[dy]) {
          for (int dx = 0; dx < dw; ++dx) {
            sum[dx] = beta * rowbuf[dx];
          }
        } else {
          for (int dx = 0; dx < dw; ++dx) {
            sum[dx] += beta * rowbuf[dx];
          }
        }
      }
      store_row(sum, dw, dst + static_cast<size_t>(dy) * dw);
    }
  }

 private:
  using Taps = std::vector<std::vector<std::pair<int, float>>>;

  AreaResizePlan() = default;

  /// uint8 source row -> float, so the tap loop does no per-tap conversion.
  static void widen_row(const uint8_t* row, int sw, float* gf) {
    int x = 0;
#if defined(ALE_AREA_RESIZE_SSE2)
    const __m128i zero = _mm_setzero_si128();
    for (; x + 16 <= sw; x += 16) {
      const __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row + x));
      const __m128i lo = _mm_unpacklo_epi8(b, zero);
      const __m128i hi = _mm_unpackhi_epi8(b, zero);
      _mm_storeu_ps(gf + x + 0, _mm_cvtepi32_ps(_mm_unpacklo_epi16(lo, zero)));
      _mm_storeu_ps(gf + x + 4, _mm_cvtepi32_ps(_mm_unpackhi_epi16(lo, zero)));
      _mm_storeu_ps(gf + x + 8, _mm_cvtepi32_ps(_mm_unpacklo_epi16(hi, zero)));
      _mm_storeu_ps(gf + x + 12, _mm_cvtepi32_ps(_mm_unpackhi_epi16(hi, zero)));
    }
#endif
    for (; x < sw; ++x) {
      gf[x] = static_cast<float>(row[x]);
    }
  }

  /**
   * rowbuf[dx] = sum over the dx'th cell's taps of gf[s0+k] * alpha[k].
   *
   * The taps of one output pixel are always consecutive source pixels, so the
   * weights live in a fixed 4-wide slot per dx, zero-padded. The SSE2 path
   * transposes four products so each lane is reduced as `((t0+t1)+t2)+t3`,
   * which is the order OpenCV accumulates taps in; the padded taps are exact
   * zeros (finite * 0.0f), and adding an exact zero does not round.
   */
  void horizontal_pass(const float* gf, float* rowbuf) const {
    const int dw = dst_w_;
    int dx = 0;
#if defined(ALE_AREA_RESIZE_SSE2)
    for (; dx + 4 <= dw; dx += 4) {
      __m128 v0 = _mm_mul_ps(_mm_loadu_ps(gf + x_s0_[dx + 0]), _mm_loadu_ps(&x_alpha4_[(dx + 0) * 4]));
      __m128 v1 = _mm_mul_ps(_mm_loadu_ps(gf + x_s0_[dx + 1]), _mm_loadu_ps(&x_alpha4_[(dx + 1) * 4]));
      __m128 v2 = _mm_mul_ps(_mm_loadu_ps(gf + x_s0_[dx + 2]), _mm_loadu_ps(&x_alpha4_[(dx + 2) * 4]));
      __m128 v3 = _mm_mul_ps(_mm_loadu_ps(gf + x_s0_[dx + 3]), _mm_loadu_ps(&x_alpha4_[(dx + 3) * 4]));
      // After the transpose v_k holds tap k of all four output pixels.
      _MM_TRANSPOSE4_PS(v0, v1, v2, v3);
      _mm_storeu_ps(rowbuf + dx, _mm_add_ps(_mm_add_ps(_mm_add_ps(v0, v1), v2), v3));
    }
#endif
    for (; dx < dw; ++dx) {
      const float* alpha = &x_alpha4_[static_cast<size_t>(dx) * 4];
      const float* g = gf + x_s0_[dx];
      float acc = g[0] * alpha[0];
      for (int k = 1; k < max_taps_x_; ++k) {
        acc += g[k] * alpha[k];
      }
      rowbuf[dx] = acc;
    }
  }

  /// Round half to even and clamp to [0, 255], i.e. `cv::saturate_cast<uchar>`.
  static void store_row(const float* sum, int dw, uint8_t* dst) {
    int dx = 0;
#if defined(ALE_AREA_RESIZE_SSE2)
    for (; dx + 16 <= dw; dx += 16) {
      const __m128i a = _mm_cvtps_epi32(_mm_loadu_ps(sum + dx + 0));
      const __m128i b = _mm_cvtps_epi32(_mm_loadu_ps(sum + dx + 4));
      const __m128i c = _mm_cvtps_epi32(_mm_loadu_ps(sum + dx + 8));
      const __m128i d = _mm_cvtps_epi32(_mm_loadu_ps(sum + dx + 12));
      _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + dx),
                       _mm_packus_epi16(_mm_packs_epi32(a, b), _mm_packs_epi32(c, d)));
    }
#endif
    for (; dx < dw; ++dx) {
      dst[dx] = cv::saturate_cast<uint8_t>(sum[dx]);
    }
  }

  /// Mirrors cv::computeResizeAreaTab for one axis, grouped by output index.
  static Taps build_axis(int ssize, int dsize, double scale) {
    Taps taps(dsize);
    for (int dx = 0; dx < dsize; ++dx) {
      const double fsx1 = dx * scale;
      const double fsx2 = fsx1 + scale;
      const double cell_width = std::min(scale, ssize - fsx1);

      int sx1 = cvCeil(fsx1), sx2 = cvFloor(fsx2);
      sx2 = std::min(sx2, ssize - 1);
      sx1 = std::min(sx1, sx2);

      if (sx1 - fsx1 > 1e-3) {
        taps[dx].emplace_back(sx1 - 1, static_cast<float>((sx1 - fsx1) / cell_width));
      }
      for (int sx = sx1; sx < sx2; ++sx) {
        taps[dx].emplace_back(sx, static_cast<float>(1.0 / cell_width));
      }
      if (fsx2 - sx2 > 1e-3) {
        taps[dx].emplace_back(
            sx2, static_cast<float>(std::min(std::min(fsx2 - sx2, 1.), cell_width) / cell_width));
      }
    }
    return taps;
  }

  /// The horizontal kernel holds one output pixel's weights in a 4-wide slot, so
  /// its taps must be at most four consecutive source pixels. The vertical pass
  /// walks a flat table and has no such restriction.
  static bool x_taps_supported(const Taps& taps) {
    for (const auto& cell : taps) {
      if (cell.empty() || cell.size() > 4) {
        return false;
      }
      for (size_t k = 1; k < cell.size(); ++k) {
        if (cell[k].first != cell[k - 1].first + 1) {
          return false;
        }
      }
    }
    return true;
  }

  static std::shared_ptr<const AreaResizePlan> create(int src_w, int src_h, int dst_w, int dst_h) {
    if (src_w <= 0 || src_h <= 0 || dst_w <= 0 || dst_h <= 0) {
      return nullptr;
    }

    // Reproduce cv::resize's own scale derivation: it recomputes the inverse
    // scale from the destination size, and 1./(84./160.) is not necessarily
    // 160./84.
    const double inv_scale_x = static_cast<double>(dst_w) / src_w;
    const double inv_scale_y = static_cast<double>(dst_h) / src_h;
    const double scale_x = 1. / inv_scale_x, scale_y = 1. / inv_scale_y;
    const int iscale_x = cv::saturate_cast<int>(scale_x);
    const int iscale_y = cv::saturate_cast<int>(scale_y);
    const bool is_area_fast = std::abs(scale_x - iscale_x) < DBL_EPSILON &&
                              std::abs(scale_y - iscale_y) < DBL_EPSILON;

    // Only the general area path is reproduced. An integer scale sends OpenCV
    // to resizeAreaFast (integer accumulation, different rounding) and an
    // upscale sends it to a bilinear emulation.
    if (!(scale_x >= 1 && scale_y >= 1) || is_area_fast) {
      return nullptr;
    }

    const Taps xtaps = build_axis(src_w, dst_w, scale_x);
    const Taps ytaps = build_axis(src_h, dst_h, scale_y);
    if (!x_taps_supported(xtaps)) {
      return nullptr;
    }

    auto plan = std::shared_ptr<AreaResizePlan>(new AreaResizePlan());
    plan->src_w_ = src_w;
    plan->src_h_ = src_h;
    plan->dst_w_ = dst_w;
    plan->dst_h_ = dst_h;

    plan->max_taps_x_ = 1;
    plan->x_s0_.assign(dst_w, 0);
    plan->x_alpha4_.assign(static_cast<size_t>(dst_w) * 4, 0.0f);
    for (int dx = 0; dx < dst_w; ++dx) {
      plan->max_taps_x_ = std::max(plan->max_taps_x_, static_cast<int>(xtaps[dx].size()));
      plan->x_s0_[dx] = xtaps[dx][0].first;
      for (size_t k = 0; k < xtaps[dx].size(); ++k) {
        plan->x_alpha4_[static_cast<size_t>(dx) * 4 + k] = xtaps[dx][k].second;
      }
    }

    // The maxpool write-back only happens for rows the vertical pass visits, so
    // every source row must be visited or frame0 would differ from what
    // maxpool_frames leaves behind.
    std::vector<bool> row_seen(src_h, false);
    plan->y_ofs_.assign(dst_h + 1, 0);
    for (int dy = 0; dy < dst_h; ++dy) {
      plan->y_ofs_[dy] = static_cast<int>(plan->y_si_.size());
      for (const auto& tap : ytaps[dy]) {
        plan->y_si_.push_back(tap.first);
        plan->y_alpha_.push_back(tap.second);
        row_seen[tap.first] = true;
      }
    }
    plan->y_ofs_[dst_h] = static_cast<int>(plan->y_si_.size());
    if (std::find(row_seen.begin(), row_seen.end(), false) != row_seen.end()) {
      return nullptr;
    }

    return plan->self_check() ? plan : nullptr;
  }

  /**
   * Run the kernel and cv::resize against each other on synthetic frames, with
   * and without the maxpool. Any difference in the output or in the written-back
   * frame disqualifies the plan.
   */
  bool self_check() const {
    const size_t frame_size = static_cast<size_t>(src_w_) * src_h_;
    std::vector<uint8_t> a(frame_size), b(frame_size);
    std::vector<uint8_t> fused(static_cast<size_t>(dst_w_) * dst_h_);
    std::vector<uint8_t> reference(fused.size());
    Scratch scratch;
    init_scratch(scratch);

    uint32_t state = 0x9e3779b9u;
    auto next = [&state]() {
      state ^= state << 13;
      state ^= state >> 17;
      state ^= state << 5;
      return static_cast<uint8_t>(state & 0xff);
    };

    for (int trial = 0; trial < 4; ++trial) {
      for (size_t i = 0; i < frame_size; ++i) {
        if (trial < 2) {  // noise: exercises every rounding boundary
          a[i] = next();
          b[i] = next();
        } else {  // blocky: flat regions and hard edges, like a real frame
          const int x = static_cast<int>(i % src_w_), y = static_cast<int>(i / src_w_);
          a[i] = static_cast<uint8_t>(((x / 7) ^ (y / 5)) * 13);
          b[i] = static_cast<uint8_t>(((x / 3) + (y / 11)) * 7);
        }
      }
      const bool with_maxpool = (trial % 2) == 0;

      std::vector<uint8_t> a_fused = a, a_ref = a;
      run(a_fused.data(), with_maxpool ? b.data() : nullptr, fused.data(), scratch);

      if (with_maxpool) {
        maxpool_frames(a_ref.data(), b.data(), static_cast<int>(frame_size));
      }
      cv::Mat src_img(src_h_, src_w_, CV_8UC1, a_ref.data());
      cv::Mat dst_img(dst_h_, dst_w_, CV_8UC1, reference.data());
      cv::resize(src_img, dst_img, dst_img.size(), 0, 0, cv::INTER_AREA);

      if (fused != reference || a_fused != a_ref) {
        return false;
      }
    }
    return true;
  }

  int src_w_ = 0, src_h_ = 0, dst_w_ = 0, dst_h_ = 0;
  int max_taps_x_ = 0;
  std::vector<int> x_s0_;        ///< first source pixel of each output pixel
  std::vector<float> x_alpha4_;  ///< 4 weights per output pixel, zero-padded
  std::vector<int> y_ofs_;       ///< dst_h + 1 offsets into y_si_ / y_alpha_
  std::vector<int> y_si_;
  std::vector<float> y_alpha_;
};

}  // namespace ale::vector

#endif  // ALE_VECTOR_AREA_RESIZE_HPP_
