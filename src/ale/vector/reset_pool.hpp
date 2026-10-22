#ifndef ALE_VECTOR_RESET_POOL_HPP_
#define ALE_VECTOR_RESET_POOL_HPP_

#include <cstddef>
#include <random>
#include <vector>

#include "ale/environment/ale_state.hpp"

namespace ale::vector {

/// Pool of start states captured from real `reset_game()` calls.
///
/// `StellaEnvironment::reset()` emulates ~68 frames on every call - 60 NOOP frames
/// of console bring-up, two 4-frame soft resets, then the ROM's starting actions -
/// which costs 3.5-6.3 ms against a few us for `restoreState`.
///
/// A single snapshot is not enough, because `reset_game()` is not a pure function
/// of the ROM: `M6532::reset()` redraws the RIOT timer, the `Cart*SC` families
/// re-randomise their onboard RAM, and the console's 128 bytes are never cleared
/// between episodes. Restoring one snapshot forever would collapse a genuinely
/// varying start state onto a single console power-on, which for a vectorizer -
/// where start diversity across the batch is the whole point - is a real
/// behavioural regression.
///
/// So the first `max_draws` resets of an env are emulated for real and kept, and
/// later resets draw uniformly from them. Every start state an agent can reach is
/// one `reset_game()` actually produced, and the sampled distribution is an
/// unbiased bootstrap of `reset_game()`'s own. Filling costs nothing: those resets
/// would have been emulated anyway.
///
/// Draws are not de-duplicated. Two resets can agree on RAM and screen and still
/// differ in state neither shows - the CPU, RIOT and TIA - which surfaces frames
/// later (elevator_action: one RAM+screen fingerprint after reset, seven distinct
/// futures under one action script). Keeping every draw is exactly as unbiased as
/// weighting de-duplicated ones, and costs `max_draws` states of a few KB each.
class ResetSnapshotPool {
 public:
  struct Entry {
    ALEState state;
    /// Frames `reset_game()` added to the ROM-lifetime frame counter while
    /// producing this state (its starting actions; 0 for most ROMs, up to 16).
    /// A restore must count them too, or `frame_number` falls behind a real reset.
    int reset_frames;
  };

  /// `max_draws <= 0` disables the pool: every reset is emulated for real.
  explicit ResetSnapshotPool(int max_draws) : max_draws_(max_draws > 0 ? max_draws : 0) {}

  bool enabled() const { return max_draws_ > 0; }

  /// True while resets must still be emulated for real - always, when disabled.
  bool filling() const { return static_cast<int>(entries_.size()) < max_draws_ || !enabled(); }

  void clear() { entries_.clear(); }

  /// Record one real reset outcome.
  void add(const ALEState& state, int reset_frames) {
    entries_.push_back(Entry{state, reset_frames});
  }

  /// Draw a start state uniformly. Must not be called while `filling()`.
  const Entry& sample(std::mt19937& rng) const {
    std::uniform_int_distribution<std::size_t> dist(0, entries_.size() - 1);
    return entries_[dist(rng)];
  }

 private:
  std::vector<Entry> entries_;
  int max_draws_;
};

}  // namespace ale::vector

#endif  // ALE_VECTOR_RESET_POOL_HPP_
