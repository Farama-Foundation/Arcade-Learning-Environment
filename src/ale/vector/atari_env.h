/*
 * Copyright 2021 Garena Online Private Limited
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ENVPOOL_ATARI_ATARI_ENV_H_
#define ENVPOOL_ATARI_ATARI_ENV_H_

#include <algorithm>
#include <deque>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "ale/ale_interface.hpp"
#include "ale/vector/async_envpool.h"
#include "ale/vector/env.h"
#include "ale/vector/utils/image_process.h"

namespace atari {

auto TurnOffVerbosity() {
  ale::Logger::setMode(ale::Logger::Error);
  return true;
}

static bool verbosity_off = TurnOffVerbosity();

auto GetRomPath(const std::string& base_path, const std::string& task) {
  std::stringstream ss;
  // hardcode path here :(
  ss << base_path << "/atari/roms/" << task << ".bin";  // TODO update
  return ss.str();
}

class AtariEnvFns {
 public:
  static decltype(auto) DefaultConfig() {
    return MakeDict(
        "stack_num"_.Bind(4), "frame_skip"_.Bind(4), "noop_max"_.Bind(30),
        "zero_discount_on_life_loss"_.Bind(false), "episodic_life"_.Bind(false),
        "reward_clip"_.Bind(false), "use_fire_reset"_.Bind(true),
        "img_height"_.Bind(84), "img_width"_.Bind(84),
        "task"_.Bind(std::string("pong")), "full_action_space"_.Bind(false),
        "repeat_action_probability"_.Bind(0.0f),
        "use_inter_area_resize"_.Bind(true), "gray_scale"_.Bind(true));
  }
  template <typename Config>
  static decltype(auto) StateSpec(const Config& conf) {
    return MakeDict("obs"_.Bind(Spec<uint8_t>(
                        {conf["stack_num"_] * (conf["gray_scale"_] ? 1 : 3),
                         conf["img_height"_], conf["img_width"_]},
                        {0, 255})),
                    "info:lives"_.Bind(Spec<int>({-1})),
                    "info:reward"_.Bind(Spec<float>({-1})),
                    "info:terminated"_.Bind(Spec<int>({-1}, {0, 1})));
  }
  template <typename Config>
  static decltype(auto) ActionSpec(const Config& conf) {
    ale::ALEInterface env;
    env.loadROM(GetRomPath(conf["base_path"_], conf["task"_]));
    int action_size = conf["full_action_space"_]
                          ? env.getLegalActionSet().size()
                          : env.getMinimalActionSet().size();
    return MakeDict("action"_.Bind(Spec<int>({-1}, {0, action_size - 1})));
  }
};

using AtariEnvSpec = EnvSpec<AtariEnvFns>;
using FrameSpec = Spec<uint8_t>;

class AtariEnv : public Env<AtariEnvSpec> {
 protected:
  const int kRawHeight = 210;
  const int kRawWidth = 160;
  const int kRawSize = kRawWidth * kRawHeight;
  std::unique_ptr<ale::ALEInterface> env_;
  ale::ActionVect action_set_;
  int max_episode_steps_, elapsed_step_, stack_num_, frame_skip_;
  bool fire_reset_{false}, reward_clip_, zero_discount_on_life_loss_;
  bool gray_scale_, episodic_life_, use_inter_area_resize_;
  bool done_{true};
  int lives_;
  FrameSpec raw_spec_, resize_spec_, transpose_spec_;
  std::deque<Array> stack_buf_;
  std::vector<Array> maxpool_buf_;
  Array resize_img_;
  std::uniform_int_distribution<> dist_noop_;
  std::string rom_path_;

 public:
  AtariEnv(const Spec& spec, int env_id)
      : Env<AtariEnvSpec>(spec, env_id),
        env_(new ale::ALEInterface()),
        max_episode_steps_(spec.config["max_episode_steps"_]),
        elapsed_step_(max_episode_steps_ + 1),
        stack_num_(spec.config["stack_num"_]),
        frame_skip_(spec.config["frame_skip"_]),
        reward_clip_(spec.config["reward_clip"_]),
        zero_discount_on_life_loss_(spec.config["zero_discount_on_life_loss"_]),
        gray_scale_(spec.config["gray_scale"_]),
        episodic_life_(spec.config["episodic_life"_]),
        use_inter_area_resize_(spec.config["use_inter_area_resize"_]),
        raw_spec_({kRawHeight, kRawWidth, gray_scale_ ? 1 : 3}),
        resize_spec_({spec.config["img_height"_], spec.config["img_width"_],
                      gray_scale_ ? 1 : 3}),
        transpose_spec_({gray_scale_ ? 1 : 3, spec.config["img_height"_],
                         spec.config["img_width"_]}),
        resize_img_(resize_spec_),
        dist_noop_(0, spec.config["noop_max"_] - 1),
        rom_path_(GetRomPath(spec.config["base_path"_], spec.config["task"_])) {
    env_->setFloat("repeat_action_probability",
                   spec.config["repeat_action_probability"_]);
    env_->setInt("random_seed", seed_);
    env_->loadROM(rom_path_);
    if (spec.config["full_action_space"_]) {
      action_set_ = env_->getLegalActionSet();
    } else {
      action_set_ = env_->getMinimalActionSet();
    }
    if (spec.config["use_fire_reset"_]) {
      // https://github.com/sail-sg/envpool/issues/221
      for (auto a : action_set_) {
        if (a == 1) {
          fire_reset_ = true;
        }
      }
    }
    // init buf
    for (int i = 0; i < 2; ++i) {
      maxpool_buf_.emplace_back(Array(raw_spec_));
    }
    for (int i = 0; i < stack_num_; ++i) {
      stack_buf_.emplace_back(Array(transpose_spec_));
    }
  }

  void Reset() override {
    int noop = dist_noop_(gen_) + 1 - static_cast<int>(fire_reset_);
    bool push_all = false;
    if (!episodic_life_ || env_->game_over() ||
        elapsed_step_ >= max_episode_steps_) {
      env_->reset_game();
      elapsed_step_ = 0;
      push_all = true;
    }
    while ((noop--) != 0) {
      env_->act(static_cast<ale::Action>(0));
      if (env_->game_over()) {
        env_->reset_game();
        push_all = true;
      }
    }
    if (fire_reset_) {
      env_->act(static_cast<ale::Action>(1));
    }
    uint8_t* ale_screen_data = env_->getScreen().getArray();
    auto* ptr = static_cast<uint8_t*>(maxpool_buf_[0].Data());
    if (gray_scale_) {
      env_->theOSystem->colourPalette().applyPaletteGrayscale(
          ptr, ale_screen_data, kRawSize);
    } else {
      env_->theOSystem->colourPalette().applyPaletteRGB(ptr, ale_screen_data,
                                                        kRawSize);
    }
    PushStack(push_all, false);
    done_ = false;
    lives_ = env_->lives();
    WriteState(0.0, 1.0, 0.0);
  }

  void Step(const Action& action) override {
    float reward = 0.0;
    done_ = false;
    int act = action["action"_];
    int skip_id = frame_skip_;
    for (; skip_id > 0 && !done_; --skip_id) {
      reward += env_->act(action_set_[act]);
      done_ = env_->game_over();
      if (skip_id <= 2) {  // put final two frames in to maxpool buffer
        uint8_t* ale_screen_data = env_->getScreen().getArray();
        auto* ptr = static_cast<uint8_t*>(maxpool_buf_[2 - skip_id].Data());
        if (gray_scale_) {
          env_->theOSystem->colourPalette().applyPaletteGrayscale(
              ptr, ale_screen_data, kRawSize);
        } else {
          env_->theOSystem->colourPalette().applyPaletteRGB(
              ptr, ale_screen_data, kRawSize);
        }
      }
    }
    // push the maxpool outcome to the stack_buf
    PushStack(false, skip_id == 0);
    ++elapsed_step_;
    done_ |= (elapsed_step_ >= max_episode_steps_);
    if (episodic_life_ && 0 < env_->lives() && env_->lives() < lives_) {
      done_ = true;
    }
    float discount;
    if (zero_discount_on_life_loss_) {
      discount = static_cast<float>(lives_ == env_->lives() && !done_);
    } else {
      discount = 1.0f - static_cast<float>(done_);
    }
    float info_reward = reward;
    if (reward_clip_) {
      if (reward > 0) {
        reward = 1;
      } else if (reward < 0) {
        reward = -1;
      }
    }
    lives_ = env_->lives();
    WriteState(reward, discount, info_reward);
  }

  bool IsDone() override { return done_; }

 private:
  void WriteState(float reward, float discount, float info_reward) {
    State state = Allocate();
    state["discount"_] = discount;
    state["trunc"_] = done_ && (elapsed_step_ >= max_episode_steps_);
    state["reward"_] = reward;
    state["info:lives"_] = lives_;
    state["info:reward"_] = info_reward;
    state["info:terminated"_] = env_->game_over();
    // overwrite current_step to make sure
    // episodic_life == True behaves correctly
    // see Issue #179
    state["elapsed_step"_] = elapsed_step_;
    for (int i = 0; i < stack_num_; ++i) {
      state["obs"_]
          .Slice(gray_scale_ ? i : i * 3, gray_scale_ ? i + 1 : (i + 1) * 3)
          .Assign(stack_buf_[i]);
    }
  }

  /**
   * FrameStack env wrapper implementation.
   *
   * The original gray scale image are saved inside maxpool_buf_.
   * The stacked result is in stack_buf_ where len(stack_buf_) == stack_num_.
   *
   * At reset time, we need to clear all data in stack_buf_ with push_all =
   * true and maxpool = false (there is only one observation); at step time,
   * we push max(maxpool_buf_[0], maxpool_buf_[1]) at the end of
   * stack_buf_, and pop the first item in stack_buf_, with push_all = false
   * and maxpool = true.
   *
   * @param push_all whether to use the most recent observation to write all
   *   of the data in stack_buf_.
   * @param maxpool whether to perform maxpool operation on the last two
   *   observation. Maybe there is only one?
   */
  void PushStack(bool push_all, bool maxpool) {
    auto* ptr = static_cast<uint8_t*>(maxpool_buf_[0].Data());
    if (maxpool) {
      auto* ptr1 = static_cast<uint8_t*>(maxpool_buf_[1].Data());
      for (std::size_t i = 0; i < maxpool_buf_[0].size; ++i) {
        ptr[i] = std::max(ptr[i], ptr1[i]);
      }
    }
    Resize(maxpool_buf_[0], &resize_img_, use_inter_area_resize_);
    Array tgt = std::move(*stack_buf_.begin());
    ptr = static_cast<uint8_t*>(tgt.Data());
    stack_buf_.pop_front();
    if (gray_scale_) {
      tgt.Assign(resize_img_);
    } else {
      auto* ptr1 = static_cast<uint8_t*>(resize_img_.Data());
      // tgt = resize_img_.transpose(1, 2, 0)
      // tgt[i, j, k] = resize_img_[j, k, i]
      std::size_t h = resize_img_.Shape(0);
      std::size_t w = resize_img_.Shape(1);
      for (std::size_t j = 0; j < h; ++j) {
        for (std::size_t k = 0; k < w; ++k) {
          for (std::size_t i = 0; i < 3; ++i) {
            ptr[i * h * w + j * w + k] = ptr1[j * w * 3 + k * 3 + i];
          }
        }
      }
    }
    std::size_t size = tgt.size;
    stack_buf_.push_back(std::move(tgt));
    if (push_all) {
      for (auto& s : stack_buf_) {
        auto* ptr_s = static_cast<uint8_t*>(s.Data());
        if (ptr != ptr_s) {
          std::memcpy(ptr_s, ptr, size);
        }
      }
    }
  }
};

using AtariEnvPool = AsyncEnvPool<AtariEnv>;

}  // namespace atari

#endif  // ENVPOOL_ATARI_ATARI_ENV_H_
