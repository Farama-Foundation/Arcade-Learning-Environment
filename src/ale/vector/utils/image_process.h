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

#ifndef ENVPOOL_UTILS_IMAGE_PROCESS_H_
#define ENVPOOL_UTILS_IMAGE_PROCESS_H_

#include <opencv2/opencv.hpp>

#include "ale/vector/array.h"

/**
 * Resize `src` image to `tgt`. Use inplace modification to reduce overhead.
 */
void Resize(const Array& src, Array* tgt, bool use_inter_area = true) {
  int channel = src.Shape(2);
  cv::Mat src_img(src.Shape(0), src.Shape(1), CV_8UC(channel), src.Data());
  cv::Mat tgt_img(tgt->Shape(0), tgt->Shape(1), CV_8UC(channel), tgt->Data());
  if (use_inter_area) {
    cv::resize(src_img, tgt_img, tgt_img.size(), 0, 0, cv::INTER_AREA);
  } else {
    cv::resize(src_img, tgt_img, tgt_img.size());
  }
}

/**
 * Change src (with RGB format) to grayscale image.
 */
void GrayScale(const Array& src, Array* tgt) {
  cv::Mat src_img(src.Shape(0), src.Shape(1), CV_8UC3, src.Data());
  cv::Mat tgt_img(tgt->Shape(0), tgt->Shape(1), CV_8UC1, tgt->Data());
  cv::cvtColor(src_img, tgt_img, cv::COLOR_RGB2GRAY);
}

#endif  // ENVPOOL_UTILS_IMAGE_PROCESS_H_
