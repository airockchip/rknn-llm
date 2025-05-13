// Copyright (c) 2024 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <opencv2/opencv.hpp>

#include "image_enc.h"

#define IMAGE_HEIGHT 392
#define IMAGE_WIDTH 392
#define IMAGE_TOKEN_NUM 196
#define EMBED_SIZE 1536

// Expand the image into a square and fill it with the specified background color
cv::Mat expand2square(const cv::Mat& img, const cv::Scalar& background_color) {
    int width = img.cols;
    int height = img.rows;

    // If the width and height are equal, return to the original image directly
    if (width == height) {
        return img.clone();
    }

    // Calculate the new size and create a new image
    int size = std::max(width, height);
    cv::Mat result(size, size, img.type(), background_color);

    // Calculate the image paste position
    int x_offset = (size - width) / 2;
    int y_offset = (size - height) / 2;

    // Paste the original image into the center of the new image
    cv::Rect roi(x_offset, y_offset, width, height);
    img.copyTo(result(roi));

    return result;
}

int main(int argc, char** argv)
{
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " model_path image_path core_num\n";
        return -1;
    }

    const char * model_path = argv[1];
    const char * image_path = argv[2];
    const int core_num = atoi(argv[3]);


    int ret;
    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

    std::chrono::high_resolution_clock::time_point t_start_us = std::chrono::high_resolution_clock::now();

    ret = init_imgenc(model_path, &rknn_app_ctx, core_num);
    if (ret != 0) {
        printf("init_imgenc fail! ret=%d model_path=%s\n", ret, model_path);
        return -1;
    }
    std::chrono::high_resolution_clock::time_point t_load_end_us = std::chrono::high_resolution_clock::now();

    auto load_time = std::chrono::duration_cast<std::chrono::microseconds>(t_load_end_us - t_start_us);
    printf("%s: Model loaded in %8.2f ms\n", __func__, load_time.count() / 1000.0);

    // The image is read in BGR format
    cv::Mat img = cv::imread(image_path);
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    // Expand the image into a square and fill it with the specified background color (According the modeling_minicpmv.py)
    cv::Scalar background_color(127.5, 127.5, 127.5);
    cv::Mat square_img = expand2square(img, background_color);

    // Resize the image to 448x448
    cv::Mat resized_img;
    cv::Size new_size(IMAGE_WIDTH, IMAGE_HEIGHT);
    cv::resize(square_img, resized_img, new_size, 0, 0, cv::INTER_LINEAR);

    std::chrono::high_resolution_clock::time_point t_every_begin_us;
    std::chrono::high_resolution_clock::time_point t_every_end_us;
    float img_vec[IMAGE_TOKEN_NUM * EMBED_SIZE];
    t_every_begin_us = std::chrono::high_resolution_clock::now();
    ret = run_imgenc(&rknn_app_ctx, resized_img.data, img_vec);
    if (ret != 0) {
        printf("run_imgenc fail! ret=%d\n", ret);
    }
    t_every_end_us = std::chrono::high_resolution_clock::now();
    auto encoder_time = std::chrono::duration_cast<std::chrono::microseconds>(t_every_end_us - t_every_begin_us);
    printf("%s: Encoder the image cost %8.2f ms\n", __func__, encoder_time.count() / 1000.0);
    
    // Writes the array img_vec to the file
    std::ofstream file("./img_vec.bin", std::ios::binary);
    file.write(reinterpret_cast<char*>(img_vec), sizeof(img_vec));
    file.close();

    ret = release_imgenc(&rknn_app_ctx);
    if (ret != 0) {
        printf("release_imgenc fail! ret=%d\n", ret);
    }

    return 0;
}