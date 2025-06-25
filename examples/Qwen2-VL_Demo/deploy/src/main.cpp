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
#include "rkllm.h"

#define IMAGE_HEIGHT 392
#define IMAGE_WIDTH 392
#define IMAGE_TOKEN_NUM 196
#define EMBED_SIZE 1536

using namespace std;
LLMHandle llmHandle = nullptr;

void exit_handler(int signal)
{
    if (llmHandle != nullptr)
    {
        {
            cout << "程序即将退出" << endl;
            LLMHandle _tmp = llmHandle;
            llmHandle = nullptr;
            rkllm_destroy(_tmp);
        }
    }
    exit(signal);
}

int callback(RKLLMResult *result, void *userdata, LLMCallState state)
{

    if (state == RKLLM_RUN_FINISH)
    {
        printf("\n");
    }
    else if (state == RKLLM_RUN_ERROR)
    {
        printf("\\run error\n");
    }
    else if (state == RKLLM_RUN_NORMAL)
    {
        printf("%s", result->text);
        // for(int i=0; i<result->num; i++)
        // {
        //     printf("%d token_id: %d logprob: %f\n", i, result->tokens[i].id, result->tokens[i].logprob);
        // }
    }
    return 0;
}

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
    if (argc < 7) {
        std::cerr << "Usage: " << argv[0] << " image_path encoder_model_path llm_model_path max_new_tokens max_context_len rknn_core_num\n";
        return -1;
    }

    const char * image_path = argv[1];
    const char * encoder_model_path = argv[2];

    //设置llm参数及初始化
    RKLLMParam param = rkllm_createDefaultParam();
    param.model_path = argv[3];
    param.top_k = 1;
    param.max_new_tokens = std::atoi(argv[4]);
    param.max_context_len = std::atoi(argv[5]);
    param.skip_special_token = true;
    param.img_start = "<|vision_start|>";
    param.img_end = "<|vision_end|>";
    param.img_content = "<|image_pad|>";
    param.extend_param.base_domain_id = 1;
    int ret;

    std::chrono::high_resolution_clock::time_point t_start_us = std::chrono::high_resolution_clock::now();

    ret = rkllm_init(&llmHandle, &param, callback);
    if (ret == 0){
        printf("rkllm init success\n");
    } else {
        printf("rkllm init failed\n");
        exit_handler(-1);
    }
    std::chrono::high_resolution_clock::time_point t_load_end_us = std::chrono::high_resolution_clock::now();

    auto load_time = std::chrono::duration_cast<std::chrono::microseconds>(t_load_end_us - t_start_us);
    printf("%s: LLM Model loaded in %8.2f ms\n", __func__, load_time.count() / 1000.0);

    // imgenc初始化
    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

    t_start_us = std::chrono::high_resolution_clock::now();

    const int core_num = atoi(argv[6]);
    ret = init_imgenc(encoder_model_path, &rknn_app_ctx, core_num);
    if (ret != 0) {
        printf("init_imgenc fail! ret=%d model_path=%s\n", ret, encoder_model_path);
        return -1;
    }
    t_load_end_us = std::chrono::high_resolution_clock::now();

    load_time = std::chrono::duration_cast<std::chrono::microseconds>(t_load_end_us - t_start_us);
    printf("%s: ImgEnc Model loaded in %8.2f ms\n", __func__, load_time.count() / 1000.0);

    // The image is read in BGR format
    cv::Mat img = cv::imread(image_path);
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    // Expand the image into a square and fill it with the specified background color (According the modeling_minicpmv.py)
    cv::Scalar background_color(127.5, 127.5, 127.5);
    cv::Mat square_img = expand2square(img, background_color);

    // Resize the image to 392x392
    cv::Mat resized_img;
    cv::Size new_size(IMAGE_WIDTH, IMAGE_HEIGHT);
    cv::resize(square_img, resized_img, new_size, 0, 0, cv::INTER_LINEAR);

    size_t n_image_tokens = IMAGE_TOKEN_NUM;
    size_t image_embed_len = EMBED_SIZE;
    int rkllm_image_embed_len = n_image_tokens * image_embed_len;
    float img_vec[rkllm_image_embed_len];
    ret = run_imgenc(&rknn_app_ctx, resized_img.data, img_vec);
    if (ret != 0) {
        printf("run_imgenc fail! ret=%d\n", ret);
    }
    
    RKLLMInput rkllm_input;

    // 初始化 infer 参数结构体
    RKLLMInferParam rkllm_infer_params;
    memset(&rkllm_infer_params, 0, sizeof(RKLLMInferParam));
    rkllm_infer_params.mode = RKLLM_INFER_GENERATE;
    
    rkllm_infer_params.keep_history = 0;
    rkllm_set_chat_template(llmHandle, "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n", "<|im_start|>user\n", "<|im_end|>\n<|im_start|>assistant\n");


    vector<string> pre_input;
    pre_input.push_back("<image>What is in the image?");
    pre_input.push_back("<image>这张图片中有什么？");
    cout << "\n**********************可输入以下问题对应序号获取回答/或自定义输入********************\n"
         << endl;
    for (int i = 0; i < (int)pre_input.size(); i++)
    {
        cout << "[" << i << "] " << pre_input[i] << endl;
    }
    cout << "\n*************************************************************************\n"
         << endl;

    while(true) {
        std::string input_str;
        printf("\n");
        printf("user: ");
        std::getline(std::cin, input_str);
        if (input_str == "exit")
        {
            break;
        }
        if (input_str == "clear")
        {
            ret = rkllm_clear_kv_cache(llmHandle, 1, nullptr, nullptr);
            if (ret != 0)
            {
                printf("clear kv cache failed!\n");
            }
            continue;
        }
        for (int i = 0; i < (int)pre_input.size(); i++)
        {
            if (input_str == to_string(i))
            {
                input_str = pre_input[i];
                cout << input_str << endl;
            }
        }
        if (input_str.find("<image>") == std::string::npos) 
        {
            rkllm_input.input_type = RKLLM_INPUT_PROMPT;
            rkllm_input.role = "user";
            rkllm_input.prompt_input = (char*)input_str.c_str();
        } else {
            rkllm_input.input_type = RKLLM_INPUT_MULTIMODAL;
            rkllm_input.role = "user";
            rkllm_input.multimodal_input.prompt = (char*)input_str.c_str();
            rkllm_input.multimodal_input.image_embed = img_vec;
            rkllm_input.multimodal_input.n_image_tokens = n_image_tokens;
            rkllm_input.multimodal_input.n_image = 1;
            rkllm_input.multimodal_input.image_height = IMAGE_HEIGHT;
            rkllm_input.multimodal_input.image_width = IMAGE_WIDTH;
        }
        printf("robot: ");
        rkllm_run(llmHandle, &rkllm_input, &rkllm_infer_params, NULL);
    }

    ret = release_imgenc(&rknn_app_ctx);
    if (ret != 0) {
        printf("release_imgenc fail! ret=%d\n", ret);
    }
    rkllm_destroy(llmHandle);

    return 0;
}