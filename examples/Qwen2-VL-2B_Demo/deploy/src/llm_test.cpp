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

#include <string.h>
#include <unistd.h>
#include <string>
#include <fstream>
#include <iostream>
#include <csignal>
#include <vector>
#include <chrono>

#include "rkllm.h"

#define PROMPT_TEXT_PREFIX "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
#define PROMPT_TEXT_POSTFIX "<|im_end|>\n<|im_start|>assistant\n"

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

void callback(RKLLMResult *result, void *userdata, LLMCallState state)
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
}

int main(int argc, char **argv)
{
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " model_path max_new_tokens max_context_len\n";
        return 1;
    }

    signal(SIGINT, exit_handler);
    printf("rkllm init start\n");

    //设置参数及初始化
    RKLLMParam param = rkllm_createDefaultParam();
    param.model_path = argv[1];
    param.top_k = 1;
    param.max_new_tokens = std::atoi(argv[2]);
    param.max_context_len = std::atoi(argv[3]);
    param.skip_special_token = true;

    std::chrono::high_resolution_clock::time_point t_start_us = std::chrono::high_resolution_clock::now();

    int ret = rkllm_init(&llmHandle, &param, callback);
    if (ret == 0){
        printf("rkllm init success\n");
    } else {
        printf("rkllm init failed\n");
        exit_handler(-1);
    }
    std::chrono::high_resolution_clock::time_point t_load_end_us = std::chrono::high_resolution_clock::now();

    auto load_time = std::chrono::duration_cast<std::chrono::microseconds>(t_load_end_us - t_start_us);
    printf("%s: Model loaded in %8.2f ms\n", __func__, load_time.count() / 1000.0);

    int seq_len = 256;
    int32_t * test_tokens = (int32_t *)malloc(sizeof(int32_t) * seq_len);
    memset(test_tokens, 0, sizeof(int32_t) * seq_len);
    printf("\n The llm_test is used to test the performance of the model. Now the seq_len is set to %d by RKLLM_INPUT_TOKEN. \n", seq_len);
    
    RKLLMInput rkllm_input;

    // 初始化 infer 参数结构体
    RKLLMInferParam rkllm_infer_params;
    memset(&rkllm_infer_params, 0, sizeof(RKLLMInferParam));
    rkllm_infer_params.mode = RKLLM_INFER_GENERATE;

    while (true)
    {
        std::string input_str;
        printf("\n");
        printf("user: ");
        std::getline(std::cin, input_str);
        if (input_str == "exit")
        {
            break;
        }

        rkllm_input.input_type = RKLLM_INPUT_TOKEN;
        rkllm_input.token_input.input_ids = test_tokens;
        rkllm_input.token_input.n_tokens = seq_len;
        printf("robot: ");
        rkllm_run(llmHandle, &rkllm_input, &rkllm_infer_params, NULL);
    }

    rkllm_destroy(llmHandle);

    return 0;
}