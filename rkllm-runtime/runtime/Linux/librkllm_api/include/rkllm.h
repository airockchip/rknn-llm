#ifndef _LLM_H_
#define _LLM_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef void* LLMHandle;

typedef enum {
    LLM_RUN_NORMAL = 0,  /*推理状态正常，推理尚未结束*/
    LLM_RUN_FINISH = 1,  /*推理状态正常，推理结束*/
    LLM_RUN_ERROR = 2    /*推理状态异常*/
} LLMCallState;

typedef struct {
    const char* modelPath;          /*模型文件的存放路径*/
    const char* target_platform;    /*模型运行的硬件平台*/
    int32_t num_npu_core;           /*模型推理时使用的 NPU 核心数量*/
    int32_t max_context_len;        /*设置提示上下文的大小*/
    int32_t max_new_tokens;         /*用于设置模型推理时生成 Token 的数量上限*/

    int32_t top_k;                  
    float top_p;
    float temperature;
    float repeat_penalty;
    float frequency_penalty;
    float presence_penalty;
    int32_t mirostat;
    float mirostat_tau;
    float mirostat_eta;

} RKLLMParam;

typedef void(*LLMResultCallback)(const char* result, void* userdata, LLMCallState state);

RKLLMParam rkllm_createDefaultParam();                                              /*初始化RKLLMParam并设置默认参数*/

int rkllm_init(LLMHandle* handle, RKLLMParam param, LLMResultCallback callback);    /*模型初始化*/

int rkllm_run(LLMHandle handle, const char* prompt, void* userdata);                /*模型推理*/

int rkllm_destroy(LLMHandle handle);                                                /*模型释放*/

#ifdef __cplusplus
}
#endif

#endif