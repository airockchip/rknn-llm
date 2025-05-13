#include "rknn_api.h"

#ifndef _RKNN_IMAGE_ENC_H_
#define _RKNN_IMAGE_ENC_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    rknn_context rknn_ctx;
    rknn_input_output_num io_num;
    rknn_tensor_attr* input_attrs;
    rknn_tensor_attr* output_attrs;
    int model_channel;
    int model_width;
    int model_height;
} rknn_app_context_t;

int init_imgenc(const char* model_path, rknn_app_context_t* app_ctx, const int core_num);

int release_imgenc(rknn_app_context_t* app_ctx);

int run_imgenc(rknn_app_context_t* app_ctx, void* img_data, float* out_result);

#ifdef __cplusplus
}
#endif

#endif //_RKNN_IMAGE_ENC_H_