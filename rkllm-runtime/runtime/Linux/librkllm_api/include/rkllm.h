#ifndef _LLM_H_
#define _LLM_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef void* LLMHandle;        /* Handle for an instance of a language model. */

/**
 * @brief Structure for possible states of an inference call.
 * 
 */
typedef enum {
    LLM_RUN_NORMAL = 0,         /* Inference status is normal and inference has not yet finished. */
    LLM_RUN_FINISH = 1,         /* Inference status is normal and inference has finished. */
    LLM_RUN_ERROR = 2           /* Inference status is abnormal. */
} LLMCallState;

/**
 * @brief Structure for setting up parameters for the language model
 * 
 */
typedef struct {
    const char* model_path;     /* Path where the model file is located. */
    int32_t num_npu_core;       /* Number of NPU cores used for model inference. */
    int32_t max_context_len;    /* Maximum size of the context. */
    int32_t max_new_tokens;     /* Maximum number of tokens to generate during model inference. */
    int32_t top_k;              /* The number of highest probability tokens to consider for generation. */
    float top_p;                /* Nucleus sampling: cumulative probability cutoff to use for token selection. */
    float temperature;          /* Hyperparameter to control the randomness of predictions by scaling the logits before applying softmax. */
    float repeat_penalty;       /* Penalty applied to the logits of previously generated tokens, helps prevent repetitive or monotonic text. */
    float frequency_penalty;    /* Penalty for repeating the same word or phrase, reducing the likelihood of repeated content. */
    float presence_penalty;     /* Penalty or reward for introducing new tokens into the generated text. */
    int32_t mirostat;           /* Enables mirostat algorithm, where 0 = off, 1 = use mirostat algorithm, 2 = use mirostat 2.0 algorithm. */
    float mirostat_tau;         /* Target entropy (perplexity) for mirostat algorithm, setting the desired complexity of the generated text. */
    float mirostat_eta;         /* Learning rate for the mirostat algorithm. */
    bool logprobs;              /* Whether to return the log probabilities for each output token along with their token ids. */
    int32_t top_logprobs;       /* The number of top tokens for which to return log probabilities, along with their token ids. */
    bool use_gpu;               /* Flag to indicate whether to use GPU for inference. */
} RKLLMParam;

/**
 * @brief Structure representing a token with its associated log probability.
 * 
 */
typedef struct {
    float logprob;              /* Log probability corresponding to the token ID. */
    int id;                     /* Token ID. */
} Token;

/**
 * @brief Structure to hold the results from the language model inference, including text and token details.
 * 
 */
typedef struct {
    const char* text;           /* Decoded text from the inference output. */
    Token* tokens;              /* Array of Token structures, each containing a log probability and a token ID. */
    int num;                    /* Number of top tokens returned, typically those with the highest probabilities. */
} RKLLMResult;

/**
 * @brief Callback function for handling inference results.
 * 
 * @param result A pointer to an RKLLMResult struct containing the inference results.
 * @param userdata A pointer to user-defined function or null if no user function was provided.
 * @param state The state of the inference process, indicating success, failure, or completion.
 */
typedef void(*LLMResultCallback)(RKLLMResult* result, void* userdata, LLMCallState state);

/**
 * @brief Initializes RKLLMParam with default settings.
 * 
 * @return RKLLMParam An RKLLMParam struct with default values set.
 */
RKLLMParam rkllm_createDefaultParam();

/**
 * @brief Initializes the model with specified parameters.
 * 
 * @param handle Pointer to a handle for the language model, which will be initialized by this function.
 * @param param An RKLLMParam struct containing all the parameters needed for the model.
 * @param callback A function pointer to the callback that handles the results of the inference.
 * @return int Returns 0 on success, or a negative error code on failure.
 */
int rkllm_init(LLMHandle* handle, RKLLMParam param, LLMResultCallback callback);

/**
 * @brief Releases the model resources.
 * 
 * @param handle The handle to the language model to be destroyed.
 * @return int Returns 0 on successful release, or a negative error code if an error occurs.
 */
int rkllm_destroy(LLMHandle handle);

/**
 * @brief Runs model inference on the given prompt.
 * 
 * @param handle The handle to the initialized language model.
 * @param prompt The text prompt on which to perform inference.
 * @param userdata Optional user-defined function that will be passed to the callback.
 * @return int Returns 0 on success, or a negative error code if an error occurs during inference.
 */
int rkllm_run(LLMHandle handle, const char* prompt, void* userdata);

/**
 * @brief Aborts the current inference process.
 * 
 * @param handle The handle to the language model whose inference is to be aborted.
 * @return int Returns 0 if the process is successfully aborted, or a negative error code
 *         if no process was running or if the abort fails.
 */
int rkllm_abort(LLMHandle handle);

#ifdef __cplusplus
} //extern "C"
#endif

#endif