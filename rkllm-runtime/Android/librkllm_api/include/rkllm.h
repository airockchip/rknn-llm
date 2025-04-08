#ifndef _RKLLM_H_
#define _RKLLM_H_

#ifdef __cplusplus
extern "C" {
#endif

#define CPU0 (1 << 0)  // 0x01
#define CPU1 (1 << 1)  // 0x02
#define CPU2 (1 << 2)  // 0x04
#define CPU3 (1 << 3)  // 0x08
#define CPU4 (1 << 4)  // 0x10
#define CPU5 (1 << 5)  // 0x20
#define CPU6 (1 << 6)  // 0x40
#define CPU7 (1 << 7)  // 0x80

/**
 * @typedef LLMHandle
 * @brief A handle used to manage and interact with the large language model.
 */
typedef void* LLMHandle;

/**
 * @enum LLMCallState
 * @brief Describes the possible states of an LLM call.
 */
typedef enum {
    RKLLM_RUN_NORMAL  = 0, /**< The LLM call is in a normal running state. */
    RKLLM_RUN_WAITING = 1, /**< The LLM call is waiting for complete UTF-8 encoded character. */
    RKLLM_RUN_FINISH  = 2, /**< The LLM call has finished execution. */
    RKLLM_RUN_ERROR   = 3, /**< An error occurred during the LLM call. */
} LLMCallState;

/**
 * @enum RKLLMInputType
 * @brief Defines the types of inputs that can be fed into the LLM.
 */
typedef enum {
    RKLLM_INPUT_PROMPT      = 0, /**< Input is a text prompt. */
    RKLLM_INPUT_TOKEN       = 1, /**< Input is a sequence of tokens. */
    RKLLM_INPUT_EMBED       = 2, /**< Input is an embedding vector. */
    RKLLM_INPUT_MULTIMODAL  = 3, /**< Input is multimodal (e.g., text and image). */
} RKLLMInputType;

/**
 * @enum RKLLMInferMode
 * @brief Specifies the inference modes of the LLM.
 */
typedef enum {
    RKLLM_INFER_GENERATE                    = 0, /**< The LLM generates text based on input. */
    RKLLM_INFER_GET_LAST_HIDDEN_LAYER       = 1, /**< The LLM retrieves the last hidden layer for further processing. */
    RKLLM_INFER_GET_LOGITS                  = 2, /**< The LLM retrieves logits for further processing. */
} RKLLMInferMode;

/**
 * @struct RKLLMExtendParam
 * @brief The extend parameters for configuring an LLM instance.
 */
typedef struct {
    int32_t      base_domain_id;        /**< base_domain_id */
    int8_t       embed_flash;           /**< Indicates whether to query word embedding vectors from flash memory (1) or not (0). */
    int8_t       enabled_cpus_num;      /**< Number of CPUs enabled for inference. */
    uint32_t     enabled_cpus_mask;     /**< Bitmask indicating which CPUs to enable for inference. */
    uint8_t      reserved[106];         /**< reserved */
} RKLLMExtendParam;

/**
 * @struct RKLLMParam
 * @brief Defines the parameters for configuring an LLM instance.
 */
typedef struct {
    const char* model_path;         /**< Path to the model file. */
    int32_t max_context_len;        /**< Maximum number of tokens in the context window. */
    int32_t max_new_tokens;         /**< Maximum number of new tokens to generate. */
    int32_t top_k;                  /**< Top-K sampling parameter for token generation. */
    int32_t n_keep;                 /** number of kv cache to keep at the beginning when shifting context window */
    float top_p;                    /**< Top-P (nucleus) sampling parameter. */
    float temperature;              /**< Sampling temperature, affecting the randomness of token selection. */
    float repeat_penalty;           /**< Penalty for repeating tokens in generation. */
    float frequency_penalty;        /**< Penalizes frequent tokens during generation. */
    float presence_penalty;         /**< Penalizes tokens based on their presence in the input. */
    int32_t mirostat;               /**< Mirostat sampling strategy flag (0 to disable). */
    float mirostat_tau;             /**< Tau parameter for Mirostat sampling. */
    float mirostat_eta;             /**< Eta parameter for Mirostat sampling. */
    bool skip_special_token;        /**< Whether to skip special tokens during generation. */
    bool is_async;                  /**< Whether to run inference asynchronously. */
    const char* img_start;          /**< Starting position of an image in multimodal input. */
    const char* img_end;            /**< Ending position of an image in multimodal input. */
    const char* img_content;        /**< Pointer to the image content. */
    RKLLMExtendParam extend_param; /**< Extend parameters. */
} RKLLMParam;

/**
 * @struct RKLLMLoraAdapter
 * @brief Defines parameters for a Lora adapter used in model fine-tuning.
 */
typedef struct {
    const char* lora_adapter_path; /**< Path to the Lora adapter file. */
    const char* lora_adapter_name; /**< Name of the Lora adapter. */
    float scale;                   /**< Scaling factor for applying the Lora adapter. */
} RKLLMLoraAdapter;

/**
 * @struct RKLLMEmbedInput
 * @brief Represents an embedding input to the LLM.
 */
typedef struct {
    float* embed;      /**< Pointer to the embedding vector (of size n_tokens * n_embed). */
    size_t n_tokens;   /**< Number of tokens represented in the embedding. */
} RKLLMEmbedInput;

/**
 * @struct RKLLMTokenInput
 * @brief Represents token input to the LLM.
 */
typedef struct {
    int32_t* input_ids; /**< Array of token IDs. */
    size_t n_tokens;    /**< Number of tokens in the input. */
} RKLLMTokenInput;

/**
 * @struct RKLLMMultiModelInput
 * @brief Represents multimodal input (e.g., text and image).
 */
typedef struct {
    char* prompt;           /**< Text prompt input. */
    float* image_embed;     /**< Embedding of the image (of size n_image_tokens * n_image_embed). */
    size_t n_image_tokens;  /**< Number of image tokens. */
} RKLLMMultiModelInput;

/**
 * @struct RKLLMInput
 * @brief Represents different types of input to the LLM via a union.
 */
typedef struct {
    RKLLMInputType input_type; /**< Specifies the type of input provided (e.g., prompt, token, embed, multimodal). */
    union {
        const char* prompt_input;               /**< Text prompt input if input_type is RKLLM_INPUT_PROMPT. */
        RKLLMEmbedInput embed_input;            /**< Embedding input if input_type is RKLLM_INPUT_EMBED. */
        RKLLMTokenInput token_input;            /**< Token input if input_type is RKLLM_INPUT_TOKEN. */
        RKLLMMultiModelInput multimodal_input;  /**< Multimodal input if input_type is RKLLM_INPUT_MULTIMODAL. */
    };
} RKLLMInput;

/**
 * @struct RKLLMLoraParam
 * @brief Structure defining parameters for Lora adapters.
 */
typedef struct {
    const char* lora_adapter_name; /**< Name of the Lora adapter. */
} RKLLMLoraParam;

/**
 * @struct RKLLMPromptCacheParam
 * @brief Structure to define parameters for caching prompts.
 */
typedef struct {
    int save_prompt_cache;          /**< Flag to indicate whether to save the prompt cache (0 = don't save, 1 = save). */
    const char* prompt_cache_path;  /**< Path to the prompt cache file. */
} RKLLMPromptCacheParam;

/**
 * @struct RKLLMInferParam
 * @brief Structure for defining parameters during inference.
 */
typedef struct {
    RKLLMInferMode mode;                        /**< Inference mode (e.g., generate or get last hidden layer). */
    RKLLMLoraParam* lora_params;                /**< Pointer to Lora adapter parameters. */
    RKLLMPromptCacheParam* prompt_cache_params; /**< Pointer to prompt cache parameters. */
    int keep_history;                           /**Flag to determine history retention (1: keep history, 0: discard history).*/
} RKLLMInferParam;

/**
 * @struct RKLLMResultLastHiddenLayer
 * @brief Structure to hold the hidden states from the last layer.
 */
typedef struct {
    const float* hidden_states; /**< Pointer to the hidden states (of size num_tokens * embd_size). */
    int embd_size;              /**< Size of the embedding vector. */
    int num_tokens;             /**< Number of tokens for which hidden states are stored. */
} RKLLMResultLastHiddenLayer;

/**
 * @struct RKLLMResultLogits
 * @brief Structure to hold the logits.
 */
typedef struct {
    const float* logits;        /**< Pointer to the logits (of size num_tokens * vocab_size). */
    int vocab_size;             /**< Size of the vocab. */
    int num_tokens;             /**< Number of tokens for which logits are stored. */
} RKLLMResultLogits;

/**
 * @struct RKLLMResult
 * @brief Structure to represent the result of LLM inference.
 */
typedef struct {
    const char* text;                        /**< Generated text result. */
    int32_t token_id;                        /**< ID of the generated token. */
    RKLLMResultLastHiddenLayer last_hidden_layer; /**< Hidden states of the last layer (if requested). */
    RKLLMResultLogits logits; 
} RKLLMResult;

/**
 * @typedef LLMResultCallback
 * @brief Callback function to handle LLM results.
 * @param result Pointer to the LLM result.
 * @param userdata Pointer to user data for the callback.
 * @param state State of the LLM call (e.g., finished, error).
 */
typedef void(*LLMResultCallback)(RKLLMResult* result, void* userdata, LLMCallState state);

/**
 * @brief Creates a default RKLLMParam structure with preset values.
 * @return A default RKLLMParam structure.
 */
RKLLMParam rkllm_createDefaultParam();

/**
 * @brief Initializes the LLM with the given parameters.
 * @param handle Pointer to the LLM handle.
 * @param param Configuration parameters for the LLM.
 * @param callback Callback function to handle LLM results.
 * @return Status code (0 for success, non-zero for failure).
 */
int rkllm_init(LLMHandle* handle, RKLLMParam* param, LLMResultCallback callback);

/**
 * @brief Loads a Lora adapter into the LLM.
 * @param handle LLM handle.
 * @param lora_adapter Pointer to the Lora adapter structure.
 * @return Status code (0 for success, non-zero for failure).
 */
int rkllm_load_lora(LLMHandle handle, RKLLMLoraAdapter* lora_adapter);

/**
 * @brief Loads a prompt cache from a file.
 * @param handle LLM handle.
 * @param prompt_cache_path Path to the prompt cache file.
 * @return Status code (0 for success, non-zero for failure).
 */
int rkllm_load_prompt_cache(LLMHandle handle, const char* prompt_cache_path);

/**
 * @brief Releases the prompt cache from memory.
 * @param handle LLM handle.
 * @return Status code (0 for success, non-zero for failure).
 */
int rkllm_release_prompt_cache(LLMHandle handle);

/**
 * @brief Destroys the LLM instance and releases resources.
 * @param handle LLM handle.
 * @return Status code (0 for success, non-zero for failure).
 */
int rkllm_destroy(LLMHandle handle);

/**
 * @brief Runs an LLM inference task synchronously.
 * @param handle LLM handle.
 * @param rkllm_input Input data for the LLM.
 * @param rkllm_infer_params Parameters for the inference task.
 * @param userdata Pointer to user data for the callback.
 * @return Status code (0 for success, non-zero for failure).
 */
int rkllm_run(LLMHandle handle, RKLLMInput* rkllm_input, RKLLMInferParam* rkllm_infer_params, void* userdata);

/**
 * @brief Runs an LLM inference task asynchronously.
 * @param handle LLM handle.
 * @param rkllm_input Input data for the LLM.
 * @param rkllm_infer_params Parameters for the inference task.
 * @param userdata Pointer to user data for the callback.
 * @return Status code (0 for success, non-zero for failure).
 */
int rkllm_run_async(LLMHandle handle, RKLLMInput* rkllm_input, RKLLMInferParam* rkllm_infer_params, void* userdata);

/**
 * @brief Aborts an ongoing LLM task.
 * @param handle LLM handle.
 * @return Status code (0 for success, non-zero for failure).
 */
int rkllm_abort(LLMHandle handle);

/**
 * @brief Checks if an LLM task is currently running.
 * @param handle LLM handle.
 * @return Status code (0 if a task is running, non-zero for otherwise).
 */
int rkllm_is_running(LLMHandle handle);

/**  
 * @brief Clear the key-value cache for a given LLM handle.  
 * @param handle LLM handle.  
 * @param keep_system_prompt Flag indicating whether to retain the system prompt in the cache (1 to retain, 0 to clear).  
 * @return Status code (0 if cache was cleared successfully, non-zero for otherwise).  
 */  
int rkllm_clear_kv_cache(LLMHandle handle, int keep_system_prompt);

/**  
 * @brief Sets the chat template for the LLM, including system prompt, prefix, and postfix.  
 *  
 * This function allows you to customize the chat template by providing a system prompt, a prompt prefix, and a prompt postfix.  
 * The system prompt is typically used to define the behavior or context of the language model,  
 * while the prefix and postfix are used to format the user input and output respectively.  
 *  
 * @param handle LLM handle.  
 * @param system_prompt The system prompt that defines the context or behavior of the language model.  
 * @param prompt_prefix The prefix added before the user input in the chat.  
 * @param prompt_postfix The postfix added after the user input in the chat.  
 *  
 * @return Status code (0 if the template was set successfully, non-zero for errors).  
 */
int rkllm_set_chat_template(LLMHandle handle, const char* system_prompt, const char* prompt_prefix, const char* prompt_postfix);

#ifdef __cplusplus
}
#endif

#endif
