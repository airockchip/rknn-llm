cmake_minimum_required(VERSION 3.10)
project(rkllm_demo)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SOURCE_FILES_1 src/llm_demo.cpp)
add_executable(llm_demo ${SOURCE_FILES_1})

set(SOURCE_FILES_2 src/multimodel_demo.cpp)
add_executable(multimodel_demo ${SOURCE_FILES_2})

set(RKLLM_API_PATH "${CMAKE_SOURCE_DIR}/../../runtime/${CMAKE_SYSTEM_NAME}/librkllm_api")
include_directories(${RKLLM_API_PATH}/include)
if(CMAKE_SYSTEM_NAME STREQUAL "Android")
    set(RKLLM_RT_LIB ${RKLLM_API_PATH}/${CMAKE_ANDROID_ARCH_ABI}/librkllmrt.so)
    find_package(OpenMP REQUIRED)
    target_link_libraries(llm_demo  ${RKLLM_RT_LIB} log OpenMP::OpenMP_CXX)
    target_link_libraries(multimodel_demo  ${RKLLM_RT_LIB} log OpenMP::OpenMP_CXX)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(RKLLM_RT_LIB ${RKLLM_API_PATH}/aarch64/librkllmrt.so)
    target_link_libraries(llm_demo  ${RKLLM_RT_LIB})
    target_link_libraries(multimodel_demo  ${RKLLM_RT_LIB})
endif()


