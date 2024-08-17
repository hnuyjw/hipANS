# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_COMPILER /usr/bin/hipcc)
set(CMAKE_CXX_LINKER /usr/bin/hipcc)
if(NOT DEFINED HIP_PATH)
    if(NOT DEFINED ENV{HIP_PATH})
        set(HIP_PATH "/opt/rocm/hip" CACHE PATH "Path to which HIP has been installed")
    else()
        set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
    endif()
endif()
set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})
project(dietgpu LANGUAGES HIP CXX VERSION 1.0)
 set(CMAKE_CXX_STANDARD 17)
 # set(CMAKE_C_STANDARD 11)

# include(CheckLanguage)
# check_language(HIP)

# if(NOT DEFINED CMAKE_HIP_STANDARD)
#     set(CMAKE_HIP_STANDARD 5.6)
#     set(CMAKE_HIP_STANDARD_REQUIRED ON)
# endif()


# Set compiler and linker
# set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})
# set(CMAKE_CXX_LINKER   ${HIP_HIPCC_EXECUTABLE})

if(NOT DEFINED HIP_PATH)
    if(NOT DEFINED ENV{HIP_PATH})
        set(HIP_PATH "/opt/rocm/hip" CACHE PATH "Path to which HIP has been installed")
    else()
        set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
    endif()
endif()
set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})
find_package(HIP)

# if(${CMAKE_VERSION} VERSION_LESS_EQUAL "3.13.4")
#     cuda_select_nvcc_arch_flags(ARCH_FLAGS "Auto")
#     message("ARCH_FLAGS = ${ARCH_FLAGS}")
#     string(REPLACE "-gencode;" "--generate-code=" ARCH_FLAGS "${ARCH_FLAGS}")
#     string(APPEND CMAKE_HIP_FLAGS "${ARCH_FLAGS}")
# else()
#     include(FindCUDA/select_compute_arch)
#     CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
#     string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
#     string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
#     string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
#     set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
#     set_property(GLOBAL PROPERTY CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}")
# endif()

# Set default build type.
# if(NOT CMAKE_BUILD_TYPE)
#   message(STATUS "Setting build type to 'RelWithDebInfo' as none was specified.")
#   set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING
#     "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
#     FORCE
#   )
# endif()
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -g -O1")
#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(third_party/glog)
add_subdirectory(third_party/googletest)

add_subdirectory(dietgpu)
add_subdirectory(dietgpu/utils)
add_subdirectory(dietgpu/ans)
add_subdirectory(dietgpu/float)
