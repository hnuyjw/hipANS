set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})

hip_add_library(gpu_float_compress SHARED
  GpuFloatCompress.cpp
  GpuFloatDecompress.cpp
  GpuFloatInfo.cpp
)
add_dependencies(gpu_float_compress
  gpu_ans
  dietgpu_utils
)

target_include_directories(gpu_float_compress PUBLIC
 $<BUILD_INTERFACE:${dietgpu_SOURCE_DIR}>
)
target_link_libraries(gpu_float_compress PUBLIC
  gpu_ans
  dietgpu_utils
)
target_link_libraries(gpu_float_compress PRIVATE
 glog::glog
)
target_compile_options(gpu_float_compress PRIVATE $<$<COMPILE_LANGUAGE:HIP>:
  --generate-line-info
  #--device-debug
>)

enable_testing()
include(GoogleTest)

hip_add_executable(float_test FloatTest.cpp)
target_link_libraries(float_test
  gpu_float_compress
 gtest_main
)

hip_add_executable(fpcompress_img fpcompress_img.cpp)
target_link_libraries(fpcompress_img
  gpu_float_compress
 gtest_main
)

hip_add_executable(fp16compress_img fp16compress_img.cpp)
target_link_libraries(fp16compress_img
  gpu_float_compress
 gtest_main
)

hip_add_executable(fpwholecompress_img fpwholecompress_img.cpp)
target_link_libraries(fpwholecompress_img
  gpu_float_compress
 gtest_main
)

gtest_discover_tests(float_test)


# get_property(GLOBAL_HIP_ARCHITECTURES GLOBAL PROPERTY HIP_ARCHITECTURES)
# set_target_properties(gpu_float_compress float_test PROPERTIES
#   HIP_ARCHITECTURES "${GLOBAL_HIP_ARCHITECTURES}"
# )
