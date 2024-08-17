hip_add_library(gpu_ans SHARED
  GpuANSDecode.cpp
  GpuANSEncode.cpp
  GpuANSInfo.cpp
)
add_dependencies(gpu_ans
  dietgpu_utils
)
target_include_directories(gpu_ans PUBLIC
 $<BUILD_INTERFACE:${dietgpu_SOURCE_DIR}>
)
target_link_libraries(gpu_ans PUBLIC
  dietgpu_utils
)
target_link_libraries(gpu_ans PRIVATE
 glog::glog
)
target_compile_options(gpu_ans PRIVATE $<$<COMPILE_LANGUAGE:HIP>:
  --generate-line-info
  #--device-debug
>)


enable_testing()
include(GoogleTest)

hip_add_executable(ans_test ANSTest.cpp)
target_link_libraries(ans_test
 gpu_ans
 gtest_main
)
gtest_discover_tests(ans_test)

hip_add_executable(ans_statistics_test ANSStatisticsTest.cpp)
target_link_libraries(ans_statistics_test
  gpu_ans
  gtest_main
  dietgpu_utils
)
gtest_discover_tests(ans_statistics_test)

hip_add_executable(batch_prefix_sum_test BatchPrefixSumTest.cpp)
target_link_libraries(batch_prefix_sum_test
  gpu_ans
  gtest_main
)
gtest_discover_tests(batch_prefix_sum_test)

hip_add_executable(compress_img compress_img.cpp)
target_link_libraries(compress_img
	dietgpu_utils
	gpu_ans
        ${TORCH_LIBRARIES})
target_include_directories(compress_img PRIVATE
  $<BUILD_INTERFACE:${dietgpu_SOURCE_DIR}>
  "${TORCH_INCLUDE_DIRS}"
)
# get_property(GLOBAL_HIP_ARCHITECTURES GLOBAL PROPERTY HIP_ARCHITECTURES)
# set_target_properties(gpu_ans ans_test ans_statistics_test batch_prefix_sum_test
#   PROPERTIES HIP_ARCHITECTURES "${GLOBAL_HIP_ARCHITECTURES}"
# )
