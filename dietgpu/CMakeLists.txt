find_package(Torch REQUIRED)

hip_add_library(dietgpu SHARED
  DietGpu.cpp
)
add_dependencies(dietgpu
  gpu_float_compress
)
target_link_libraries(dietgpu PRIVATE
	gpu_float_compress
  glog::glog
  "${TORCH_LIBRARIES}"
  )
target_include_directories(dietgpu PRIVATE
  $<BUILD_INTERFACE:${dietgpu_SOURCE_DIR}>
  "${TORCH_INCLUDE_DIRS}"
)
cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

hip_add_executable(benchmark_test benchmark_test.cpp)
add_dependencies(benchmark_test
  gpu_float_compress
)
target_link_libraries(benchmark_test 
	gpu_float_compress
	${TORCH_LIBRARIES})
target_include_directories(benchmark_test PRIVATE
  $<BUILD_INTERFACE:${dietgpu_SOURCE_DIR}>
  "${TORCH_INCLUDE_DIRS}"
)

hip_add_executable(benchmark_img benchmark_img.cpp)
add_dependencies(benchmark_img
  gpu_float_compress
)
target_link_libraries(benchmark_img
        gpu_float_compress
        ${TORCH_LIBRARIES})
target_include_directories(benchmark_img PRIVATE
  $<BUILD_INTERFACE:${dietgpu_SOURCE_DIR}>
  "${TORCH_INCLUDE_DIRS}"
)

