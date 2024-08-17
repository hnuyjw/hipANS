hip_add_library(dietgpu_utils SHARED
  DeviceUtils.cpp
  StackDeviceMemory.cpp
)

target_include_directories(dietgpu_utils PUBLIC
 $<BUILD_INTERFACE:${dietgpu_SOURCE_DIR}>
 "${HIP_INCLUDE_DIRS}"
)
target_link_libraries(dietgpu_utils PUBLIC
 ${HIP_LIBRARIES}
 glog::glog
)
#target_compile_options(dietgpu_utils PRIVATE $<$<COMPILE_LANGUAGE:HIP>:
#  --generate-line-info
  #--device-debug
  #)

  # get_property(GLOBAL_HIP_ARCHITECTURES GLOBAL PROPERTY HIP_ARCHITECTURES)
 # set_target_properties(dietgpu_utils PROPERTI
#   HIP_ARCHITECTURES "${GLOBAL_HIP_ARCHITECTURES}"
# )
