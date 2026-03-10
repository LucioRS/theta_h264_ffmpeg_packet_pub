cmake_minimum_required(VERSION 3.8)
project(theta_h264_ffmpeg_packet_pub)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(ffmpeg_image_transport_msgs REQUIRED)
find_package(PkgConfig REQUIRED)

pkg_check_modules(GSTREAMER REQUIRED IMPORTED_TARGET
  gstreamer-1.0
  gstreamer-app-1.0
  gstreamer-video-1.0
)

pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
  libavcodec
  libavutil
)

add_executable(theta_h264_ffmpeg_packet_node
  src/theta_h264_ffmpeg_packet_node.cpp
)

target_compile_features(theta_h264_ffmpeg_packet_node PUBLIC cxx_std_17)

ament_target_dependencies(theta_h264_ffmpeg_packet_node
  rclcpp
  builtin_interfaces
  ffmpeg_image_transport_msgs
)

target_link_libraries(theta_h264_ffmpeg_packet_node
  PkgConfig::GSTREAMER
  PkgConfig::FFMPEG
)

install(TARGETS
  theta_h264_ffmpeg_packet_node
  DESTINATION lib/${PROJECT_NAME}
)

install(FILES
  package.xml
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
