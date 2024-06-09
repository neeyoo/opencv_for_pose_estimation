cmake_minimum_required(VERSION 3.0.0)
project(vision_for_assemblyrecon VERSION 0.1.0 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Include OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})

# Add the executable
add_executable(CameraCapture src/photo_shoot.cpp)
add_executable(CameraCalib src/camera_calib.cpp)
add_executable(ArucoDetect src/aruco_detect.cpp)
add_executable(CalibChArUco src/calib_charuco.cpp)

# Link the OpenCV libraries
target_link_libraries(CameraCapture ${OpenCV_LIBS})
target_link_libraries(CameraCalib ${OpenCV_LIBS})
target_link_libraries(ArucoDetect ${OpenCV_LIBS})
target_link_libraries(CalibChArUco ${OpenCV_LIBS})