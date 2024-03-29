cmake_minimum_required(VERSION 2.6.0)

project(ElasticFusion)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}")
set(CMAKE_BUILD_TYPE RELEASE)

if(WIN32)
  macro(CANONIFY_BOOL var)
    if(${var})
      set(${var} TRUE)
    else()
      set(${var} FALSE)
    endif()
  endmacro()
endif()

if(WIN32)
  find_package(JPEG REQUIRED)
endif()

SET(OpenCV_DIR /home/jyli/packages/installation/OpenCV-4.1.1/lib/cmake/opencv4)
find_package( OpenCV 4 REQUIRED )
find_package(LAPACK REQUIRED)
find_package(BLAS REQUIRED)
find_package(ZLIB REQUIRED)
find_package(Pangolin 0.1 REQUIRED)
find_package(CUDA REQUIRED)
# find_package(OpenNI2 REQUIRED)
find_package(efusion REQUIRED)
find_package(SuiteSparse REQUIRED)

set(OPENNI2_INCLUDE_DIR /home/jyli/packages/OpenNI-Linux-x64-2.2/Include)
set(OPENNI2_LIBRARY /home/jyli/packages/OpenNI-Linux-x64-2.2/Tools/libOpenNI2.so)
if(WIN32)
  find_package(RealSense QUIET)
  CANONIFY_BOOL(REALSENSE_FOUND)
  message(STATUS "librealsense found: ${REALSENSE_FOUND}")
  option(WITH_REALSENSE "Build with Intel RealSense support?" ${REALSENSE_FOUND})
endif()

if(WIN32)
  include_directories(${JPEG_INCLUDE_DIR})
endif()

set(EIGEN_INCLUDE_DIRS /home/jyli/projects/ICPORB_VO/Thirdparty/Eigen/eigen3)
set(ICPORBVO_INCLUDE_DIR 
	/home/jjhu/Desktop/hei/cy_code/ICPORB_VO/
	/home/jjhu/Desktop/hei/cy_code/ICPORB_VO/include
	/home/jjhu/Desktop/hei/cy_code/ICPORB_VO/Thirdparty/ORB_SLAM2/include
	/home/jjhu/Desktop/hei/cy_code/ICPORB_VO/Thirdparty/g2o/include
	)
include_directories(${ICPORBVO_INCLUDE_DIR})
include_directories(${ZLIB_INCLUDE_DIR})
include_directories(${EIGEN_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIR})
include_directories(${Pangolin_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${EFUSION_INCLUDE_DIR})
include_directories(${OPENNI2_INCLUDE_DIR})

if(WITH_REALSENSE)
  include_directories(${REALSENSE_INCLUDE_DIR})
  add_definitions(-DWITH_REALSENSE)
  set(EXTRA_LIBS ${EXTRA_LIBS} ${REALSENSE_LIBRARY})
endif()

file(GLOB srcs *.cpp)
file(GLOB tools_srcs Tools/*.cpp)

if(WIN32)
  file(GLOB hdrs *.h)
  file(GLOB tools_hdrs Tools/*.h)
endif()

if(WIN32)
  add_definitions(-Dlinux=0)
  add_definitions(-DWIN32_LEAN_AND_MEAN)
  add_definitions(-DNOMINMAX)
  set(ADDITIONAL_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
  set(EXTRA_WINDOWS_LIBS ${EXTRA_WINDOWS_LIBS} ws2_32 ${JPEG_LIBRARY})
else()
  add_definitions(-Dlinux=1)
endif()

set(CMAKE_CXX_FLAGS ${ADDITIONAL_CMAKE_CXX_FLAGS} "-O3 -msse2 -msse3 -Wall -std=c++11")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -std=c++11")

add_executable(ElasticFusion 
               ${srcs}
               ${tools_srcs}
               ${hdrs}
               ${tools_hdrs}
)

target_link_libraries(ElasticFusion
		${EXTRA_WINDOWS_LIBS}
                      ${ZLIB_LIBRARY}
                      ${Pangolin_LIBRARIES}
                      ${CUDA_LIBRARIES}
                      ${EXTRA_LIBS}
                      ${EFUSION_LIBRARY}
                      ${OPENNI2_LIBRARY}
                      ${SUITESPARSE_LIBRARIES}
                      ${BLAS_LIBRARIES}
                      ${LAPACK_LIBRARIES}
                      ${OpenCV_LIBS}
)


INSTALL(TARGETS ElasticFusion
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
)
