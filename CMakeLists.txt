cmake_minimum_required(VERSION 3.10)

#set( CMAKE_BUILD_TYPE "Release")

project(CV_pictrack)


if(CMAKE_HOST_WIN32)
    message(STATUS "System is WIN32, path need to set manually.")
    message(STATUS "CMAKE_SOURCE_DIR = ${CMAKE_SOURCE_DIR}")
    # set(OpenCV_DIR "E:/Lib/opencv3/build/x64/vc14/lib")
    # set(OpenCV_DIR "E:/Lib/opencv3/build/x64/vc15/lib")
    # set(OpenCV_DIR "E:/Lib/opencv4/opencv/build/x64/vc16/lib")
    set(OpenCV_DIR "E:/Lib/opencv4/opencv/build/x64/mingw")
    set(Eigen3_DIR  "${CMAKE_SOURCE_DIR}/Thirdparty/Eigen3/share/eigen3/cmake")
    set(g2o_DIR  "E:/Lib/g2o/G2O-Windows/g2o/lib/cmake/g2o")
endif()

set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O2 -Wall -fexec-charset=GBK")
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11" )
#set(OpenCV_STATIC ON)

if(CMAKE_HOST_WIN32)
    find_package(OpenCV 4 REQUIRED)
else()
    find_package(OpenCV 3 REQUIRED)
endif()

message(STATUS "OpenCV library status:")
message(STATUS "  version: ${OpenCV_VERSION}")
message(STATUS "  libraries: ${OpenCV_LIBS}")
message(STATUS "  libraries path: ${}")
message(STATUS "  include path: ${OpenCV_INCLUDE_DIRS}")

find_package(Eigen3 3.1.0 REQUIRED)
message(STATUS "Eigen3 library status:")
message(STATUS "  version: ${Eigen3_VERSION}")
message(STATUS "  libraries: ${Eigen3_LIBS}")
message(STATUS "  libraries path: ${}")
message(STATUS "  include path: ${Eigen3_INCLUDE_DIRS}")

find_package(g2o REQUIRED)
message(STATUS "g2o library status:")
message(STATUS "  version: ${g2o_VERSION}")
message(STATUS "  libraries: ${g2o_LIBS}")
message(STATUS "  libraries path: ${}")
message(STATUS "  include path: ${g2o_INCLUDE_DIRS}")


include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( include )

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)
add_executable(${PROJECT_NAME} 
src/main.cpp 
src/fileload.cpp 
src/cv_method.cpp 
src/ORBextractor.cc
# src/ORBmatcher.cc
)
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )

