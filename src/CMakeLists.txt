cmake_minimum_required(VERSION 3.10)
project(HoleDiameterDetection)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Provide a hint to CMake for where your OpenCV installation is.
# IMPORTANT: CHANGE THIS PATH to match your system.
set(OpenCV_DIR "C:/opencv/build")

# Find the required OpenCV components.
find_package(OpenCV REQUIRED COMPONENTS
    core
    imgproc
    imgcodecs
    highgui
)

# Add the OpenCV header directory to the include path.
include_directories(${OpenCV_INCLUDE_DIRS})

# Add the executable target. Listing headers helps IDEs find them.
add_executable(HoleDiameterDetection
    main.cpp
    HoleParameters.hpp
    vision_apis.h
)

# Link your executable against the OpenCV libraries.
target_link_libraries(HoleDiameterDetection ${OpenCV_LIBS})

# Print status to confirm OpenCV was found correctly.
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")