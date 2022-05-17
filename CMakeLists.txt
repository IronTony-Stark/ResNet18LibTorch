# Visual Studio 2019 toolchain with amd64 architecture

cmake_minimum_required (VERSION 3.19)

project(ResNet18Intel CXX)

set(CMAKE_CXX_STANDARD 17)

# Just setting Torch_DIR is not enough for some reason..
set(CMAKE_PREFIX_PATH "../libtorch/${CMAKE_BUILD_TYPE}")

# [WARN] On Windows Debug and Release versions of LibTorch are NOT ABI compatible
set(Torch_DIR "../libtorch/${CMAKE_BUILD_TYPE}")
set(OpenCV_DIR "../opencv/build")

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(ResNet18Intel
    ./src/main.cpp
    ./src/utils.cpp
)

target_include_directories(ResNet18Intel PUBLIC
    ./include
    ${TORCH_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(ResNet18Intel PRIVATE
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
)

file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
add_custom_command(TARGET ResNet18Intel
                   POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_if_different
                   ${TORCH_DLLS}
                   $<TARGET_FILE_DIR:ResNet18Intel>)

install(TARGETS ResNet18Intel
        DESTINATION .
)

install(FILES ${TORCH_DLLS}
        DESTINATION .
)
