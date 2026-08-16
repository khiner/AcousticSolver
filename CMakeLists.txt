cmake_minimum_required(VERSION 3.24)

project(AcousticSolver LANGUAGES C CXX)

if(NOT APPLE)
    message(FATAL_ERROR "AcousticSolver is a Metal port and requires macOS")
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # Emit build/compile_commands.json for clangd and clang-tidy.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -stdlib=libc++")

include(FetchContent)

# The only fetched dependency, and header-only (Eigen is vendored, see script/VendorEigen).
# metal-cpp has no release tags, so pin the mirror commit for the "macOS 27, iOS 27" drop.
# SOURCE_SUBDIR points at a nonexistent directory so MakeAvailable skips its own CMake.
FetchContent_Declare(metalcpp
    URL https://github.com/apple/metal-cpp/archive/27c4382b7151d55a51692cdcb27aaa98752240de.tar.gz
    URL_HASH SHA256=01b6e76f10c0f5b6eb44be22a3d276cd9922da62a51d4d345b70c5177d720231
    SOURCE_SUBDIR cmake_disabled)
FetchContent_MakeAvailable(metalcpp)

file(GLOB SHADER_SOURCES src/Shaders/*.cpp)

add_executable(${PROJECT_NAME}
    main.cpp
    src/BubbleUtils.cpp
    src/FluidSound.cpp
    src/Integrators.cpp
    src/Mesh.cpp
    src/ModalSound.cpp
    src/Oscillator.cpp
    src/WaveBlender.cpp
    src/Metal/MetalContext.cpp
    ${SHADER_SOURCES}
)

target_include_directories(${PROJECT_NAME} PRIVATE src src/FDTD src/Metal)
target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE
    src/external
    ${metalcpp_SOURCE_DIR}
)

# Kernels are compiled at runtime from these sources (KernelParams.h + Kernels.metal).
# macOS caches the compilation, so warm-start compiles are ~1 ms (see startup/msl_compile
# under ACOUSTIC_PROFILE).
target_compile_definitions(${PROJECT_NAME} PRIVATE
    ACOUSTIC_MSL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/src/FDTD")

target_link_libraries(${PROJECT_NAME} PRIVATE "-framework Metal" "-framework Foundation" "-framework Accelerate")
