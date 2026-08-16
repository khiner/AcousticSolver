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

# Both dependencies are consumed header-only. SOURCE_SUBDIR points at a
# nonexistent directory so FetchContent_MakeAvailable skips their own CMake.
FetchContent_Declare(eigen
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
    SOURCE_SUBDIR cmake_disabled)
# metal-cpp has no release tags (Apple distributes per-SDK zips), so pin the mirror
# commit corresponding to the "macOS 27, iOS 27" drop.
FetchContent_Declare(metalcpp
    URL https://github.com/apple/metal-cpp/archive/27c4382b7151d55a51692cdcb27aaa98752240de.tar.gz
    URL_HASH SHA256=01b6e76f10c0f5b6eb44be22a3d276cd9922da62a51d4d345b70c5177d720231
    SOURCE_SUBDIR cmake_disabled)
FetchContent_MakeAvailable(eigen metalcpp)

file(GLOB SHADER_SOURCES Source/Shaders/*.cpp)
set(VENDORED_SOURCES
    Source/ModalSound/Source/ModalSound.cpp
    Source/FluidSound/Source/FluidSound.cpp
    Source/FluidSound/Source/Oscillator.cpp
    Source/FluidSound/Source/BubbleUtils.cpp
    Source/FluidSound/Source/Integrators/Integrators.cpp
)
# Vendored reference code is kept byte-identical upstream. Ignore its warnings.
set_source_files_properties(${VENDORED_SOURCES} PROPERTIES COMPILE_FLAGS "-w")

add_executable(${PROJECT_NAME}
    main.cpp
    Source/Mesh.cpp
    Source/WaveBlender.cpp
    Source/Metal/MetalContext.cpp
    ${SHADER_SOURCES}
    ${VENDORED_SOURCES}
)

target_include_directories(${PROJECT_NAME} PRIVATE Source Source/FDTD Source/Metal)
target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE
    Source/ThirdParty
    Source/ModalSound/Source Source/FluidSound/Source
    ${eigen_SOURCE_DIR}
    ${metalcpp_SOURCE_DIR}
)

# Kernels are compiled at runtime from these sources (KernelParams.h + Kernels.metal).
target_compile_definitions(${PROJECT_NAME} PRIVATE
    ACOUSTIC_MSL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/Source/FDTD")

target_link_libraries(${PROJECT_NAME} PRIVATE "-framework Metal" "-framework Foundation")
