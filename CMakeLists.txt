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

# The SonicRadiation solver [Jin et al. 2025]: CQM weight generation, the TDBEM tables, and
# the Metal stepper. Shared by the scene runner and the validation harness.
set(RADIATION_SOURCES
    src/Radiation/Cqm.cpp
    src/Radiation/RadiationMesh.cpp
    src/Radiation/Tdbem.cpp
    src/Radiation/RadiationTables.cpp
    src/Radiation/RadiationGpu.cpp
    src/Radiation/RadiationScene.cpp)

# The room-acoustics solver [Bilbao, Hamilton, Botts and Savioja 2016; PFFDTD-equivalent]:
# the scene loader and the Metal stepper. Shared by the scene runner and its harness.
set(ROOM_SOURCES
    src/Room/RoomGpu.cpp
    src/Room/RoomImplicitScene.cpp
    src/Room/RoomScene.cpp)

# Kernels are compiled at runtime from these sources (KernelParams.h + Kernels.metal, and
# each other solver's Params header + kernels). macOS caches the compilation, so warm-start
# compiles are ~1 ms (see startup/msl_compile under ACOUSTIC_PROFILE).
set(MSL_DEFINITIONS
    ACOUSTIC_MSL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/src/FDTD"
    ACOUSTIC_RADIATION_MSL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/src/Radiation"
    ACOUSTIC_ROOM_MSL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/src/Room")

# What the scene runner and the validation harness both need. The include paths, the
# runtime-MSL directories and the frameworks ride along as usage requirements, so the two
# executables cannot drift apart on any of them, and the shared sources compile once.
add_library(AcousticCore STATIC
    src/Mesh.cpp
    src/ModalSound.cpp
    src/Oscillator.cpp
    src/Metal/MetalContext.cpp
    ${RADIATION_SOURCES}
    ${ROOM_SOURCES}
)
target_include_directories(AcousticCore PUBLIC src src/FDTD src/Metal src/Radiation src/Room)
target_include_directories(AcousticCore SYSTEM PUBLIC src/external ${metalcpp_SOURCE_DIR})
target_compile_definitions(AcousticCore PUBLIC ${MSL_DEFINITIONS})
target_link_libraries(AcousticCore PUBLIC "-framework Metal" "-framework Foundation" "-framework Accelerate")

add_executable(${PROJECT_NAME}
    main.cpp
    src/BubbleUtils.cpp
    src/FluidSound.cpp
    src/Integrators.cpp
    src/WaveBlender.cpp
    ${SHADER_SOURCES}
)
target_link_libraries(${PROJECT_NAME} PRIVATE AcousticCore)

# The SonicRadiation solver's validation harness (see src/Radiation/RadiationTest.cpp for
# the test ladder).
add_executable(RadiationTest src/Radiation/RadiationTest.cpp)
target_link_libraries(RadiationTest PRIVATE AcousticCore)

# The room solver's validation harness (see src/Room/RoomTest.cpp for the test ladder).
add_executable(RoomTest src/Room/RoomTest.cpp)
target_link_libraries(RoomTest PRIVATE AcousticCore)
