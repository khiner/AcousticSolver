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
    src/Room/Porous.cpp
    src/Room/PorousMaterial.cpp
    src/Room/PorousScene.cpp
    src/Room/RoomGpu.cpp
    src/Room/RoomImplicitScene.cpp
    src/Room/RoomScene.cpp)

set(IMMERSED_SOURCES
    src/Immersed/ImmersedDelta.cpp
    src/Immersed/ImmersedExact.cpp
    src/Immersed/ImmersedGeometry.cpp
    src/Immersed/ImmersedGpu.cpp
    src/Immersed/ImmersedImmittance.cpp
    src/Immersed/ImmersedScene.cpp)

# Kernels are compiled at runtime from these sources (KernelParams.h + Kernels.metal, and
# each other solver's Params header + kernels). macOS caches the compilation, so warm-start
# compiles are ~1 ms (see startup/msl_compile under ACOUSTIC_PROFILE).
set(MSL_DEFINITIONS
    ACOUSTIC_PROJECT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
    ACOUSTIC_MSL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/src/FDTD"
    ACOUSTIC_BEM_MSL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/src/Bem"
    ACOUSTIC_IMMERSED_MSL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/src/Immersed"
    ACOUSTIC_RADIATION_MSL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/src/Radiation"
    ACOUSTIC_ROOM_MSL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/src/Room")

# What the scene runner and the validation harness both need. The include paths, the
# runtime-MSL directories and the frameworks ride along as usage requirements, so the two
# executables cannot drift apart on any of them, and the shared sources compile once.
add_library(AcousticCore STATIC
    src/Bem/Bem.cpp
    src/Bem/BemGpu.cpp
    src/Bem/BemCompressed.cpp
    src/Bem/BemTexture.cpp
    src/Bem/BemIO.cpp
    src/Mesh.cpp
    src/ModalSound.cpp
    src/Oscillator.cpp
    src/Metal/MetalContext.cpp
    ${RADIATION_SOURCES}
    ${ROOM_SOURCES}
    ${IMMERSED_SOURCES}
)
target_include_directories(AcousticCore PUBLIC src src/FDTD src/Immersed src/Metal src/Radiation src/Room)
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

add_executable(PorousTest src/Room/PorousTest.cpp src/Room/PorousSlab.cpp)
target_link_libraries(PorousTest PRIVATE AcousticCore)

add_executable(ImmersedTest src/Immersed/ImmersedTest.cpp)
target_link_libraries(ImmersedTest PRIVATE AcousticCore)

add_executable(DgTest src/Dg/DgTest.cpp src/Dg/DgGpu.cpp)
target_compile_definitions(DgTest PRIVATE ACOUSTIC_DG_MSL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/src/Dg")
target_link_libraries(DgTest PRIVATE AcousticCore)

add_library(DgBasis STATIC src/Dg/DgBasis.cpp)
target_include_directories(DgBasis PUBLIC src)
target_include_directories(DgBasis SYSTEM PUBLIC src/external)
target_compile_options(DgBasis PRIVATE -ffp-contract=off)
add_executable(DgReference src/Dg/DgReference.cpp src/Dg/DgReferenceMain.cpp)
target_include_directories(DgReference SYSTEM PRIVATE src/external)
target_compile_definitions(DgReference PRIVATE ACOUSTIC_PROJECT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}" EIGEN_USE_BLAS)
target_compile_options(DgReference PRIVATE -ffp-contract=off)
target_link_libraries(DgReference PRIVATE DgBasis "-framework Accelerate")

add_library(ElasticCore STATIC src/Elastic/Elastic.cpp src/Elastic/ElasticSparse.cpp src/Elastic/ElasticRun.cpp src/Elastic/ElasticGpu.cpp)
target_link_libraries(ElasticCore PUBLIC DgBasis "-framework Accelerate" "-framework Metal" "-framework Foundation")
target_include_directories(ElasticCore SYSTEM PRIVATE ${metalcpp_SOURCE_DIR})
target_compile_definitions(ElasticCore PRIVATE ACOUSTIC_ELASTIC_MSL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/src/Elastic")
set_source_files_properties(src/Elastic/Elastic.cpp src/Elastic/ElasticGpu.cpp PROPERTIES COMPILE_DEFINITIONS EIGEN_USE_BLAS)
target_compile_options(ElasticCore PRIVATE -ffp-contract=off)
add_executable(ElasticSolve src/Elastic/ElasticMain.cpp)
target_link_libraries(ElasticSolve PRIVATE ElasticCore)
add_executable(ElasticTest src/Elastic/ElasticTest.cpp)
target_link_libraries(ElasticTest PRIVATE ElasticCore)
add_executable(ElasticReference src/Elastic/ElasticReference.cpp)
target_link_libraries(ElasticReference PRIVATE ElasticCore)

add_executable(BemSolve src/Bem/BemMain.cpp)
target_link_libraries(BemSolve PRIVATE AcousticCore)
add_executable(BemTextureSolve src/Bem/BemTextureSolve.cpp)
target_link_libraries(BemTextureSolve PRIVATE AcousticCore)

add_library(PlateCore STATIC src/Plate/Plate.cpp src/Plate/PlateGpu.cpp src/Plate/PlateFft.cpp)
find_path(MPFR_INCLUDE_DIR mpfr.h HINTS /opt/homebrew/include REQUIRED)
find_library(MPFR_LIBRARY mpfr HINTS /opt/homebrew/lib REQUIRED)
find_library(GMP_LIBRARY gmp HINTS /opt/homebrew/lib REQUIRED)
find_path(BOOST_INCLUDE_DIR boost/multiprecision/mpfr.hpp HINTS /opt/homebrew/include REQUIRED)
target_include_directories(PlateCore SYSTEM PUBLIC ${MPFR_INCLUDE_DIR} ${BOOST_INCLUDE_DIR})
target_link_libraries(PlateCore PUBLIC AcousticCore ${MPFR_LIBRARY} ${GMP_LIBRARY})
target_compile_definitions(PlateCore PRIVATE ACOUSTIC_PLATE_MSL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/src/Plate")

add_executable(PlateSolve src/Plate/PlateMain.cpp)
target_link_libraries(PlateSolve PRIVATE PlateCore)
add_executable(PlateTest src/Plate/PlateTest.cpp)
target_link_libraries(PlateTest PRIVATE PlateCore)

add_library(StringCore STATIC src/String/String.cpp src/String/StringGpu.cpp)
target_link_libraries(StringCore PUBLIC AcousticCore)
target_compile_definitions(StringCore PRIVATE ACOUSTIC_STRING_MSL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/src/String")
target_compile_options(StringCore PRIVATE -ffp-contract=off)
add_executable(StringSolve src/String/StringMain.cpp)
target_link_libraries(StringSolve PRIVATE StringCore)
add_executable(StringTest src/String/StringTest.cpp)
target_link_libraries(StringTest PRIVATE StringCore)
add_executable(StringPaper src/String/StringPaper.cpp)
target_link_libraries(StringPaper PRIVATE StringCore)

add_library(TubeCore STATIC src/Tube/Tube.cpp)
target_include_directories(TubeCore PUBLIC src)
target_include_directories(TubeCore SYSTEM PUBLIC src/external)
target_compile_options(TubeCore PRIVATE -ffp-contract=off)
add_executable(TubeSolve src/Tube/TubeMain.cpp)
target_link_libraries(TubeSolve PRIVATE TubeCore)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/ness/src/PlateEmbedded.cpp")
    set(NESS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/ness/src")
    # Guard exhausted circular masks without modifying the pinned submodule.
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${NESS_DIR}/PlateEmbedded.cpp")
    file(READ "${NESS_DIR}/PlateEmbedded.cpp" NESS_PLATE_SOURCE)
    string(REPLACE "if (TruePhi[I] == l)" "if (I < TruePhiSize && TruePhi[I] == l)" NESS_PLATE_SOURCE "${NESS_PLATE_SOURCE}")
    string(REPLACE "if (TruePhi[j] == i)" "if (j < TruePhiSize && TruePhi[j] == i)" NESS_PLATE_SOURCE "${NESS_PLATE_SOURCE}")
    string(REPLACE "double K, sig0, c;" "double K, sig0, c = 0;" NESS_PLATE_SOURCE "${NESS_PLATE_SOURCE}")
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/ness")
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/ness/PlateEmbedded.cpp" "${NESS_PLATE_SOURCE}")
    execute_process(COMMAND git -C "${NESS_DIR}" rev-parse HEAD OUTPUT_VARIABLE NESS_COMMIT OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
    add_library(NessReference STATIC
        ${NESS_DIR}/Airbox.cpp ${NESS_DIR}/AirboxIndexed.cpp ${CMAKE_CURRENT_BINARY_DIR}/ness/PlateEmbedded.cpp
        ${NESS_DIR}/Embedding.cpp ${NESS_DIR}/Component.cpp ${NESS_DIR}/Component2D.cpp ${NESS_DIR}/Component3D.cpp
        ${NESS_DIR}/Connection.cpp ${NESS_DIR}/GlobalSettings.cpp ${NESS_DIR}/SettingsManager.cpp
        ${NESS_DIR}/Input.cpp ${NESS_DIR}/InputSample.cpp ${NESS_DIR}/InputStrike.cpp
        ${NESS_DIR}/Logger.cpp ${NESS_DIR}/Material.cpp ${NESS_DIR}/Profiler.cpp
        ${NESS_DIR}/Task.cpp ${NESS_DIR}/TaskWholeComponent.cpp ${NESS_DIR}/TaskWholeConnection.cpp
        ${NESS_DIR}/MathUtil.cpp ${NESS_DIR}/matgen.c
        ${NESS_DIR}/lib/csrmatrix.c ${NESS_DIR}/lib/arralloc.c ${NESS_DIR}/lib/pcg.c ${NESS_DIR}/lib/banded.c)
    target_include_directories(NessReference SYSTEM PUBLIC ${NESS_DIR} ${NESS_DIR}/lib)
    target_compile_options(NessReference PRIVATE -w -ffp-contract=off)
    add_library(CoupledCore STATIC src/Coupled/CoupledReference.cpp src/Coupled/CoupledBoundary.cpp src/Coupled/CoupledGpu.cpp)
    target_link_libraries(CoupledCore PUBLIC AcousticCore PRIVATE NessReference)
    target_compile_definitions(CoupledCore PRIVATE ACOUSTIC_NESS_COMMIT="${NESS_COMMIT}")
    target_compile_options(CoupledCore PRIVATE -ffp-contract=off)
    add_executable(CoupledSolve src/Coupled/CoupledMain.cpp)
    target_link_libraries(CoupledSolve PRIVATE CoupledCore)
    add_executable(CoupledBoundaryTest src/Coupled/CoupledBoundaryTest.cpp)
    target_link_libraries(CoupledBoundaryTest PRIVATE CoupledCore)
endif()
