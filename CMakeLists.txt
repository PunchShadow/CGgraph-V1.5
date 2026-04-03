cmake_minimum_required(VERSION 3.5.0)
project(CGgraph-V1.5 VERSION 0.1.0 LANGUAGES C CXX)

set(MY_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR} CACHE PATH "Path to CMakeLists parent directory" FORCE)
SET(MY_CXX_VERSION -std=c++20)
#SET(CMAKE_BUILD_TYPE Release) # Debug Release MinSizeRel RelWithDebInfo
SET(MY_BUILD_TYPE Release) # Debug Release
SET(MIN_CUDA_VERSION 12.0)
SET(COMPILE_COMMANDS  ON) # "Whether to export compile commands"
SET(USE_SIMD  ON) # "Whether to use SIMD instructions"
SET(USE_BOOST OFF) # find jemalloc
SET(USE_JEMALLOC OFF) # find jemalloc
SET(COMPAT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/src/Compat)

if (COMPILE_COMMANDS)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

if (USE_SIMD)
    message(STATUS "[Enable SIMD]: ON")
    add_definitions(-DUSE_BMISS_STTNI_INTERSECT)
    add_definitions(-DUSE_SIMD_SORT)
    SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -Xcompiler; -mavx2; -Xcompiler; -march=native)
else()
    message(STATUS "[Enable SIMD]: OFF")
endif ()

# ------------------------------------------------------------------------------
# Build Type
# ------------------------------------------------------------------------------
if (MY_BUILD_TYPE STREQUAL Debug)
    message(STATUS "[My Build Type]: Debug")
elseif(MY_BUILD_TYPE STREQUAL Release)
    message(STATUS "[My Build Type]: Release")
else()
    message(FATAL_ERROR "[My Build Type]: Unknow")
endif ()

# ------------------------------------------------------------------------------
# OpenMP
# ------------------------------------------------------------------------------
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    message(STATUS "[Find OpenMP]: YES")
    message(STATUS "\t- The OpenMP Lib: ${OpenMP_C_LIBRARIES}")
    SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -Xcompiler; -fopenmp; -Xcompiler; -pthread; -Xcompiler; -lgomp)
else(OPENMP_FOUND)
	message(FATAL_ERROR "[Find OpenMP]: NO")
endif(OPENMP_FOUND)



# ------------------------------------------------------------------------------
# CUDA, 12.0 at least
# ------------------------------------------------------------------------------
find_package(CUDA REQUIRED)
if(${CUDA_FOUND})
    message(STATUS "[Find CUDA]: YES")
    if (CUDA_VERSION VERSION_LESS ${MIN_CUDA_VERSION})
        message(FATAL_ERROR "\t- The CUDA Version Need >= ${MIN_CUDA_VERSION}, Current Version: ${MIN_CUDA_VERSION}")
    else()
        message(STATUS "\t- The CUDA Version: ${MIN_CUDA_VERSION}")
        message(STATUS "\t- The CUDA TOOLKIT INCLUDE: ${CUDA_TOOLKIT_INCLUDE}")
        include_directories(${CUDA_TOOLKIT_INCLUDE})
        set(CUDA_LIBS ${CUDA_TOOLKIT_TARGET_DIR}/lib64/stubs/libcuda.so
                    ${CUDA_TOOLKIT_TARGET_DIR}/lib64/libnvToolsExt.so
                    ${CUDA_TOOLKIT_TARGET_DIR}/lib64/libcudart.so)
        include(FindCUDA/select_compute_arch)
        CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1) 
        string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2) 
        string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
        string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}") 
        SET(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
        message(STATUS "\t- The CUDA Arthitecture = ${CUDA_ARCH_LIST}")
        set_property(GLOBAL PROPERTY CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}")
        list(GET CUDA_ARCH_LIST 0 CUDA_ARCH_CODE)
        SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -arch=sm_${CUDA_ARCH_CODE})
        SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -Wno-deprecated-gpu-targets)

        if (MY_BUILD_TYPE STREQUAL Debug)
            set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-g;-G;-O0;-lineinfo;-Xcompiler;-ggdb;-Xcompiler;-Wall;${MY_CXX_VERSION};--expt-extended-lambda)
        else()
            set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3;${MY_CXX_VERSION};--expt-extended-lambda)
        endif ()
        set(CUDA_PROPAGATE_HOST_FLAGS OFF)
        set(CUDA_SEPARABLE_COMPILATION OFF) #ON
        message(STATUS "\t- The CUDA Host Compiler: ${CUDA_HOST_COMPILER}")
    endif()
else()
    message(FATAL_ERROR "[Find CUDA]: NO")
endif()


find_package(MPI QUIET)
if(MPI_FOUND)
    message(STATUS "[Find MPI]: YES")
    if (MPI_C_INCLUDE_DIRS)
        include_directories(${MPI_C_INCLUDE_DIRS})
    elseif(MPI_INCLUDE_PATH)
        include_directories(${MPI_INCLUDE_PATH}) # legacy CMake var
    endif()
    message(STATUS "\t- The MPI Lib: ${MPI_C_LIBRARIES}")
else()
    message(WARNING "[Find MPI]: NO, using local single-process MPI compatibility header")
    include_directories(BEFORE ${COMPAT_ROOT}/mpi)
endif()


# ------------------------------------------------------------------------------
# gflags
# ------------------------------------------------------------------------------
find_path(GFLAGS_INCLUDE_DIR gflags/gflags.h)
find_library(GFLAGS_LIBRARY NAMES gflags libgflags)
if(GFLAGS_INCLUDE_DIR AND GFLAGS_LIBRARY)
    message(STATUS "[Find Gflags]: YES")
    message(STATUS "\t- The Gflags Include: ${GFLAGS_INCLUDE_DIR}")
    message(STATUS "\t- The Gflags Lib: ${GFLAGS_LIBRARY}")
    include_directories(${GFLAGS_INCLUDE_DIR})
else()
    message(WARNING "[Find Gflags]: NO, using local compatibility parser")
    include_directories(BEFORE ${COMPAT_ROOT}/gflags)
endif()

# ------------------------------------------------------------------------------
# TBB
# ------------------------------------------------------------------------------
find_library(TBB_LIBRARY NAMES tbb PATHS /usr/local/tbb-2019_U8/lib /usr/lib/x86_64-linux-gnu)
if(TBB_LIBRARY)
    message(STATUS "[Find TBB]: YES")
    message(STATUS "\t- The TBB Lib: ${TBB_LIBRARY}")
else()
    message(WARNING "[Find TBB]: NO, continue without linking tbb explicitly")
endif()

# ------------------------------------------------------------------------------
# jemalloc
# ------------------------------------------------------------------------------
if (USE_JEMALLOC)
    include("cmake/FindJemalloc.cmake")
    if (JEMALLOC_FOUND)
        message(STATUS "[Find JEMALLOC]: YES")
        message(STATUS "\t- The JEMALLOC Include: ${JEMALLOC_INCLUDE_DIRS}")
        add_definitions(-DUSE_JEMALLOC)
        include_directories(SYSTEM ${JEMALLOC_INCLUDE_DIRS})
    else ()
        message(STATUS "[Find JEMALLOC]: NO, Skip...")
    endif ()
endif ()

#**********************************************************************************
# include_directories
#**********************************************************************************
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/project/)


#**********************************************************************************
# cuda_add_library
# 这样会打包成库, SHARED是动态库(多个程序共用), STATIC是静态库(每个程序单独使用)
#**********************************************************************************
cuda_add_library(MY_CPP_CU SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/project/flag.cpp
)


SET(Program CGgraphV1.5)
CUDA_ADD_EXECUTABLE(${Program} CGgraphV1.5.cu)

set(CGGRAPH_LINK_LIBS MY_CPP_CU ${CUDA_LIBS})

find_library(LIBDL_LIBRARY NAMES dl)
if(LIBDL_LIBRARY)
    list(APPEND CGGRAPH_LINK_LIBS ${LIBDL_LIBRARY})
endif()

find_library(LIBZ_LIBRARY NAMES z)
if(LIBZ_LIBRARY)
    list(APPEND CGGRAPH_LINK_LIBS ${LIBZ_LIBRARY})
endif()

find_library(LIBNUMA_LIBRARY NAMES numa)
if(LIBNUMA_LIBRARY)
    list(APPEND CGGRAPH_LINK_LIBS ${LIBNUMA_LIBRARY})
endif()

find_library(LIBNVML_LIBRARY NAMES nvidia-ml)
if(LIBNVML_LIBRARY)
    list(APPEND CGGRAPH_LINK_LIBS ${LIBNVML_LIBRARY})
endif()

if(OPENMP_FOUND)
    list(APPEND CGGRAPH_LINK_LIBS ${OpenMP_CXX_LIBRARIES})
endif()

if(MPI_FOUND)
    list(APPEND CGGRAPH_LINK_LIBS ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES})
endif()

if(GFLAGS_LIBRARY)
    list(APPEND CGGRAPH_LINK_LIBS ${GFLAGS_LIBRARY})
endif()

if(TBB_LIBRARY)
    list(APPEND CGGRAPH_LINK_LIBS ${TBB_LIBRARY})
endif()

if (USE_JEMALLOC)
    target_link_libraries(${Program} /usr/local/lib/libjemalloc.so.2)
endif()

target_link_libraries(${Program} ${CGGRAPH_LINK_LIBS})
