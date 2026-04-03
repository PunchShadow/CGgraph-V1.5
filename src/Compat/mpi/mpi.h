#pragma once

#include <cstring>
#include <unistd.h>

typedef int MPI_Comm;
typedef int MPI_Datatype;

static constexpr int MPI_COMM_WORLD = 0;
static constexpr int MPI_THREAD_MULTIPLE = 3;
static constexpr int MPI_MAX_PROCESSOR_NAME = 256;

static constexpr MPI_Datatype MPI_CHAR = 1;
static constexpr MPI_Datatype MPI_UNSIGNED_CHAR = 2;
static constexpr MPI_Datatype MPI_INT = 3;
static constexpr MPI_Datatype MPI_UNSIGNED = 4;
static constexpr MPI_Datatype MPI_LONG = 5;
static constexpr MPI_Datatype MPI_UNSIGNED_LONG = 6;
static constexpr MPI_Datatype MPI_FLOAT = 7;
static constexpr MPI_Datatype MPI_DOUBLE = 8;

inline int MPI_Init(int* /*argc*/, char*** /*argv*/) { return 0; }

inline int MPI_Init_thread(int* /*argc*/, char*** /*argv*/, int /*required*/, int* provided)
{
    if (provided) *provided = MPI_THREAD_MULTIPLE;
    return 0;
}

inline int MPI_Comm_rank(MPI_Comm /*comm*/, int* rank)
{
    if (rank) *rank = 0;
    return 0;
}

inline int MPI_Comm_size(MPI_Comm /*comm*/, int* size)
{
    if (size) *size = 1;
    return 0;
}

inline int MPI_Get_processor_name(char* name, int* resultlen)
{
    if (!name || !resultlen) return -1;

    if (gethostname(name, MPI_MAX_PROCESSOR_NAME) != 0)
    {
        std::strncpy(name, "localhost", MPI_MAX_PROCESSOR_NAME);
    }
    name[MPI_MAX_PROCESSOR_NAME - 1] = '\0';
    *resultlen = static_cast<int>(std::strlen(name));
    return 0;
}

inline int MPI_Finalized(int* flag)
{
    if (flag) *flag = 0;
    return 0;
}

inline int MPI_Barrier(MPI_Comm /*comm*/) { return 0; }
inline int MPI_Finalize() { return 0; }
