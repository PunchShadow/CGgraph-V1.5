# Repository Guidelines

## Project Structure & Module Organization
- `CGgraphV1.5.cu` is the executable entry point; CLI flags are declared in `project/flag.hpp` and defined in `project/flag.cpp`.
- `project/` contains algorithm-level orchestration and runtime logic (`CG_BFS.hpp`, `CG_SSSP.hpp`, `CGgraphV1.5.hpp`, scheduling, result checks).
- `src/Basic/` provides reusable infrastructure (graph I/O, CUDA helpers, memory utilities, timers, logging, sorting, threading).
- `src/Comm/MPI/` contains MPI environment setup.
- `fig/` holds documentation images; `README_EN.md` and `README.md` describe usage and data assumptions.

## Build, Test, and Development Commands
- Update local dependency paths in `CMakeLists.txt` before configuring (notably gflags and TBB absolute paths).
- Configure: `cmake -S . -B build`
- Build: `cmake --build build -j`
- Show runtime flags: `./build/CGgraphV1.5 --help`
- Example run (BFS): `./build/CGgraphV1.5 -graphName friendster -algorithm 0 -gpuMemory 0 -root 100 -runs 5 -useDeviceId 0`

## Coding Style & Naming Conventions
- Language target is C++20/CUDA (`.hpp`, `.cuh`, `.cu`).
- Format using `.clang-format` (LLVM-based): 4-space indentation, no tabs, 150-column limit, pointer alignment on the left.
- Match local naming patterns: macros/constants in `UPPER_SNAKE_CASE`, classes/types in `PascalCase`, functions and variables following surrounding file conventions (`snake_case` is common).
- Keep includes ordered and case-sensitive per formatter settings.

## Testing Guidelines
- There is no dedicated `tests/` or `ctest` suite in this version; verification is command-driven.
- For algorithm changes, run at least one BFS and one SSSP case on a known graph and capture command, runtime, and correctness notes.
- If baseline binaries exist under `/data/webgraph/checkResult/`, use checks in `project/checkResult.hpp` and `src/Basic/Graph/checkAlgResult.hpp`.

## Commit & Pull Request Guidelines
- Current history favors short, single-line commit subjects (for example, `v1.5`, `Update README.md`).
- Prefer imperative, scoped messages such as `SSSP: reduce frontier merge overhead`.
- PRs should include intent, touched modules, exact run command(s), hardware/software environment (CPU/GPU/CUDA), and before/after correctness or performance evidence.

## Configuration Tips
- Graph and validation file locations are hard-coded in several headers (for example `/data/webgraph/...`); keep local paths configurable in your branch.
- Do not commit machine-specific absolute paths unless they are intentionally documented defaults.
