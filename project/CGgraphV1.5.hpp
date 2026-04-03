#pragma once

#include "Basic/Console/console_V3_3.hpp"
#include "Basic/Graph/basic_struct.hpp"
#include "Basic/Graph/checkAlgResult.hpp"
#include "Basic/Graph/graphFileList.hpp"
#include "Basic/Timer/timer_CPJ.hpp"
#include "CG_BFS.hpp"
#include "CG_SSSP.hpp"
#include "flag.hpp"
#include "processorSpeed.hpp"
#include "single.hpp"
#include "subgraphExtraction.hpp"
#include <string>

void Run_CGgraph()
{
    Algorithm_type algorithm = static_cast<Algorithm_type>(FLAGS_algorithm);
    std::string graphSelector = FLAGS_input.empty() ? FLAGS_graphName : FLAGS_input;
    std::string graphTag = CPJ::getGraphTag(graphSelector);
    const int runs = FLAGS_runs;

    double extractSubgraph_time_ms = 0.0;
    double execution_time_ms = 0.0;

    CSR_Result_type csrResult;
    vertex_id_type* old2new = CPJ::getGraphData(csrResult, graphSelector, true, false, false);
    assert_msg(csrResult.vertexNum > 0, "Input graph has zero vertices, graph = [%s]", graphSelector.c_str());

    int64_t root_raw = FLAGS_root;
    if (root_raw < 0 || root_raw >= static_cast<int64_t>(csrResult.vertexNum))
    {
        Msg_warn("Root[%ld] is out of range [0, %zu), fallback to root = 0", root_raw, SCU64(csrResult.vertexNum));
        root_raw = 0;
    }
    int64_t root = static_cast<int64_t>(old2new[static_cast<count_type>(root_raw)]);

    CPJ::SubgraphExt subgraphExt(csrResult, static_cast<Algorithm_type>(FLAGS_algorithm), FLAGS_useDeviceId);
    auto gpuMemType = subgraphExt.getGPUMemType();
    CPJ::ProcessorSpeed speed(csrResult, algorithm, graphTag, gpuMemType);
    double ratio = speed.getRatio();

    if (ratio <= 0.5)
    {
        Msg_warn("GPU is too slow, CPU-ONLY RUN");
        CPJ::Timer executionTimer;
        executionTimer.start();
        if (algorithm == Algorithm_type::BFS)
        {
            CPJ::CPU_BFS cpu_bfs(csrResult);
            cpu_bfs.measureBFS(root);
        }
        else if (algorithm == Algorithm_type::SSSP)
        {
            CPJ::CPU_SSSP cpu_sssp(csrResult, true);
            cpu_sssp.measureSSSP(root);
        }
        else
        {
            assert_msg(false, "Unknow algorithm");
        }
        execution_time_ms = executionTimer.get_time_ms();
    }
    else
    {
        Msg_info("Begin: usedGPUMem = %.2lf (GB)",
                 BYTES_TO_GB(CPJ::MemoryInfo::getMemoryTotal_Device(FLAGS_useDeviceId) - CPJ::MemoryInfo::getMemoryFree_Device(FLAGS_useDeviceId)));
        Host_dataPointer_type hostData;
        Device_dataPointer_type deviceData;
        CPJ::Timer extractSubgraphTimer;
        extractSubgraphTimer.start();
        subgraphExt.extractSubgraph(hostData, deviceData, gpuMemType);
        extractSubgraph_time_ms = extractSubgraphTimer.get_time_ms();
        if (algorithm == Algorithm_type::BFS)
        {
            double total_time{0.0};
            CPJ::CG_BFS cg_bfs(hostData, deviceData, gpuMemType, ratio);
            for (int run_id = 0; run_id < runs; run_id++)
            {
                total_time += cg_bfs.doBFS(root);
            }
            execution_time_ms = total_time;
        }
        else if (algorithm == Algorithm_type::SSSP)
        {
            double total_time{0.0};
            CPJ::CG_SSSP cg_sssp(hostData, deviceData, gpuMemType, ratio);
            for (int run_id = 0; run_id < runs; run_id++)
            {
                total_time += cg_sssp.doSSSP(root);
            }
            execution_time_ms = total_time;
        }
        else
        {
            assert_msg(false, "Unknow algorithm");
        }
    }

    const double total_execution_time_ms = extractSubgraph_time_ms + execution_time_ms;
    Msg_finish("[Complete]: extractSubgraph time: %.3lf (ms), execution time (exclude extractSubgraph): %.3lf (ms), total execution time (include): "
               "%.3lf (ms)",
               extractSubgraph_time_ms, execution_time_ms, total_execution_time_ms);

    if (runs > 0)
    {
        Msg_info("[Extra]: avg execution time per run (exclude extractSubgraph): %.3lf (ms), avg total time per run (include): %.3lf (ms)",
                 execution_time_ms / runs, total_execution_time_ms / runs);
    }
}
