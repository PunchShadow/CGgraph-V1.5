#pragma once

#include "Basic/Console/console_V3_3.hpp"
#include "Basic/Graph/basic_def.hpp"
#include "Basic/Graph/basic_struct.hpp"
#include "Basic/Graph/getGraphCSR.hpp"
#include "Basic/Other/fileSystem_CPJ.hpp"
#include "Basic/Type/data_type.hpp"
#include "flag.hpp"
#include "reorderGraph.hpp"
#include <algorithm>
#include <assert.h>
#include <cctype>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace CPJ {

namespace detail {

struct SubwayEdge {
    vertex_id_type src;
    vertex_id_type dst;
    edge_data_type weight;
};

struct SubwayOutEdge {
    uint32_t end;
};

struct SubwayOutEdgeWeighted {
    uint32_t end;
    uint32_t weight;
};

struct SubwayOutEdge64 {
    uint64_t end;
};

struct SubwayOutEdgeWeighted64 {
    uint64_t end;
    uint64_t weight;
};

inline bool isCommentOrEmptyLine(const std::string& line)
{
    for (char c : line)
    {
        if (!std::isspace(static_cast<unsigned char>(c)))
        {
            return c == '#' || c == '%';
        }
    }
    return true;
}

inline std::string toLower(std::string str)
{
    std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return str;
}

inline std::string getLowerExtension(const std::string& path)
{
    return toLower(CPJ::FS::getFileExtension(path));
}

inline bool isSubwayGraphExtension(const std::string& extension)
{
    return extension == ".el" || extension == ".wel" || extension == ".txt" || extension == ".snap" || extension == ".bcsr" || extension == ".bwcsr" ||
           extension == ".bcsr64" || extension == ".bwcsr64";
}

inline std::string makeSafeGraphTag(std::string graphNameOrPath)
{
    const std::string extension = getLowerExtension(graphNameOrPath);
    if (isSubwayGraphExtension(extension))
    {
        std::string stem = CPJ::FS::getFileStem(graphNameOrPath);
        if (!stem.empty()) graphNameOrPath = stem;
    }

    if (graphNameOrPath.empty()) graphNameOrPath = "graph";
    for (char& c : graphNameOrPath)
    {
        if (!std::isalnum(static_cast<unsigned char>(c)) && c != '-' && c != '_') c = '_';
    }
    return graphNameOrPath;
}

inline vertex_id_type* buildIdentityOld2New(const count_type vertexNum)
{
    vertex_id_type* old2new = new vertex_id_type[vertexNum];
    omp_par_for(count_type vertex_id = 0; vertex_id < vertexNum; vertex_id++) { old2new[vertex_id] = vertex_id; }
    return old2new;
}

inline void buildCSRFromSubwayEdgeList(CSR_Result_type& csrResult, const std::vector<SubwayEdge>& edges, const bool needWeight)
{
    uint64_t maxVertex = 0;
    for (const auto& edge : edges)
    {
        maxVertex = std::max(maxVertex, static_cast<uint64_t>(std::max(edge.src, edge.dst)));
    }

    const count_type vertexNum = static_cast<count_type>(maxVertex + 1);
    const countl_type edgeNum = static_cast<countl_type>(edges.size());

    csrResult.vertexNum = vertexNum;
    csrResult.edgeNum = edgeNum;
    csrResult.csr_offset = new countl_type[vertexNum + 1];
    memset(csrResult.csr_offset, 0, sizeof(countl_type) * (vertexNum + 1));
    csrResult.csr_dest = new vertex_id_type[edgeNum];
    memset(csrResult.csr_dest, 0, sizeof(vertex_id_type) * edgeNum);
    if (needWeight)
    {
        csrResult.csr_weight = new edge_data_type[edgeNum];
        memset(csrResult.csr_weight, 0, sizeof(edge_data_type) * edgeNum);
    }

    std::vector<countl_type> degree(vertexNum, 0);
    for (const auto& edge : edges) degree[edge.src]++;

    countl_type offset = 0;
    for (count_type vertex_id = 0; vertex_id < vertexNum; vertex_id++)
    {
        csrResult.csr_offset[vertex_id] = offset;
        offset += degree[vertex_id];
    }
    csrResult.csr_offset[vertexNum] = edgeNum;

    std::vector<countl_type> outDegreeCounter(vertexNum, 0);
    for (const auto& edge : edges)
    {
        countl_type location = csrResult.csr_offset[edge.src] + outDegreeCounter[edge.src];
        csrResult.csr_dest[location] = edge.dst;
        if (needWeight) csrResult.csr_weight[location] = edge.weight;
        outDegreeCounter[edge.src]++;
    }
}

inline void loadSubwayTextGraph(CSR_Result_type& csrResult, const std::string& graphPath, const bool needWeight)
{
    std::ifstream infile(graphPath);
    assert_msg(infile.is_open(), "Open input graph file failed, graphPath = [%s]", graphPath.c_str());

    std::vector<SubwayEdge> edge_vec;
    edge_vec.reserve(1024 * 1024);

    std::stringstream ss;
    std::string line;
    while (std::getline(infile, line))
    {
        if (isCommentOrEmptyLine(line)) continue;

        ss.str("");
        ss.clear();
        ss << line;

        uint64_t src_u64{0};
        uint64_t dst_u64{0};
        if (!(ss >> src_u64 >> dst_u64)) continue;

        uint64_t weight_u64{1};
        if (needWeight && !(ss >> weight_u64)) weight_u64 = 1;

        assert_msg(src_u64 <= std::numeric_limits<vertex_id_type>::max(), "src overflow, src = %zu", SCU64(src_u64));
        assert_msg(dst_u64 <= std::numeric_limits<vertex_id_type>::max(), "dst overflow, dst = %zu", SCU64(dst_u64));
        assert_msg(weight_u64 <= std::numeric_limits<edge_data_type>::max(), "weight overflow, weight = %zu", SCU64(weight_u64));

        SubwayEdge edge;
        edge.src = static_cast<vertex_id_type>(src_u64);
        edge.dst = static_cast<vertex_id_type>(dst_u64);
        edge.weight = static_cast<edge_data_type>(weight_u64);
        edge_vec.push_back(edge);
    }

    if (edge_vec.empty())
    {
        Msg_warn("No valid edge loaded from [%s], build a 1-vertex empty graph", graphPath.c_str());
        csrResult.vertexNum = 1;
        csrResult.edgeNum = 0;
        csrResult.csr_offset = new countl_type[2];
        csrResult.csr_offset[0] = 0;
        csrResult.csr_offset[1] = 0;
        csrResult.csr_dest = nullptr;
        if (needWeight) csrResult.csr_weight = nullptr;
        return;
    }

    buildCSRFromSubwayEdgeList(csrResult, edge_vec, needWeight);
}

inline void loadSubwayBinaryGraph(CSR_Result_type& csrResult, const std::string& graphPath, const bool needWeight, const bool fileHasWeight)
{
    std::ifstream infile(graphPath, std::ios::binary);
    assert_msg(infile.is_open(), "Open input graph file failed, graphPath = [%s]", graphPath.c_str());

    uint32_t num_nodes_u32{0};
    uint32_t num_edges_u32{0};
    infile.read(reinterpret_cast<char*>(&num_nodes_u32), sizeof(uint32_t));
    infile.read(reinterpret_cast<char*>(&num_edges_u32), sizeof(uint32_t));
    assert_msg(!infile.fail(), "Read binary graph header failed, graphPath = [%s]", graphPath.c_str());

    assert_msg(num_nodes_u32 <= std::numeric_limits<count_type>::max(), "vertexNum overflow, num_nodes = %u", num_nodes_u32);
    assert_msg(num_edges_u32 <= std::numeric_limits<countl_type>::max(), "edgeNum overflow, num_edges = %u", num_edges_u32);

    const count_type vertexNum = static_cast<count_type>(num_nodes_u32);
    const countl_type edgeNum = static_cast<countl_type>(num_edges_u32);
    csrResult.vertexNum = vertexNum;
    csrResult.edgeNum = edgeNum;
    csrResult.csr_offset = new countl_type[vertexNum + 1];
    memset(csrResult.csr_offset, 0, sizeof(countl_type) * (vertexNum + 1));

    std::vector<uint32_t> nodePointer_u32(vertexNum, 0);
    if (!nodePointer_u32.empty())
    {
        infile.read(reinterpret_cast<char*>(nodePointer_u32.data()), sizeof(uint32_t) * vertexNum);
        assert_msg(!infile.fail(), "Read nodePointer failed, graphPath = [%s]", graphPath.c_str());
    }

    for (count_type vertex_id = 0; vertex_id < vertexNum; vertex_id++)
    {
        csrResult.csr_offset[vertex_id] = static_cast<countl_type>(nodePointer_u32[vertex_id]);
    }
    csrResult.csr_offset[vertexNum] = edgeNum;

    csrResult.csr_dest = new vertex_id_type[edgeNum];
    memset(csrResult.csr_dest, 0, sizeof(vertex_id_type) * edgeNum);
    if (needWeight)
    {
        csrResult.csr_weight = new edge_data_type[edgeNum];
        memset(csrResult.csr_weight, 0, sizeof(edge_data_type) * edgeNum);
    }

    if (fileHasWeight)
    {
        std::vector<SubwayOutEdgeWeighted> edgeList(edgeNum);
        if (!edgeList.empty())
        {
            infile.read(reinterpret_cast<char*>(edgeList.data()), sizeof(SubwayOutEdgeWeighted) * edgeNum);
            assert_msg(!infile.fail(), "Read weighted edge list failed, graphPath = [%s]", graphPath.c_str());
            for (countl_type edge_id = 0; edge_id < edgeNum; edge_id++)
            {
                csrResult.csr_dest[edge_id] = static_cast<vertex_id_type>(edgeList[edge_id].end);
                if (needWeight) csrResult.csr_weight[edge_id] = static_cast<edge_data_type>(edgeList[edge_id].weight);
            }
        }
    }
    else
    {
        std::vector<SubwayOutEdge> edgeList(edgeNum);
        if (!edgeList.empty())
        {
            infile.read(reinterpret_cast<char*>(edgeList.data()), sizeof(SubwayOutEdge) * edgeNum);
            assert_msg(!infile.fail(), "Read edge list failed, graphPath = [%s]", graphPath.c_str());
            for (countl_type edge_id = 0; edge_id < edgeNum; edge_id++)
            {
                csrResult.csr_dest[edge_id] = static_cast<vertex_id_type>(edgeList[edge_id].end);
                if (needWeight) csrResult.csr_weight[edge_id] = static_cast<edge_data_type>(1);
            }
        }
    }
}

inline void loadSubwayBinary64Graph(CSR_Result_type& csrResult, const std::string& graphPath, const bool needWeight, const bool fileHasWeight)
{
    std::ifstream infile(graphPath, std::ios::binary);
    assert_msg(infile.is_open(), "Open input graph file failed, graphPath = [%s]", graphPath.c_str());

    uint64_t num_nodes_u64{0};
    uint64_t num_edges_u64{0};
    infile.read(reinterpret_cast<char*>(&num_nodes_u64), sizeof(uint64_t));
    infile.read(reinterpret_cast<char*>(&num_edges_u64), sizeof(uint64_t));
    assert_msg(!infile.fail(), "Read binary64 graph header failed, graphPath = [%s]", graphPath.c_str());

    const count_type vertexNum = static_cast<count_type>(num_nodes_u64);
    const countl_type edgeNum = static_cast<countl_type>(num_edges_u64);
    csrResult.vertexNum = vertexNum;
    csrResult.edgeNum = edgeNum;
    csrResult.csr_offset = new countl_type[vertexNum + 1];

    // Read 64-bit node pointers directly into csr_offset
    static_assert(sizeof(countl_type) == sizeof(uint64_t), "countl_type must be 64-bit for .bcsr64/.bwcsr64 support");
    if (vertexNum > 0)
    {
        infile.read(reinterpret_cast<char*>(csrResult.csr_offset), sizeof(uint64_t) * vertexNum);
        assert_msg(!infile.fail(), "Read 64-bit nodePointer failed, graphPath = [%s]", graphPath.c_str());
    }
    csrResult.csr_offset[vertexNum] = edgeNum;

    csrResult.csr_dest = new vertex_id_type[edgeNum];
    if (needWeight)
    {
        csrResult.csr_weight = new edge_data_type[edgeNum];
    }

    if (fileHasWeight)
    {
        // Read in chunks: each edge is {uint64_t dest, uint64_t weight}
        constexpr countl_type CHUNK = 1ULL << 20; // 1M edges per chunk
        std::vector<SubwayOutEdgeWeighted64> buf(std::min(edgeNum, CHUNK));
        for (countl_type pos = 0; pos < edgeNum; pos += CHUNK)
        {
            countl_type cnt = std::min(CHUNK, edgeNum - pos);
            infile.read(reinterpret_cast<char*>(buf.data()), sizeof(SubwayOutEdgeWeighted64) * cnt);
            assert_msg(!infile.fail(), "Read 64-bit weighted edge list failed at offset %zu, graphPath = [%s]", SCU64(pos), graphPath.c_str());
            for (countl_type i = 0; i < cnt; i++)
            {
                csrResult.csr_dest[pos + i] = static_cast<vertex_id_type>(buf[i].end);
                if (needWeight) csrResult.csr_weight[pos + i] = static_cast<edge_data_type>(buf[i].weight);
            }
        }
    }
    else
    {
        // Read in chunks: each edge is {uint64_t dest}
        constexpr countl_type CHUNK = 1ULL << 20;
        std::vector<SubwayOutEdge64> buf(std::min(edgeNum, CHUNK));
        for (countl_type pos = 0; pos < edgeNum; pos += CHUNK)
        {
            countl_type cnt = std::min(CHUNK, edgeNum - pos);
            infile.read(reinterpret_cast<char*>(buf.data()), sizeof(SubwayOutEdge64) * cnt);
            assert_msg(!infile.fail(), "Read 64-bit edge list failed at offset %zu, graphPath = [%s]", SCU64(pos), graphPath.c_str());
            for (countl_type i = 0; i < cnt; i++)
            {
                csrResult.csr_dest[pos + i] = static_cast<vertex_id_type>(buf[i].end);
                if (needWeight) csrResult.csr_weight[pos + i] = static_cast<edge_data_type>(1);
            }
        }
    }
}

inline vertex_id_type* getGraphData_Subway(CSR_Result_type& csrResult, const std::string& graphPath, Algorithm_type algorithm, bool isSortNbr,
                                           bool getOutDegree, bool getInDegree)
{
    assert_msg(CPJ::FS::isFile(graphPath), "Input graph file does not exist, graphPath = [%s]", graphPath.c_str());

    const bool needWeight = (algorithm == Algorithm_type::SSSP);
    const std::string extension = getLowerExtension(graphPath);
    assert_msg(isSubwayGraphExtension(extension), "Input graph format [%s] is not supported", extension.c_str());

    if (extension == ".bcsr64" || extension == ".bwcsr64")
    {
        loadSubwayBinary64Graph(csrResult, graphPath, needWeight, extension == ".bwcsr64");
    }
    else if (extension == ".bcsr" || extension == ".bwcsr")
    {
        loadSubwayBinaryGraph(csrResult, graphPath, needWeight, extension == ".bwcsr");
    }
    else
    {
        loadSubwayTextGraph(csrResult, graphPath, needWeight);
    }

    if (isSortNbr)
    {
        if (needWeight) CPJ::sortNbr(csrResult, true, false);
        else CPJ::sortNbr_noWeight(csrResult, true, false);
    }

    if (getOutDegree || getInDegree)
    {
        Compute_degree* computeDegree = new Compute_degree(csrResult);
        if (getOutDegree) csrResult.outDegree = computeDegree->getOutdegree();
        if (getInDegree) csrResult.inDegree = computeDegree->getIndegree();
    }

    Msg_info("Load Subway-format graph file[%s], |V| = %zu, |E| = %zu", graphPath.c_str(), SCU64(csrResult.vertexNum), SCU64(csrResult.edgeNum));
    return buildIdentityOld2New(csrResult.vertexNum);
}

} // namespace detail

inline std::string getGraphTag(const std::string& graphNameOrPath) { return detail::makeSafeGraphTag(graphNameOrPath); }

/* *********************************************************************************************************
 * @description: 按需修改图数据路径
 * @param [string&] graphName
 * @return [*]
 * *********************************************************************************************************/
vertex_id_type* getGraphData(CSR_Result_type& csrResult, std::string graphName, bool isSortNbr, bool getOutDegree, bool getInDegree)
{
    const auto algorithm = static_cast<Algorithm_type>(SCI32(FLAGS_algorithm));
    std::string inputGraphPath = FLAGS_input;
    if (inputGraphPath.empty())
    {
        const std::string extension = detail::getLowerExtension(graphName);
        if (detail::isSubwayGraphExtension(extension)) inputGraphPath = graphName;
    }
    if (!inputGraphPath.empty())
    {
        return detail::getGraphData_Subway(csrResult, inputGraphPath, algorithm, isSortNbr, getOutDegree, getInDegree);
    }

    vertex_id_type* old2new{nullptr};
    GraphFile_type graphFile;

    if (graphName == "twitter2010")
    {
        graphFile.vertices = 61578415;
        graphFile.edges = 1468364884;

        graphFile.csrOffsetFile = "/data/webgraph/bin/twitter2010/native_csrOffset_u32.bin";
        graphFile.csrDestFile = "/data/webgraph/bin/twitter2010/native_csrDest_u32.bin";
        graphFile.csrWeightFile = "/data/webgraph/bin/twitter2010/native_csrWeight_u32.bin";

    } // end of [twitter2010]

    else if (graphName == "friendster")
    {
        graphFile.vertices = 124836180;
        graphFile.edges = 1806067135;

        graphFile.csrOffsetFile = "/data/webgraph/bin/friendster/native_csrOffset_u32.bin";
        graphFile.csrDestFile = "/data/webgraph/bin/friendster/native_csrDest_u32.bin";
        graphFile.csrWeightFile = "/data/webgraph/bin/friendster/native_csrWeight_u32.bin";
    }
    else if (graphName == "uk-union")
    {
        graphFile.vertices = 133633040;
        graphFile.edges = 5475109924;

        graphFile.csrOffsetFile = "/data/webgraph/bin/uk-union/native_csrOffset_u64.bin";
        graphFile.csrDestFile = "/data/webgraph/bin/uk-union/native_csrDest_u32.bin";
        graphFile.csrWeightFile = "/data/webgraph/bin/uk-union/native_csrWeight_u32.bin";
    }
    else
    {
        assert_msg(false, "Unknow graphName [%s]", graphName.c_str());
    }

    assert_msg(graphFile.vertices < std::numeric_limits<count_type>::max(),
               "Total vertices need set the <count_type> and <vertex_id_type> to uint64_t");
    assert_msg(graphFile.edges < std::numeric_limits<countl_type>::max(), "Total edges need set the <countl_type> to uint64_t");
    if (graphFile.edges < std::numeric_limits<uint32_t>::max())
    {
        bool isSame = std::is_same_v<countl_type, uint32_t>;
        assert_msg(isSame, "Total edges can be stored by uint32_t, So set the <countl_type> to uint32_t");
    }

    if (CPJ::FS::isExist(getCGgraph_reorder_rankFile(graphName)))
    {
        graphFile.rankFile = getCGgraph_reorder_rankFile(graphName);
        graphFile.old2newFile = getCGgraph_reorder_old2newFile(graphName);
        graphFile.csrOffsetFile = getCGgraph_reorder_csrOffset(graphName);
        graphFile.csrDestFile = getCGgraph_reorder_csrDest(graphName);
        graphFile.csrWeightFile = getCGgraph_reorder_csrWeight(graphName);

        old2new = CPJ::getGraphCSR(csrResult, graphFile, algorithm, OrderMethod_type::CGgraphRV1_5, isSortNbr, getOutDegree, getInDegree);
    }
    else
    {
        CSR_Result_type csrResult_native;
        CPJ::getGraphCSR(csrResult_native, graphFile, algorithm, OrderMethod_type::NATIVE, isSortNbr, true, getInDegree);
        CPJ::ReorderGraph reorderGraph(csrResult_native);
        csrResult = reorderGraph.doReorder(graphName);
        reorderGraph.freeOldCSR();
        old2new = reorderGraph.getOld2New();
    }

    return old2new;
}

} // namespace CPJ
