/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <numeric>

#include <raft/handle.hpp>

#include <cuml/cluster/dbscan.hpp>

#ifndef CUDA_RT_CALL
#define CUDA_RT_CALL(call)                                                    \
    {                                                                           \
        cudaError_t cudaStatus = call;                                            \
        if (cudaSuccess != cudaStatus)                                            \
        fprintf(stderr,                                                         \
                "ERROR: CUDA RT call \"%s\" in line %d of file %s failed with " \
                "%s (%d).\n",                                                   \
                #call,                                                          \
                __LINE__,                                                       \
                __FILE__,                                                       \
                cudaGetErrorString(cudaStatus),                                 \
                cudaStatus);                                                    \
    }
#endif  // CUDA_RT_CALL

template <typename T>
T get_argval(char** begin, char** end, const std::string& arg, const T default_val)
{
    T argval   = default_val;
    char** itr = std::find(begin, end, arg);
    if (itr != end && ++itr != end) {
        std::istringstream inbuf(*itr);
        inbuf >> argval;
    }
    return argval;
}

bool get_arg(char** begin, char** end, const std::string& arg)
{
    char** itr = std::find(begin, end, arg);
    if (itr != end) { return true; }
    return false;
}

void printUsage()
{
    std::cout << "To run default example use:" << std::endl
            << "    dbscan_example [-dev_id <GPU id>]" << std::endl
            << "For other cases:" << std::endl
            << "    dbscan_example [-dev_id <GPU id>] -input <samples-file> "
            << "-num_samples <number of samples> -num_features <number of features> "
            << "[-min_pts <minimum number of samples in a cluster>] "
            << "[-eps <maximum distance between any two samples of a cluster>] "
            << "[-max_bytes_per_batch <maximum memory to use (in bytes) for batch size "
            "calculation>] "
            << std::endl;
    return;
}

void generateDefaultDataset(std::vector<float>& inputData,
                            size_t batchSize,
                            int *pNbRows,
                            size_t nCols,
                            int minPts,
                            float eps,
                            size_t& max_bytes_per_batch)
{
    constexpr size_t MAX_NUM_ROWS = 10000;
    constexpr size_t MAX_NUM_COLS = 1024;
    size_t nRows = 0;
    for(int b = 0; b < batchSize; ++b) {
        nRows += pNbRows[b];
    }
    assert(nRows <= MAX_NUM_ROWS);
    assert(nCols <= MAX_NUM_COLS);

    std::vector<float> vData(nRows * nCols, 0.f);
    std::srand(std::time(nullptr));
    for(auto &it : vData) {
        // it = static_cast<float>(std::rand()) / RAND_MAX * 1.0 - 0.5;
        it = static_cast<float>(std::rand()) / RAND_MAX * 3.0 - 1.5;
    }

    max_bytes_per_batch = 0;  // allow algorithm to set this
    inputData.assign(vData.begin(), vData.end());
}

int main(int argc, char* argv[])
{
    int devId         = get_argval<int>(argv, argv + argc, "-dev_id", 0);
    int val           = get_argval<int>(argv, argv + argc, "-val", 20);
    size_t nGroups    = get_argval<size_t>(argv, argv + argc, "-num_groups", 1);
    size_t nCols      = get_argval<size_t>(argv, argv + argc, "-num_features", 3);
    std::string input = get_argval<std::string>(argv, argv + argc, "-input", std::string(""));
    int minPts        = get_argval<int>(argv, argv + argc, "-min_pts", 3);
    float eps         = get_argval<float>(argv, argv + argc, "-eps", 1.0f);
    size_t max_bytes_per_batch =
        get_argval<size_t>(argv, argv + argc, "-max_bytes_per_batch", (size_t)13e9);

    {
        cudaError_t cudaStatus = cudaSuccess;
        cudaStatus             = cudaSetDevice(devId);
        if (cudaSuccess != cudaStatus) {
            std::cerr << "ERROR: Could not select CUDA device with the id: " << devId << "("
                        << cudaGetErrorString(cudaStatus) << ")" << std::endl;
            return 1;
        }
        cudaStatus = cudaFree(0);
        if (cudaSuccess != cudaStatus) {
        std::cerr << "ERROR: Could not initialize CUDA on device: " << devId << "("
                    << cudaGetErrorString(cudaStatus) << ")" << std::endl;
        return 1;
        }
    }

    std::srand(std::time(nullptr));
    std::vector<int> vConsRows;
    for(int i = 0; i < nGroups; ++i) {
        vConsRows.emplace_back(rand() / RAND_MAX * val + val);
    }
    // const std::vector<int> vConsRows{23, 24, 25, 25, 26, 26, 27, 27, 28, 28, 29, 29, 30, 30, 31, 31, 32, 32, 33, 33, 40, 40};
    // const std::vector<int> vConsRows{2300, 2400, 2500, 2500, 2600, 2600, 2700, 2700, 2800, 2800, 2900, 2900, 3000, 3000, 3100, 3100, 3200, 3200, 3300, 3300, 4000, 4000};
    assert(nGroups <= vConsRows.size());
    std::vector<int> vRows(nGroups, 1);
    vRows.assign(vConsRows.begin(), vConsRows.begin() + nGroups);
    size_t nTotalRows = std::accumulate(vRows.begin(), vRows.end(), 0);

    std::vector<float> h_inputData;

    if (input == "") {
        // Samples file not specified, run with defaults
        std::cout << "Samples file not specified. (-input option)" << std::endl;
        std::cout << "Running with default dataset:" << std::endl;
        generateDefaultDataset(h_inputData, nGroups, vRows.data(), nCols, minPts, eps, max_bytes_per_batch);
    } else if (vRows.empty() || nCols == 0) {
        // Samples file specified but nRows and nCols is not specified
        // Print usage and quit
        std::cerr << "Samples file: " << input << std::endl;
        std::cerr << "Incorrect value for (num_samples x num_features): (" << vRows.size() << " x " << nCols
                  << ")" << std::endl;
        printUsage();
        return 1;
    } else {
        // All options are correctly specified
        // Try to read input file now
        std::ifstream input_stream(input, std::ios::in);
        if (!input_stream.is_open()) {
            std::cerr << "ERROR: Could not open input file " << input << std::endl;
            return 1;
        }
        std::cout << "Trying to read samples from " << input << std::endl;
        h_inputData.reserve(nTotalRows * nCols);
        float val = 0.0;
        while (input_stream >> val && h_inputData.size() <= nTotalRows * nCols) {
            h_inputData.push_back(val);
        }
        if (h_inputData.size() != nTotalRows * nCols) {
            std::cerr << "ERROR: Read " << h_inputData.size() << " from " << input
                      << ", while expecting to read: " << nTotalRows * nCols << " (num_samples*num_features)"
                      << std::endl;
            return 1;
        }
    }

    cudaStream_t stream;
    CUDA_RT_CALL(cudaStreamCreate(&stream));
    raft::handle_t handle{stream};

    std::vector<int> h_labels(nTotalRows);
    int* d_labels      = nullptr;
    float* d_inputData = nullptr;

    CUDA_RT_CALL(cudaMalloc(&d_labels, nTotalRows * sizeof(int)));
    CUDA_RT_CALL(cudaMalloc(&d_inputData, nTotalRows * nCols * sizeof(float)));
    CUDA_RT_CALL(cudaMemcpyAsync(d_inputData,
                                h_inputData.data(),
                                nTotalRows * nCols * sizeof(float),
                                cudaMemcpyHostToDevice,
                                stream));

    std::cout << "Running DBSCAN with following parameters:" << std::endl
              << "Number of groups - " << nGroups << std::endl 
              << "Number of samples - " << nTotalRows << std::endl
              << "Number of features - " << nCols << std::endl
              << "min_pts - " << minPts << std::endl
              << "eps - " << eps << std::endl
              << "max_bytes_per_batch - " << max_bytes_per_batch << std::endl;

    std::cout << std::endl << "=====\t batched version\t =====" << std::endl;
    ML::Dbscan::fit(handle,
                    d_inputData,
                    nGroups,
                    vRows.data(),
                    nTotalRows,
                    nCols,
                    eps,
                    minPts,
                    raft::distance::L2SqrtUnexpanded,
                    d_labels,
                    nullptr,
                    max_bytes_per_batch,
                    CUML_LEVEL_INFO,
                    false);

    CUDA_RT_CALL(cudaMemcpyAsync(h_labels.data(), d_labels, nTotalRows * sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_RT_CALL(cudaStreamSynchronize(stream));

    for(int b = 0, start_idx = 0; b < nGroups; ++b) {
        std::cout << "Group " << b << " : ";
        for(int i = 0; i < vRows[b]; ++i) {
            std::cout << h_labels[start_idx++] << " ";
        }
        std::cout << std::endl;
    }

    // for(int b = 0; b < nGroups; ++b) {
    //     std::map<long, size_t> histogram;
    //     int start_row = 0;
    //     for (int row = start_row; row < start_row + vRows[b]; row++) {
    //         if (histogram.find(h_labels[row]) == histogram.end()) {
    //             histogram[h_labels[row]] = 1;
    //         } else {
    //             histogram[h_labels[row]]++;
    //         }
    //     }

    //     size_t nClusters = 0;
    //     size_t noise     = 0;
    //     std::cout << "Group " << b << std::endl;
    //     std::cout << "Histogram of samples" << std::endl;
    //     std::cout << "Cluster id, Number samples" << std::endl;
    //     for (auto it = histogram.begin(); it != histogram.end(); it++) {
    //         if (it->first != -1) {
    //             std::cout << std::setw(10) << it->first << ", " << it->second << std::endl;
    //             nClusters++;
    //         } else {
    //             noise += it->second;
    //         }
    //     }

    //     std::cout << "Total number of clusters: " << nClusters << std::endl;
    //     std::cout << "Noise samples: " << noise << std::endl;
    // }
    
    /* print inputData */
    // std::for_each(h_inputData.begin(), h_inputData.end(), [=](float x){ std::cout << x << " "; }); std::cout << std::endl;
    std::cout << std::endl << "=====\t compare version\t =====" << std::endl;
    std::vector<std::vector<int>> h_all_labels;
    int _startRow = 0;
    for(int b = 0; b < nGroups; ++b) {
        int _nRows = vRows.at(b);
        int _nCols = nCols;
        std::vector<float> _h_inputData(_nRows * _nCols, 0.0);
        _h_inputData.assign(h_inputData.begin() + _startRow * _nCols, h_inputData.begin() + (_startRow + _nRows) * _nCols);
        /* print inputData */
        // std::for_each(_h_inputData.begin(), _h_inputData.end(), [=](float x){ std::cout << x << " "; }); std::cout << std::endl;

        std::vector<int> _h_labels(_nRows, 0);
        int *_d_labels = nullptr;
        float *_d_inputData = nullptr;
        CUDA_RT_CALL(cudaMalloc(&_d_labels, _h_labels.size() * sizeof(int)));
        CUDA_RT_CALL(cudaMalloc(&_d_inputData, _h_inputData.size() * sizeof(float)));
        CUDA_RT_CALL(cudaMemcpyAsync(_d_inputData, _h_inputData.data(), _h_inputData.size() * sizeof(float), cudaMemcpyHostToDevice, stream));

        ML::Dbscan::fit(handle,
                        _d_inputData,
                        _nRows,
                        _nCols,
                        eps,
                        minPts,
                        raft::distance::L2SqrtUnexpanded,
                        _d_labels,
                        nullptr,
                        max_bytes_per_batch,
                        CUML_LEVEL_INFO,
                        false);

        CUDA_RT_CALL(cudaMemcpyAsync(_h_labels.data(), _d_labels, _h_labels.size() * sizeof(int), cudaMemcpyDeviceToHost, stream));
        CUDA_RT_CALL(cudaStreamSynchronize(stream));
        h_all_labels.emplace_back(_h_labels);

        CUDA_RT_CALL(cudaFree(_d_labels));
        CUDA_RT_CALL(cudaFree(_d_inputData));
        _startRow += _nRows;
    }
    for(int b = 0; b < h_all_labels.size(); ++b) {
        std::vector<int> _h_labels = h_all_labels.at(b);
        std::cout << "Group " << b << " : ";
        std::for_each(_h_labels.begin(), _h_labels.end(), [=](int x){ std::cout << x << " "; }); std::cout << std::endl;
    }

    int _diff = 0;
    for(int b = 0, _idx = 0; b < h_all_labels.size(); ++b) {
        std::vector<int> _h_labels = h_all_labels.at(b);
        for(int i = 0; i < _h_labels.size(); ++i) {
            _diff += h_labels.at(_idx++) - _h_labels.at(i);
        }
    } std::cout << "Diff is " << _diff << std::endl;

    CUDA_RT_CALL(cudaFree(d_labels));
    CUDA_RT_CALL(cudaFree(d_inputData));
    CUDA_RT_CALL(cudaStreamDestroy(stream));
    CUDA_RT_CALL(cudaDeviceSynchronize());
    return 0;
}
