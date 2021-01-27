/*
Copyright (c) 2020, Intel Corporation

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
 * Neither the name of Intel Corporation nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <iostream>
#include <vector>
#include <cmath>

#include <algorithm>
#include <string>

#include <cl2.hpp>
 

//using dtype = double;
const int classesNum = 3;
//const size_t dataDim = 4;
const int dataDim = 16;

// Define the number of nearest neighbors
const int k = 10;


struct KNN
{
    cl::Context context;
    cl::CommandQueue queue;
    cl::Kernel kernel;

    cl::Buffer bufferTrain;
    cl::Buffer bufferLabels;
    cl::Buffer bufferTest;
    cl::Buffer bufferPredictions;

    void init()
    {
        std::vector<cl::Platform> allPlatforms;
        cl::Platform::get(&allPlatforms);

        if (allPlatforms.size() == 0) {
            std::cout << "No platforms found. Check OpenCL installation!\n";
            exit(1);
        }

        cl::Platform defaultPlatform = allPlatforms[1];
        std::cout << "Using platform: " << defaultPlatform.getInfo<CL_PLATFORM_NAME>() << "\n";

        std::vector<cl::Device> allDevices;
        defaultPlatform.getDevices(CL_DEVICE_TYPE_ALL, &allDevices);
        if (allDevices.size() == 0) {
            std::cout << "No devices found. Check OpenCL installation!\n";
            exit(1);
        }

        cl::Device defaultDevice = allDevices[0];
        std::cout << "Using device: " << defaultDevice.getInfo<CL_DEVICE_NAME>() << "\n";

        context = cl::Context({ defaultDevice });

        std::ifstream kernelFile("knn_kernel.hpp");
        std::string kernel_code;

        if (kernelFile)
        {
            std::ostringstream ss;
            ss << kernelFile.rdbuf(); // reading data
            kernel_code = ss.str();
        }

        cl::Program::Sources sources;
        sources.push_back({ kernel_code.c_str(), kernel_code.length() });

        cl::Program program(context, sources);
        if (program.build({ defaultDevice }) != CL_SUCCESS) {
            std::cout << "Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(defaultDevice) << std::endl;
            exit(1);
        }

        queue = cl::CommandQueue(context, defaultDevice);
        kernel = cl::Kernel(program, "run_knn");
    }

    void allocate_buffers(int trainSize, int testSize)
    {
        bufferTrain = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(double) * trainSize * dataDim);
        bufferLabels = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(int) * trainSize);
        bufferTest = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(double) * testSize * dataDim);
        bufferPredictions = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(int) * testSize);
    }

    double* map_train(int trainSize)
    {
        return (double*)queue.enqueueMapBuffer(bufferTrain, CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION, 0, sizeof(double) * trainSize * dataDim);
    }

    double* map_test(int testSize)
    {
        return (double*)queue.enqueueMapBuffer(bufferTest, CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION, 0, sizeof(double) * testSize * dataDim);
    }

    int* map_labels(int trainSize)
    {
        return (int*)queue.enqueueMapBuffer(bufferLabels, CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION, 0, sizeof(int) * trainSize);
    }

    int* map_predictions(int testSize)
    {
        return (int*)queue.enqueueMapBuffer(bufferPredictions, CL_TRUE, CL_MAP_READ, 0, sizeof(int) * testSize);
    }

    void map_all(int trainSize, int testSize)
    {
        map_train(trainSize);
        map_test(testSize);
        map_labels(trainSize);
        map_predictions(testSize);
    }

    void unmap_all(double* train_ptr, double* test_ptr, int* labels_ptr, int* predictions_ptr)
    {
        queue.enqueueUnmapMemObject(bufferTrain, train_ptr);
        queue.enqueueUnmapMemObject(bufferTest, test_ptr);
        queue.enqueueUnmapMemObject(bufferLabels, labels_ptr);
        queue.enqueueUnmapMemObject(bufferPredictions, predictions_ptr);
    }

    //MapUnmap
    void run_knn_opencl(double* train_ptr, double* test_ptr, int* labels_ptr, int* predictions_ptr, int trainSize, int testSize)
    {
        unmap_all(train_ptr, test_ptr, labels_ptr, predictions_ptr);

        // RUN KERNEL
        kernel.setArg(0, bufferTrain);
        kernel.setArg(1, trainSize);
        kernel.setArg(2, bufferLabels);
        kernel.setArg(3, bufferTest);
        kernel.setArg(4, testSize);
        kernel.setArg(5, bufferPredictions);

        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(testSize), cl::NullRange);
        map_all(trainSize, testSize);
    }

    //Write Read
    void run_knn_opencl_wr(double* train, int trainSize, int* trainLabels, double* test, int testSize, std::vector<int>& predictionsVec)
    {
        int* predictions = predictionsVec.data();

        // push write commands to queue
        queue.enqueueWriteBuffer(bufferTrain, CL_FALSE, 0, sizeof(double) * trainSize * dataDim, train);
        queue.enqueueWriteBuffer(bufferLabels, CL_FALSE, 0, sizeof(int) * trainSize, trainLabels);
        queue.enqueueWriteBuffer(bufferTest, CL_FALSE, 0, sizeof(double) * testSize * dataDim, test);

        // RUN KERNEL
        kernel.setArg(0, bufferTrain);
        kernel.setArg(1, trainSize);
        kernel.setArg(2, bufferLabels);
        kernel.setArg(3, bufferTest);
        kernel.setArg(4, testSize);
        kernel.setArg(5, bufferPredictions);

        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(testSize), cl::NullRange);

        // read result from GPU to here
        queue.enqueueReadBuffer(bufferPredictions, CL_TRUE, 0, sizeof(int) * testSize, predictions);
    }
};
