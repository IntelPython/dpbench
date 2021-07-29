/* file: gbt_reg_dense_batch.cpp */
/*******************************************************************************
* Copyright 2014-2020 Intel Corporation
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
*******************************************************************************/

/*
!  Content:
!    C++ example of gradient boosted trees regression in the batch processing mode
!    with DPC++ interfaces.
!
!    The program trains the gradient boosted trees regression model on a training
!    datasetFileName and computes regression for the test data.
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-GBT_REG_DENSE_BATCH"></a>
 * \example gbt_reg_dense_batch.cpp
 */

#include "daal_sycl.h"
#include "service.h"
#include "service_sycl.h"

#include "rdtsc.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;

/* Input data set parameters */
const string dataFileName = "pca_normalized.csv";

template<class Data>
pca::ResultPtr computePCA(Data data) {
    pca::Batch<> algorithm;

    /* Set the algorithm input data */
    algorithm.input.set(pca::data, data);
    algorithm.parameter.resultsToCompute = pca::mean | pca::variance | pca::eigenvalue;
    algorithm.parameter.isDeterministic  = true;

    /* Compute results of the PCA algorithm */
    algorithm.compute();

    pca::ResultPtr result = algorithm.getResult();

    return result;
}

template<class Data>
pca::transform::ResultPtr transformPCA(Data data, pca::ResultPtr& pcaResult, const size_t nComponents) {
    /* Apply transform with whitening because means and eigenvalues are provided*/
    pca::transform::Batch<float> pcaTransform(nComponents);
    pcaTransform.input.set(pca::transform::data, data);
    pcaTransform.input.set(pca::transform::eigenvectors, pcaResult->get(pca::eigenvectors));

    pcaTransform.input.set(pca::transform::dataForTransform, pcaResult->get(pca::dataForTransform));

    pcaTransform.compute();

    /* Output transformed data */
    return pcaTransform.getResult();
}

int main(int argc, char * argv[]) {
    const size_t nFeatures = 10;
    int repeat = 1;
    int nopt;

    if (argc < 2) {
        printf("Usage: expect input size parameter, Exiting\n");
        return 0;
    } else {
        sscanf(argv[1], "%d", &nopt);
        if (argc == 3) {
            sscanf(argv[2], "%d", &nFeatures);
        }
        if (argc == 4) {
            sscanf(argv[3], "%d", &repeat);
        }
    }

    FILE *fptr;
    fptr = fopen("perf_output.csv", "a");
    if(fptr == NULL) {
        printf("Error!");
        exit(1);
    }

    FILE *fptr1;
    fptr1 = fopen("runtimes.csv", "a");
    if(fptr1 == NULL) {
        printf("Error!");
        exit(1);
    }

    double MOPS = 0.0;
    for (const auto & deviceSelector : getListOfDevices()) {
        const auto & nameDevice = deviceSelector.first;
        const auto & device     = deviceSelector.second;
        cl::sycl::queue queue(device);

        daal::services::SyclExecutionContext ctx(queue);
        services::Environment::getInstance()->setDefaultExecutionContext(ctx);

        /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
        FileDataSource<CSVFeatureManager> dataSource(dataFileName, DataSource::notAllocateNumericTable, DataSource::doDictionaryFromContext);

        auto data = SyclHomogenNumericTable<>::create(nFeatures, 0, NumericTable::notAllocate);
        /* Retrieve the data from the input file */
        dataSource.loadDataBlock(data.get());

        /* Warm up cycle */
        for(int j = 0; j < 1; j++) {
            pca::ResultPtr result = computePCA(data);
        }

        clock_t t1 = 0, t2 = 0;
        t1 = timer_rdtsc();
	pca::transform::ResultPtr pcaTransformResult;
        for(int j = 0; j < repeat; j++) {
            pca::ResultPtr result = computePCA(data);
	    pcaTransformResult = transformPCA(data, result, nFeatures);
        }
        t2 = timer_rdtsc();

        MOPS = nopt * repeat / 1e6 / ((double) (t2 - t1) / getHz());

        printf("Size: %d MOPS: %.6lf\n", nopt, MOPS);
        fflush(stdout);
        fprintf(fptr, "%d,%.6lf\n", nopt, MOPS);
        fprintf(fptr1, "%d,%.6lf\n", nopt, ((double) (t2 - t1) / getHz()));
    }
    fclose(fptr);
    fclose(fptr1);

    return 0;
}
