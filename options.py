import os
import sys
import enum

VTUNE_HOTSPOTS_CMD = ["vtune", "-run-pass-thru=--no-altstack", "-collect=hotspots"]
VTUNE_THREADING_CMD = ["vtune", "-run-pass-thru=--no-altstack", "-collect=threading"]
VTUNE_GPU_OFFLOAD_CMD = ["vtune", "-run-pass-thru=--no-altstack", "-collect=gpu-offload", "-result-dir=vtune_dir"]
VTUNE_GPU_HOTSPOTS_CMD = ["vtune", "-run-pass-thru=--no-altstack", "-collect=gpu-hotspots", "-result-dir=vtune_hotspots_dir"]

ADVISOR_SURVEY_CMD = ["advisor", "--collect=survey", "-run-pass-thru=--no-altstack", "-project-dir=roofline",
                      "--search-dir", "src:r=."]
ADVISOR_FLOP_CMD = ["advisor", "--collect=tripcounts", "--project-dir=roofline", "--search-dir src:r=.", "--flop",
                    "--no-trip-counts"]
ADVISOR_ROOFLINE_CMD = ["advisor", "--report=roofline", "--project-dir=roofline",
                        "--report-output=roofline/roofline.html"]

ADVISOR_GPU_SURVEY_CMD = ["advisor", "--collect=survey", "--profile-gpu", "-run-pass-thru=--no-altstack",
                          "-project-dir=roofline", "--search-dir", "src:r=."]
ADVISOR_GPU_FLOP_CMD = ["advisor", "--collect=tripcounts", "--profile-gpu",
                        "--project-dir=roofline", "--search-dir src:r=.", "--flop", "--no-trip-counts"]
ADVISOR_GPU_ROOFLINE_CMD = ["advisor", "--report=roofline", "--gpu", "--project-dir=roofline",
                            "--report-output=roofline/roofline.html"]

if 'A21_SDK_ROOT' in os.environ:
    ADVISOR_GPU_METRICS_CMD = ["advixe-python", os.environ['A21_SDK_ROOT'] + "/advisor/latest" + "/pythonapi/examples/survey_gpu.py",
                               "roofline"]
elif 'ONEAPI_ROOT' in os.environ:
    ADVISOR_GPU_METRICS_CMD = ["advixe-python", os.environ['ONEAPI_ROOT'] + "/advisor/latest" + "/pythonapi/examples/survey_gpu.py",
                               "roofline"]

# advixe-python /opt/intel/inteloneapi/advisor/latest/pythonapi/examples/survey_gpu.py roofline > GPU_Metrics.txt


class all_workloads(enum.Enum):
    blackscholes = 'blackscholes'
    dbscan = 'dbscan'
    gpairs = 'gpairs'
    kmeans = 'kmeans'
    knn = 'knn'
    l2_distance = 'l2_distance'
    pairwise_distance = 'pairwise_distance'
    pca = 'pca'
    rambo = 'rambo'
    #pathfinder = 'pathfinder'
    # pygbm = 'pygbm'
    # random_forest = 'random_forest'
    # svm = 'svm'
    # umap = 'umap'

    def __str__(self):
        return self.value


class run(enum.Enum):
    all = 'all'
    execute = 'execute'
    plot = 'plot'

    def __str__(self):
        return self.value


class implementation(enum.Enum):
    all = 'all'
    native = 'native'
    native_dpcpp = 'native_dpcpp'
    native_optimised = 'native_optimised'
    numba = 'numba'
    scikit_learn = 'scikit_learn'
    daal4py = 'daal4py'
    dpnp = 'dpnp'

    def __str__(self):
        return self.value


class platform(enum.Enum):
    all = 'all'
    cpu = 'cpu'
    gpu = 'gpu'

    def __str__(self):
        return self.value


class analysis(enum.Enum):
    all = 'all'
    test = 'test'
    perf = 'perf'
    vtune = 'vtune'
    advisor = 'advisor'

    def __str__(self):
        return self.value


class workloads():
    def __init__(self, input_wls=[], kernel_mode=False):
        print(input_wls)

        wl_names = {all_workloads.blackscholes.value:{'numba':"bs_erf_numba_numpy.py",
                                                      'kernel':"bs_erf_numba_kernel.py"},
                    all_workloads.dbscan.value:{'numba':"dbscan.py",
                                                'kernel':"dbscan_kernel.py"},
                    all_workloads.kmeans.value:{'numba':"kmeans.py",
                                                'kernel':"kmeans_kernel.py"},
                    all_workloads.knn.value:{'numba':"knn.py",
                                             'kernel':"knn_kernel.py"},
                    all_workloads.l2_distance.value:{'numba':"l2_distance.py",
                                                     'kernel':"l2_distance.py"},
                    all_workloads.pairwise_distance.value:{'numba':"pairwise_distance.py",
                                                           'kernel':"pairwise_distance_kernel.py"},
                    all_workloads.pca.value:{'numba':"pca.py",
                                             'kernel':"pca.py"},
                    # all_workloads.pathfinder.value:{'numba':"pathfinder.py",
                    #                            'kernel':"pathfinder.py"},
                    all_workloads.rambo.value:{'numba':"rambo.py",
                                               'kernel':"rambo_kernel.py"},
                    all_workloads.gpairs.value:{'numba':"run_gpairs.py",
                                                'kernel':"run_gpairs.py"},
        }

        self.wl_list = {
            all_workloads.blackscholes.value: {
                'execute': False,
                'ref_input': 2**28,
                'NUMBA_TEST_CMD': ["python", wl_names[all_workloads.blackscholes.value]['numba'] if not kernel_mode else wl_names[all_workloads.blackscholes.value]['kernel'], "--steps", "1"],
                'NUMBA_PERF_CMD': ["python", wl_names[all_workloads.blackscholes.value]['numba'] if not kernel_mode else wl_names[all_workloads.blackscholes.value]['kernel']],
                'NUMBA_VTUNE_CMD': ["python", wl_names[all_workloads.blackscholes.value]['numba'] if not kernel_mode else wl_names[all_workloads.blackscholes.value]['kernel'], "--steps", "1", "--size", str(2**28)],
                'NUMBA_ADVISOR_CMD': ["python", wl_names[all_workloads.blackscholes.value]['numba'] if not kernel_mode else wl_names[all_workloads.blackscholes.value]['kernel'], "--steps", "1", "--size", str(2**28)],
                'NUMBA_CPU_TEST_CMD': ["python", wl_names[all_workloads.blackscholes.value]['numba'], "--steps", "1"],
                'NUMBA_CPU_PERF_CMD': ["python", wl_names[all_workloads.blackscholes.value]['numba']],
                'NUMBA_CPU_VTUNE_CMD': ["python", wl_names[all_workloads.blackscholes.value]['numba'], "--steps", "1", "--size", str(2**28)],
                'NUMBA_CPU_ADVISOR_CMD': ["python", wl_names[all_workloads.blackscholes.value]['numba'], "--steps", "1", "--size", str(2**28)],
                'NATIVE_TEST_CMD': ["./black_scholes", "1"],
                'NATIVE_PERF_CMD': ["./black_scholes"],
                'NATIVE_VTUNE_CMD': ["./black_scholes", "1", str(2**28), "1"],
                'NATIVE_ADVISOR_CMD': ["./black_scholes", "1", str(2**28), "1"],
            },
            all_workloads.dbscan.value: {
                'execute': False,
                'ref_input': 2**14,
                'NUMBA_TEST_CMD': ["python", wl_names[all_workloads.dbscan.value]['numba'] if not kernel_mode else wl_names[all_workloads.dbscan.value]['kernel'], "--steps", "1"],
                'NUMBA_PERF_CMD': ["python", wl_names[all_workloads.dbscan.value]['numba'] if not kernel_mode else wl_names[all_workloads.dbscan.value]['kernel']],
                'NUMBA_VTUNE_CMD': ["python", wl_names[all_workloads.dbscan.value]['numba'] if not kernel_mode else wl_names[all_workloads.dbscan.value]['kernel'], "--steps", "1", "--size", str(2 ** 14)],
                'NUMBA_ADVISOR_CMD': ["python", wl_names[all_workloads.dbscan.value]['numba'] if not kernel_mode else wl_names[all_workloads.dbscan.value]['kernel'], "--steps", "1", "--size", str(2 ** 14)],
                'NUMBA_CPU_TEST_CMD': ["python", wl_names[all_workloads.dbscan.value]['numba'], "--steps", "1"],
                'NUMBA_CPU_PERF_CMD': ["python", wl_names[all_workloads.dbscan.value]['numba']],
                'NUMBA_CPU_VTUNE_CMD': ["python", wl_names[all_workloads.dbscan.value]['numba'], "--steps", "1", "--size", str(2 ** 14)],
                'NUMBA_CPU_ADVISOR_CMD': ["python", wl_names[all_workloads.dbscan.value]['numba'], "--steps", "1", "--size", str(2 ** 14)],
                'SCIKIT_LEARN_TEST_CMD': ["python", "dbscan.py", "--steps", "1"],
                'SCIKIT_LEARN_PERF_CMD': ["python", "dbscan.py"],
                'SCIKIT_LEARN_VTUNE_CMD': ["python", "dbscan.py", "--steps", "1", "--size", str(2 ** 13)],
                'SCIKIT_LEARN_ADVISOR_CMD': ["python", "dbscan.py", "--steps", "1", "--size", str(2 ** 13)],
                'DAAL4PY_TEST_CMD': ["python", "dbscan.py", "--steps", "1"],
                'DAAL4PY_PERF_CMD': ["python", "dbscan.py"],
                'DAAL4PY_VTUNE_CMD': ["python", "dbscan.py", "--steps", "1", "--size", str(2 ** 13)],
                'DAAL4PY_ADVISOR_CMD': ["python", "dbscan.py", "--steps", "1", "--size", str(2 ** 13)],
                'NATIVE_TEST_CMD': ["python", "base_dbscan.py", "--steps", "1"],
                'NATIVE_PERF_CMD': ["python", "base_dbscan.py"],
                'NATIVE_VTUNE_CMD': ["./dbscan", "1", str(2 ** 14), "10", "20", "0.6","1"], #./dbscan 1 8192 10 20 0.6 2
                'NATIVE_ADVISOR_CMD': ["./dbscan", "1", str(2 ** 14), "10", "20", "0.6","1"], #./dbscan 1 8192 10 20 0.6 2
                'NATIVE_OPTIMISED_TEST_CMD': ["python", "base_dbscan.py", "--steps", "1"],
                'NATIVE_OPTIMISED_PERF_CMD': ["python", "base_dbscan.py"],
                'NATIVE_OPTIMISED_VTUNE_CMD': ["./dbscan", "1", str(2 ** 13), "10", "20", "0.6","100"],
                'NATIVE_OPTIMISED_ADVISOR_CMD': ["./dbscan", "1", str(2 ** 13), "10", "20", "0.6","100"],
            },
            all_workloads.kmeans.value: {
                'execute': False,
                'ref_input': 2**22,
                'NUMBA_TEST_CMD': ["python", wl_names[all_workloads.kmeans.value]['numba'] if not kernel_mode else wl_names[all_workloads.kmeans.value]['kernel'], "--steps", "1"],
                'NUMBA_PERF_CMD': ["python", wl_names[all_workloads.kmeans.value]['numba'] if not kernel_mode else wl_names[all_workloads.kmeans.value]['kernel']],
                'NUMBA_VTUNE_CMD': ["python", wl_names[all_workloads.kmeans.value]['numba'] if not kernel_mode else wl_names[all_workloads.kmeans.value]['kernel'], "--steps", "1", "--size", str(2**20)],
                'NUMBA_ADVISOR_CMD': ["python", wl_names[all_workloads.kmeans.value]['numba'] if not kernel_mode else wl_names[all_workloads.kmeans.value]['kernel'], "--steps", "1", "--size", str(2**20)],
                'NUMBA_CPU_TEST_CMD': ["python", wl_names[all_workloads.kmeans.value]['numba'], "--steps", "1"],
                'NUMBA_CPU_PERF_CMD': ["python", wl_names[all_workloads.kmeans.value]['numba']],
                'NUMBA_CPU_VTUNE_CMD': ["python", wl_names[all_workloads.kmeans.value]['numba'], "--steps", "1", "--size", str(2**20)],
                'NUMBA_CPU_ADVISOR_CMD': ["python", wl_names[all_workloads.kmeans.value]['numba'], "--steps", "1", "--size", str(2**20)],
                'SCIKIT_LEARN_TEST_CMD': ["python", "kmeans.py", "--steps", "1"],
                'SCIKIT_LEARN_PERF_CMD': ["python", "kmeans.py"],
                'SCIKIT_LEARN_VTUNE_CMD': ["python", "kmeans.py", "--steps", "1", "--size", str(2**20)],
                'SCIKIT_LEARN_ADVISOR_CMD': ["python", "kmeans.py", "--steps", "1", "--size", str(2**20)],
                'DAAL4PY_TEST_CMD': ["python", "kmeans.py", "--steps", "1"],
                'DAAL4PY_PERF_CMD': ["python", "kmeans.py"],
                'DAAL4PY_VTUNE_CMD': ["python", "kmeans.py", "--steps", "1", "--size", str(2**20)],
                'DAAL4PY_ADVISOR_CMD': ["python", "kmeans.py", "--steps", "1", "--size", str(2**20)],
                'NATIVE_TEST_CMD': ["./kmeans", "1"],
                'NATIVE_PERF_CMD': ["./kmeans"],
                'NATIVE_VTUNE_CMD': ["./kmeans", "1", str(2**20)],
                'NATIVE_ADVISOR_CMD': ["./kmeans", "1", str(2**20)],
                'NATIVE_OPTIMISED_TEST_CMD': ["python", "base_kmeans.py", "--steps", "1"],
                'NATIVE_OPTIMISED_PERF_CMD': ["python", "base_kmeans.py"],
                'NATIVE_OPTIMISED_VTUNE_CMD': ["python", "base_kmeans.py", "--steps", "1", "--size", str(2**20)],
                'NATIVE_OPTIMISED_ADVISOR_CMD': ["python", "base_kmeans.py", "--steps", "1", "--size", str(2**20)],
            },
            all_workloads.knn.value: {
                'execute': False,
                'ref_input': 2**19,
                'NUMBA_TEST_CMD': ["python", wl_names[all_workloads.knn.value]['numba'] if not kernel_mode else wl_names[all_workloads.knn.value]['kernel'], "--steps", "1"],
                'NUMBA_PERF_CMD': ["python", wl_names[all_workloads.knn.value]['numba'] if not kernel_mode else wl_names[all_workloads.knn.value]['kernel']],
                'NUMBA_VTUNE_CMD': ["python", wl_names[all_workloads.knn.value]['numba'] if not kernel_mode else wl_names[all_workloads.knn.value]['kernel'], "--steps", "1", "--size", str(2 ** 19)],
                'NUMBA_ADVISOR_CMD': ["python", wl_names[all_workloads.knn.value]['numba'] if not kernel_mode else wl_names[all_workloads.knn.value]['kernel'], "--steps", "1", "--size", str(2 ** 19)],
                'NUMBA_CPU_TEST_CMD': ["python", wl_names[all_workloads.knn.value]['numba'], "--steps", "1"],
                'NUMBA_CPU_PERF_CMD': ["python", wl_names[all_workloads.knn.value]['numba']],
                'NUMBA_CPU_VTUNE_CMD': ["python", wl_names[all_workloads.knn.value]['numba'], "--steps", "1", "--size", str(2 ** 10)],
                'NUMBA_CPU_ADVISOR_CMD': ["python", wl_names[all_workloads.knn.value]['numba'], "--steps", "1", "--size", str(2 ** 10)],
                'SCIKIT_LEARN_TEST_CMD': ["python", "knn.py", "--steps", "1"],
                'SCIKIT_LEARN_PERF_CMD': ["python", "knn.py"],
                'SCIKIT_LEARN_VTUNE_CMD': ["python", "knn.py", "--steps", "1", "--size", str(2 ** 10)],
                'SCIKIT_LEARN_ADVISOR_CMD': ["python", "knn.py", "--steps", "1", "--size", str(2 ** 10)],
                'DAAL4PY_TEST_CMD': ["python", "knn.py", "--steps", "1"],
                'DAAL4PY_PERF_CMD': ["python", "knn.py"],
                'DAAL4PY_VTUNE_CMD': ["python", "knn.py", "--steps", "1", "--size", str(2 ** 10)],
                'DAAL4PY_ADVISOR_CMD': ["python", "knn.py", "--steps", "1", "--size", str(2 ** 10)],
                'NATIVE_TEST_CMD': ["./knn", "1"],
                'NATIVE_PERF_CMD': ["./knn"],
                'NATIVE_VTUNE_CMD': ["./knn", "1", str(2 ** 19)],
                'NATIVE_ADVISOR_CMD': ["./knn", "1", str(2 ** 19)],
                'NATIVE_OPTIMISED_TEST_CMD': ["python", "base_knn.py", "--steps", "1"],
                'NATIVE_OPTIMISED_PERF_CMD': ["python", "base_knn.py"],
                'NATIVE_OPTIMISED_VTUNE_CMD': ["python", "base_knn.py", "--steps", "1", "--size", str(2 ** 10)],
                'NATIVE_OPTIMISED_ADVISOR_CMD': ["python", "base_knn.py", "--steps", "1", "--size", str(2 ** 10)],
            },
            all_workloads.l2_distance.value: {
                'execute': False,
                'ref_input': 2**25,
                'NUMBA_TEST_CMD': ["python", wl_names[all_workloads.l2_distance.value]['numba'] if not kernel_mode else wl_names[all_workloads.l2_distance.value]['kernel'], "--steps", "1"],
                'NUMBA_PERF_CMD': ["python", wl_names[all_workloads.l2_distance.value]['numba'] if not kernel_mode else wl_names[all_workloads.l2_distance.value]['kernel']],
                'NUMBA_VTUNE_CMD': ["python", wl_names[all_workloads.l2_distance.value]['numba'] if not kernel_mode else wl_names[all_workloads.l2_distance.value]['kernel'], "--steps", "1", "--size", str(2**25)],
                'NUMBA_ADVISOR_CMD': ["python", wl_names[all_workloads.l2_distance.value]['numba'] if not kernel_mode else wl_names[all_workloads.l2_distance.value]['kernel'], "--steps", "1", "--size", str(2**25)],
                'NUMBA_CPU_TEST_CMD': ["python", wl_names[all_workloads.l2_distance.value]['numba'], "--steps", "1"],
                'NUMBA_CPU_PERF_CMD': ["python", wl_names[all_workloads.l2_distance.value]['numba']],
                'NUMBA_CPU_VTUNE_CMD': ["python", wl_names[all_workloads.l2_distance.value]['numba'], "--steps", "1", "--size", str(2**25)],
                'NUMBA_CPU_ADVISOR_CMD': ["python", wl_names[all_workloads.l2_distance.value]['numba'], "--steps", "1", "--size", str(2**25)],
                'NATIVE_TEST_CMD': ["./l2_distance", "1"],
                'NATIVE_PERF_CMD': ["./l2_distance"],
                'NATIVE_VTUNE_CMD': ["./l2_distance", "1", str(2**25)],
                'NATIVE_ADVISOR_CMD': ["./l2_distance", "1", str(2**25)],
                # 'NATIVE_OPTIMISED_TEST_CMD': ["./l2_distance", "1"],
                # 'NATIVE_OPTIMISED_PERF_CMD': ["./l2_distance"],
                # 'NATIVE_OPTIMISED_VTUNE_CMD': ["./l2_distance", "1", str(2**22)],
                # 'NATIVE_OPTIMISED_ADVISOR_CMD': ["./l2_distance", "1", str(2**22)],
            },
            all_workloads.pairwise_distance.value: {
                'execute': False,
                'ref_input': 2**14,
                'NUMBA_TEST_CMD': ["python", wl_names[all_workloads.pairwise_distance.value]['numba'] if not kernel_mode else wl_names[all_workloads.pairwise_distance.value]['kernel'], "--steps", "1"],
                'NUMBA_PERF_CMD': ["python", wl_names[all_workloads.pairwise_distance.value]['numba'] if not kernel_mode else wl_names[all_workloads.pairwise_distance.value]['kernel']],
                'NUMBA_VTUNE_CMD': ["python", wl_names[all_workloads.pairwise_distance.value]['numba'] if not kernel_mode else wl_names[all_workloads.pairwise_distance.value]['kernel'], "--steps", "1", "--size", str(2**14)],
                'NUMBA_ADVISOR_CMD': ["python", wl_names[all_workloads.pairwise_distance.value]['numba'] if not kernel_mode else wl_names[all_workloads.pairwise_distance.value]['kernel'], "--steps", "1", "--size", str(2**14)],
                'NUMBA_CPU_TEST_CMD': ["python", wl_names[all_workloads.pairwise_distance.value]['numba'], "--steps", "1"],
                'NUMBA_CPU_PERF_CMD': ["python", wl_names[all_workloads.pairwise_distance.value]['numba']],
                'NUMBA_CPU_VTUNE_CMD': ["python", wl_names[all_workloads.pairwise_distance.value]['numba'], "--steps", "1", "--size", str(2**13)],
                'NUMBA_CPU_ADVISOR_CMD': ["python", wl_names[all_workloads.pairwise_distance.value]['numba'], "--steps", "1", "--size", str(2**13)],
                'NATIVE_TEST_CMD': ["./pairwise_distance", "1"],
                'NATIVE_PERF_CMD': ["./pairwise_distance"],
                'NATIVE_VTUNE_CMD': ["./pairwise_distance", "1", str(2**14), "1"],
                'NATIVE_ADVISOR_CMD': ["./pairwise_distance", "1", str(2**14), "1"],
            },
            all_workloads.pca.value: {
                'execute': False,
                'ref_input': 2**19,
                'NUMBA_TEST_CMD': ["python", wl_names[all_workloads.pca.value]['numba'] if not kernel_mode else wl_names[all_workloads.pca.value]['kernel'], "--steps", "1"],
                'NUMBA_PERF_CMD': ["python", wl_names[all_workloads.pca.value]['numba'] if not kernel_mode else wl_names[all_workloads.pca.value]['kernel']],
                'NUMBA_VTUNE_CMD': ["python", wl_names[all_workloads.pca.value]['numba'] if not kernel_mode else wl_names[all_workloads.pca.value]['kernel'], "--steps", "1", "--size", str(2**15)],
                'NUMBA_ADVISOR_CMD': ["python", wl_names[all_workloads.pca.value]['numba'] if not kernel_mode else wl_names[all_workloads.pca.value]['kernel'], "--steps", "1", "--size", str(2**15)],
                'NUMBA_CPU_TEST_CMD': ["python", wl_names[all_workloads.pca.value]['numba'], "--steps", "1"],
                'NUMBA_CPU_PERF_CMD': ["python", wl_names[all_workloads.pca.value]['numba']],
                'NUMBA_CPU_VTUNE_CMD': ["python", wl_names[all_workloads.pca.value]['numba'], "--steps", "1", "--size", str(2**15)],
                'NUMBA_CPU_ADVISOR_CMD': ["python", wl_names[all_workloads.pca.value]['numba'], "--steps", "1", "--size", str(2**15)],
                'SCIKIT_LEARN_TEST_CMD': ["python", "pca.py", "--steps", "1"],
                'SCIKIT_LEARN_PERF_CMD': ["python", "pca.py"],
                'SCIKIT_LEARN_VTUNE_CMD': ["python", "pca.py", "--steps", "1", "--size", str(2**15)],
                'SCIKIT_LEARN_ADVISOR_CMD': ["python", "pca.py", "--steps", "1", "--size", str(2**15)],
                'DAAL4PY_TEST_CMD': ["python", "pca.py", "--steps", "1"],
                'DAAL4PY_PERF_CMD': ["python", "pca.py"],
                'DAAL4PY_VTUNE_CMD': ["python", "pca.py", "--steps", "1", "--size", str(2**15)],
                'DAAL4PY_ADVISOR_CMD': ["python", "pca.py", "--steps", "1", "--size", str(2**15)],
                'NATIVE_TEST_CMD': ["python", "base_pca.py", "--steps", "1"],
                'NATIVE_PERF_CMD': ["python", "base_pca.py"],
                'NATIVE_VTUNE_CMD': ["python", "base_pca.py", "--steps", "1", "--size", str(2**15)],
                'NATIVE_ADVISOR_CMD': ["python", "base_pca.py", "--steps", "1", "--size", str(2**15)],
                'NATIVE_OPTIMISED_TEST_CMD': ["python", "base_pca.py", "--steps", "1"],
                'NATIVE_OPTIMISED_PERF_CMD': ["python", "base_pca.py"],
                'NATIVE_OPTIMISED_VTUNE_CMD': ["python", "base_pca.py", "--steps", "1", "--size", str(2**15)],
                'NATIVE_OPTIMISED_ADVISOR_CMD': ["python", "base_pca.py", "--steps", "1", "--size", str(2**15)],
            },
            all_workloads.rambo.value: {
                'execute': False,
                'ref_input': 524288,
                'NUMBA_TEST_CMD': ["python", wl_names[all_workloads.rambo.value]['numba'] if not kernel_mode else wl_names[all_workloads.rambo.value]['kernel'], "--steps", "1"],
                'NUMBA_PERF_CMD': ["python", wl_names[all_workloads.rambo.value]['numba'] if not kernel_mode else wl_names[all_workloads.rambo.value]['kernel']],
                'NUMBA_VTUNE_CMD': ["python", wl_names[all_workloads.rambo.value]['numba'] if not kernel_mode else wl_names[all_workloads.rambo.value]['kernel'], "--steps", "1", "--size", "262144"],
                'NUMBA_ADVISOR_CMD': ["python", wl_names[all_workloads.rambo.value]['numba'] if not kernel_mode else wl_names[all_workloads.rambo.value]['kernel'], "--steps", "1", "--size", "262144"],
                'NUMBA_CPU_TEST_CMD': ["python", wl_names[all_workloads.rambo.value]['numba'], "--steps", "1"],
                'NUMBA_CPU_PERF_CMD': ["python", wl_names[all_workloads.rambo.value]['numba']],
                'NUMBA_CPU_VTUNE_CMD': ["python", wl_names[all_workloads.rambo.value]['numba'], "--steps", "1"],
                'NUMBA_CPU_ADVISOR_CMD': ["python", wl_names[all_workloads.rambo.value]['numba'], "--steps", "1"],
                'NATIVE_TEST_CMD': ["./rambo", "1"],
                'NATIVE_PERF_CMD': ["./rambo"],
                'NATIVE_VTUNE_CMD': ["./rambo", "1", "262144"],
                'NATIVE_ADVISOR_CMD': ["./rambo", "1", "262144"],
            },
            all_workloads.gpairs.value: {
                'execute': False,
                'ref_input': 524288,
                'NUMBA_TEST_CMD': ["python", wl_names[all_workloads.gpairs.value]['numba'] if not kernel_mode else wl_names[all_workloads.gpairs.value]['kernel'], "--steps", "1"],
                'NUMBA_PERF_CMD': ["python", wl_names[all_workloads.gpairs.value]['numba'] if not kernel_mode else wl_names[all_workloads.gpairs.value]['kernel']],
                'NUMBA_VTUNE_CMD': ["python", wl_names[all_workloads.gpairs.value]['numba'] if not kernel_mode else wl_names[all_workloads.gpairs.value]['kernel'], "--steps", "1"],
                'NUMBA_ADVISOR_CMD': ["python", wl_names[all_workloads.gpairs.value]['numba'] if not kernel_mode else wl_names[all_workloads.gpairs.value]['kernel'], "--steps", "1"],
                'NUMBA_CPU_TEST_CMD': ["python", wl_names[all_workloads.gpairs.value]['numba'], "--steps", "1"],
                'NUMBA_CPU_PERF_CMD': ["python", wl_names[all_workloads.gpairs.value]['numba']],
                'NUMBA_CPU_VTUNE_CMD': ["python", wl_names[all_workloads.gpairs.value]['numba'], "--steps", "1"],
                'NUMBA_CPU_ADVISOR_CMD': ["python", wl_names[all_workloads.gpairs.value]['numba'], "--steps", "1"],
                'NATIVE_TEST_CMD': ["./gpairs", "1"],
                'NATIVE_PERF_CMD': ["./gpairs"],
                'NATIVE_VTUNE_CMD': ["./gpairs", "1"],
                'NATIVE_ADVISOR_CMD': ["./gpairs", "1"],
            },
            # all_workloads.pathfinder.value: {
            #     'execute': False,
            #     'ref_input': 2**14,
            #     'NUMBA_TEST_CMD': ["python", wl_names[all_workloads.pathfinder.value]['numba'] if not kernel_mode else wl_names[all_workloads.pathfinder.value]['kernel'], "--steps", "1", "--usm"],
            #     'NUMBA_PERF_CMD': ["python", wl_names[all_workloads.pathfinder.value]['numba'] if not kernel_mode else wl_names[all_workloads.pathfinder.value]['kernel'], "--usm"],
            #     'NUMBA_VTUNE_CMD': ["python", wl_names[all_workloads.pathfinder.value]['numba'] if not kernel_mode else wl_names[all_workloads.pathfinder.value]['kernel'], "--steps", "1", "--size", str(2**14), "--usm"],
            #     'NUMBA_ADVISOR_CMD': ["python", wl_names[all_workloads.pathfinder.value]['numba'] if not kernel_mode else wl_names[all_workloads.pathfinder.value]['kernel'], "--steps", "1", "--size", str(2**14), "--usm"],
            #     'NUMBA_CPU_TEST_CMD': ["python", wl_names[all_workloads.pathfinder.value]['numba'], "--steps", "1"],
            #     'NUMBA_CPU_PERF_CMD': ["python", wl_names[all_workloads.pathfinder.value]['numba']],
            #     'NUMBA_CPU_VTUNE_CMD': ["python", wl_names[all_workloads.pathfinder.value]['numba'], "--steps", "1", "--size", str(2**14)],
            #     'NUMBA_CPU_ADVISOR_CMD': ["python", wl_names[all_workloads.pathfinder.value]['numba'], "--steps", "1", "--size", str(2**14)],
            #     'NATIVE_TEST_CMD': ["./pathfinder", "1"],
            #     'NATIVE_PERF_CMD': ["./pathfinder"],
            #     'NATIVE_VTUNE_CMD': ["./pathfinder", "1", str(2**14), "1"],
            #     'NATIVE_ADVISOR_CMD': ["./pathfinder", "1", str(2**14), "1"],
            # },
            # all_workloads.pygbm.value: {
            #     'execute': False,
            #     'ref_input': 2**10,
            #     'NUMBA_TEST_CMD': ["python", wl_names[all_workloads.pygbm.value]['numba'] if not kernel_mode else wl_names[all_workloads.pygbm.value]['kernel'], "--steps", "1"],
            #     'NUMBA_PERF_CMD': ["python", wl_names[all_workloads.pygbm.value]['numba'] if not kernel_mode else wl_names[all_workloads.pygbm.value]['kernel']],
            #     'NUMBA_VTUNE_CMD': ["python", wl_names[all_workloads.pygbm.value]['numba'] if not kernel_mode else wl_names[all_workloads.pygbm.value]['kernel'], "--steps", "1", "--size", str(2 ** 10)],
            #     'NUMBA_ADVISOR_CMD': ["python", wl_names[all_workloads.pygbm.value]['numba'] if not kernel_mode else wl_names[all_workloads.pygbm.value]['kernel'], "--steps", "1", "--size", str(2 ** 10)],
            #     'NUMBA_CPU_TEST_CMD': ["python", wl_names[all_workloads.pygbm.value]['numba'], "--steps", "1"],
            #     'NUMBA_CPU_PERF_CMD': ["python", wl_names[all_workloads.pygbm.value]['numba']],
            #     'NUMBA_CPU_VTUNE_CMD': ["python", wl_names[all_workloads.pygbm.value]['numba'], "--steps", "1", "--size", str(2 ** 10)],
            #     'NUMBA_CPU_ADVISOR_CMD': ["python", wl_names[all_workloads.pygbm.value]['numba'], "--steps", "1", "--size", str(2 ** 10)],
            #     'SCIKIT_LEARN_TEST_CMD': ["python", "pygbm.py", "--steps", "1"],
            #     'SCIKIT_LEARN_PERF_CMD': ["python", "pygbm.py"],
            #     'SCIKIT_LEARN_VTUNE_CMD': ["python", "pygbm.py", "--steps", "1", "--size", str(2 ** 10)],
            #     'SCIKIT_LEARN_ADVISOR_CMD': ["python", "pygbm.py", "--steps", "1", "--size", str(2 ** 10)],
            #     'DAAL4PY_TEST_CMD': ["python", "pygbm.py", "--steps", "1"],
            #     'DAAL4PY_PERF_CMD': ["python", "pygbm.py"],
            #     'DAAL4PY_VTUNE_CMD': ["python", "pygbm.py", "--steps", "1", "--size", str(2 ** 10)],
            #     'DAAL4PY_ADVISOR_CMD': ["python", "pygbm.py", "--steps", "1", "--size", str(2 ** 10)],
            #     'NATIVE_TEST_CMD': ["python", "base_gbm.py", "--steps", "1"],
            #     'NATIVE_PERF_CMD': ["python", "base_gbm.py"],
            #     'NATIVE_VTUNE_CMD': ["python", "base_gbm.py", "--steps", "1", "--size", "1024"],
            #     'NATIVE_ADVISOR_CMD': ["python", "base_gbm.py", "--steps", "1", "--size", "1024"],
            #     'NATIVE_OPTIMISED_TEST_CMD': ["python", "base_gbm.py", "--steps", "1"],
            #     'NATIVE_OPTIMISED_PERF_CMD': ["python", "base_gbm.py"],
            #     'NATIVE_OPTIMISED_VTUNE_CMD': ["python", "base_gbm.py", "--steps", "1", "--size", "1024"],
            #     'NATIVE_OPTIMISED_ADVISOR_CMD': ["python", "base_gbm.py", "--steps", "1", "--size", "1024"],
            # },
            # all_workloads.random_forest.value: {
            #     'execute': False,
            #     'ref_input': 32768,
            #     'NUMBA_TEST_CMD': ["python", "random_forest.py", "--steps", "1"],
            #     'NUMBA_PERF_CMD': ["python", "random_forest.py"],
            #     'NUMBA_VTUNE_CMD': ["python", "random_forest.py", "--steps", "1", "--size", "32768"],
            #     'NUMBA_ADVISOR_CMD': ["python", "random_forest.py", "--steps", "1", "--size", "32768"],
            #     'SCIKIT_LEARN_TEST_CMD': ["python", "random_forest.py", "--steps", "1"],
            #     'SCIKIT_LEARN_PERF_CMD': ["python", "random_forest.py"],
            #     'SCIKIT_LEARN_VTUNE_CMD': ["python", "random_forest.py", "--steps", "1", "--size", "32768"],
            #     'SCIKIT_LEARN_ADVISOR_CMD': ["python", "random_forest.py", "--steps", "1", "--size", "32768"],
            #     'DAAL4PY_TEST_CMD': ["python", "random_forest.py", "--steps", "1"],
            #     'DAAL4PY_PERF_CMD': ["python", "random_forest.py"],
            #     'DAAL4PY_VTUNE_CMD': ["python", "random_forest.py", "--steps", "1", "--size", "32768"],
            #     'DAAL4PY_ADVISOR_CMD': ["python", "random_forest.py", "--steps", "1", "--size", "32768"],
            #     'NATIVE_TEST_CMD': ["python", "base_random_forest.py", "--steps", "1"],
            #     'NATIVE_PERF_CMD': ["python", "base_random_forest.py"],
            #     'NATIVE_VTUNE_CMD': ["python", "base_random_forest.py", "--steps", "1", "--size", "32768"],
            #     'NATIVE_ADVISOR_CMD': ["python", "base_random_forest.py", "--steps", "1", "--size", "32768"],
            #     'NATIVE_OPTIMISED_TEST_CMD': ["python", "base_random_forest.py", "--steps", "1"],
            #     'NATIVE_OPTIMISED_PERF_CMD': ["python", "base_random_forest.py"],
            #     'NATIVE_OPTIMISED_VTUNE_CMD': ["python", "base_random_forest.py", "--steps", "1", "--size", "32768"],
            #     'NATIVE_OPTIMISED_ADVISOR_CMD': ["python", "base_random_forest.py", "--steps", "1", "--size", "32768"],
            # },
            # all_workloads.svm.value: {
            #     'execute': False,
            #     'ref_input': 32768,
            #     'NUMBA_TEST_CMD': ["python", "svm.py", "--steps", "1"],
            #     'NUMBA_PERF_CMD': ["python", "svm.py"],
            #     'NUMBA_VTUNE_CMD': ["python", "svm.py", "--steps", "1", "--size", "32768"],
            #     'NUMBA_ADVISOR_CMD': ["python", "svm.py", "--steps", "1", "--size", "32768"],
            #     'SCIKIT_LEARN_TEST_CMD': ["python", "svm.py", "--steps", "1"],
            #     'SCIKIT_LEARN_PERF_CMD': ["python", "svm.py"],
            #     'SCIKIT_LEARN_VTUNE_CMD': ["python", "svm.py", "--steps", "1", "--size", "32768"],
            #     'SCIKIT_LEARN_ADVISOR_CMD': ["python", "svm.py", "--steps", "1", "--size", "32768"],
            #     'DAAL4PY_TEST_CMD': ["python", "svm.py", "--steps", "1"],
            #     'DAAL4PY_PERF_CMD': ["python", "svm.py"],
            #     'DAAL4PY_VTUNE_CMD': ["python", "svm.py", "--steps", "1", "--size", "32768"],
            #     'DAAL4PY_ADVISOR_CMD': ["python", "svm.py", "--steps", "1", "--size", "32768"],
            #     'NATIVE_TEST_CMD': ["python", "base_svm.py", "--steps", "1"],
            #     'NATIVE_PERF_CMD': ["python", "base_svm.py"],
            #     'NATIVE_VTUNE_CMD': ["python", "base_svm.py", "--steps", "1", "--size", "32768"],
            #     'NATIVE_ADVISOR_CMD': ["python", "base_svm.py", "--steps", "1", "--size", "32768"],
            #     'NATIVE_OPTIMISED_TEST_CMD': ["python", "base_svm.py", "--steps", "1"],
            #     'NATIVE_OPTIMISED_PERF_CMD': ["python", "base_svm.py"],
            #     'NATIVE_OPTIMISED_VTUNE_CMD': ["python", "base_svm.py", "--steps", "1", "--size", "32768"],
            #     'NATIVE_OPTIMISED_ADVISOR_CMD': ["python", "base_svm.py", "--steps", "1", "--size", "32768"],
            # },
            # all_workloads.umap.value: {
            #     'execute': False,
            #     'ref_input': 16384,
            #     'NUMBA_TEST_CMD': ["python", "umap_numba.py", "--steps", "1"],
            #     'NUMBA_PERF_CMD': ["python", "umap_numba.py"],
            #     'NUMBA_VTUNE_CMD': ["python", "umap_numba.py", "--steps", "1", "--size", str(2 ** 10)],
            #     'NUMBA_ADVISOR_CMD': ["python", "umap_numba.py", "--steps", "1", "--size", str(2 ** 10)],
            #     'NATIVE_TEST_CMD': ["./umap", "1"],
            #     'NATIVE_PERF_CMD': ["./umap"],
            #     'NATIVE_VTUNE_CMD': ["./umap", "1"],
            #     'NATIVE_ADVISOR_CMD': ["./umap", "1"],
            # },
        }

        if not input_wls:
            # iterate through all workload and set execute=True
            for val in self.wl_list.values():
                val['execute'] = True
        else:
            try:
                for input_wl in input_wls:
                    self.wl_list[input_wl]['execute'] = True
            except:
                print("Invalid workload: " + str(input_wl) + "\n")
                sys.exit(1)


class options:
    def __init__(self):
        pass
