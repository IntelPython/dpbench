# clang++ -Wall -w -I./source/utils -fsycl ./source/gradient_boosted_trees/gbt_reg_dense_batch.cpp -o _results/clang_intel64_parallel_so/gbt_reg_dense_batch.exe -Wl,--start-group -Wl,-L"/localdisk/work/adarshyoga/software/daal/latest/lib/intel64",-Bdynamic,-ldaal_core,-Bdynamic -Wl,-Bdynamic,-ldaal_thread,-Bdynamic -lpthread -ldl -lOpenCL -L"/opt/intel/inteloneapi/tbb/2021.1-beta06/env/../lib/intel64/gcc4.8" -ltbb -ltbbmalloc "/localdisk/work/adarshyoga/software/daal/latest/lib/intel64"/libdaal_sycl.a  -Wl,--end-group

#_results/clang_intel64_parallel_so/gbt_reg_dense_batch.exe > _results/clang_intel64_parallel_so/gbt_reg_dense_batch.res


# Copyright (C) 2014-2015, 2018 Intel Corporation
#
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# Makefile for GNU make

# ==============================================================================
#  Content:
#      Black-Scholes formula example makefile
# ==============================================================================
#
#    Parameters of this makefile:
#    ----------------------------
#
#      TARGET_ARCH= SSE, AVX <default>, AVX2, MIC
#
#      PREC= s (float) <default>, d (double)
#
#      ACC=ha, la, ep <default> : meaning math function accuracy level
# ==============================================================================

SRC:= main.cpp


# ==============================================================================
# ############## Configure CFLAGS  #############################################
# ==============================================================================
CC          := clang++
QOPT        :=
FQOPT       :=f
EQCOLON     :="="
TARGET      := pca

CFLAGS      += -g -O3
CFLAGS      += -Wall -w
CFLAGS      += -I./

DAAL_PATH = $(DAALROOT)/lib/intel64
override threading = parallel
DAAL_LIB_T := -ldaal_thread
DAAL_LIB := -L$(DAAL_PATH),-ldaal_core -Wl,$(DAAL_LIB_T)

TBB_PATH = $(TBBROOT)/lib/intel64/gcc4.8
EXT_LIB += -lpthread -ldl $(addprefix -L,$(TBB_PATH)) -ltbb -ltbbmalloc

LFLAGS := -Wl,--start-group -Wl,$(DAAL_LIB),$(EXT_LIB) $(DAAL_PATH)/libdaal_thread.a -Wl,--end-group

PREC ?= d
ifeq ($(PREC),d)
else
    CFLAGS += -D__DO_FLOAT__
endif

# ==============================================================================
# ############## Define make rules #############################################
# ==============================================================================

all: pca-gpu

pca-gpu: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $^ -o $(TARGET) $(CPPFLAGS) $(CFLAGS) $(LFLAGS)

clean:
	rm -rf *.o *.out *.optrpt *.yaml *~ pca

.PHONY: all clean
