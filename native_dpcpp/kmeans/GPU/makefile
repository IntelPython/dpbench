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

SRC:=                      \
       data_gen.cpp          \
       main.cpp


# ==============================================================================
# ############## Configure CFLAGS  #############################################
# ==============================================================================
CC          := icpx
QOPT        :=
FQOPT       :=f
EQCOLON     :="="
TARGET      := kmeans

CFLAGS      += -g -O3
CFLAGS      += -fsycl -qopenmp
CFLAGS      += -I./

PREC ?= d
ifeq ($(PREC),d)
else
    CFLAGS += -D__DO_FLOAT__
endif

# ==============================================================================
# ############## Define make rules #############################################
# ==============================================================================

all: kmeans-gpu

kmeans-gpu: $(TARGET)

comp:$(TARGET)_comp

$(TARGET): data_gen.cpp main.cpp kmeans.cpp
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $(TARGET)

$(TARGET)_comp: data_gen_comp.cpp main.cpp kmeans_comp.cpp
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $(TARGET)_comp

clean:
	rm -rf *.o *.out *.optrpt *.yaml *~ kmeans kmeans_comp

.PHONY: all clean
