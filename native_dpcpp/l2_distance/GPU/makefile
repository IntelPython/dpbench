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

# ==============================================================================
# ############## Configure CFLAGS  #############################################
# ==============================================================================
CC          := icpx
QOPT        :=
FQOPT       :=f
EQCOLON     :="="
TARGET      := l2_distance

CFLAGS      += -g -O3
CFLAGS      += -fsycl
CFLAGS      += -I./
CFLAGS      += -DSYCL_USE_NATIVE_FP_ATOMICS

PREC ?= f
ifeq ($(PREC),d)
else
    CFLAGS += -D__DO_FLOAT__
endif

# ==============================================================================
# ############## Define make rules #############################################
# ==============================================================================

all: l2-distance

l2-distance: $(TARGET)

comp: $(TARGET)_comp

atomic: $(TARGET)_atomic

atomic_comp: $(TARGET)_atomic_comp

$(TARGET): data_gen.cpp main.cpp l2_distance.cpp
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $(TARGET)

$(TARGET)_comp: main.cpp data_gen_comp.cpp l2_distance_comp.cpp
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $(TARGET)_comp

$(TARGET)_atomic: main.cpp data_gen.cpp l2_distance_atomic.cpp
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $(TARGET)_atomic

$(TARGET)_atomic_comp: main.cpp data_gen_comp.cpp l2_distance_atomic_comp.cpp
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $(TARGET)_atomic_comp


clean:
	rm -rf *.o *.out *.optrpt *.yaml *~ l2_distance l2_distance_comp l2_distance_atomic l2_distance_atomic_comp *.bin

.PHONY: all clean
