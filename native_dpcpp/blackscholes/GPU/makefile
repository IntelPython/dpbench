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
TARGET      := black_scholes

CFLAGS      += -g -O3
CFLAGS      += -fsycl
CFLAGS      += -I./

PREC ?= d
ifeq ($(PREC),d)
else
    CFLAGS += -D__DO_FLOAT__
endif

# ==============================================================================
# ############## Define make rules #############################################
# ==============================================================================

all: black_scholes

black_scholes: $(TARGET)

comp: $(TARGET)_comp

$(TARGET): data_gen.cpp main.cpp black-scholes.cpp
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $(TARGET)

$(TARGET)_comp: main_comp.cpp data_gen_comp.cpp black-scholes_comp.cpp
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $(TARGET)_comp

clean:
	rm -rf *.o *.out *.optrpt *.yaml *.bin *~ $(foreach acc,ha la ep,black_scholes black_scholes_comp)

.PHONY: all clean comp black_scholes
