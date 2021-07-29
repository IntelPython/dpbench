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
CC          := icc
QOPT        :=
FQOPT       :=f
EQCOLON     :="="
TARGET      := l2_distance

CFLAGS      += -g -O3
CFLAGS      += -qopt-report
#CFLAGS      += -qopt-report-phase$(EQCOLON)vec
#CFLAGS      += -$(QOPT)restrict
CFLAGS      += -qopenmp
CFLAGS      += -I./

PREC ?= f
ifeq ($(PREC),d)
else
    CFLAGS += -D__DO_FLOAT__
endif

TARGET_ARCH ?= auto

ifeq ($(TARGET_ARCH),SSE)
    CFLAGS += -$(QOPT)xSSE4.2
endif
ifeq ($(TARGET_ARCH),AVX)
    CFLAGS += -$(QOPT)xAVX
endif
ifeq ($(TARGET_ARCH),AVX2)
    CFLAGS += -$(QOPT)xCORE_AVX2
endif
ifeq ($(TARGET_ARCH),MIC)
    CFLAGS += -mmic -opt-streaming-stores always
endif
ifeq ($(TARGET_ARCH),host)
    CFLAGS += -xhost
endif
ifeq ($(TARGET_ARCH),auto)
    CFLAGS += -xCORE-AVX2 -axCOMMON-AVX512
endif

ACC ?= ha
ifeq ($(ACC),ha)
    CFLAGS += -$(FQOPT)imf-precision$(EQCOLON)high -D_VML_ACCURACY_HA_
endif
ifeq ($(ACC),la)
    CFLAGS += -$(FQOPT)imf-precision$(EQCOLON)medium -D_VML_ACCURACY_LA_
endif
ifeq ($(ACC),ep)
    CFLAGS += -$(FQOPT)imf-precision$(EQCOLON)low -$(FQOPT)imf-domain-exclusion$(EQCOLON)31 -D_VML_ACCURACY_EP_
endif

#TARGET := $(TARGET)_$(ACC)

# ==============================================================================
# ############## Define make rules #############################################
# ==============================================================================

all: l2-distance

nomkl: l2_distance
	./$(TARGET) 1024

l2-distance: $(TARGET)

$(TARGET): $(SRC) l2_distance.cpp
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $(TARGET)

clean:
	rm -rf *.o *.out *.optrpt *.yaml *~ l2_distance

.PHONY: all clean
