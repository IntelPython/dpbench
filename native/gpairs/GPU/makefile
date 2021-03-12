# Copyright (C) 2014-2015, 2018 Intel Corporation
#
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# Makefile for GNU make

SRC:=                      \
       data_gen.cpp          \
       main.cpp


# ==============================================================================
# ############## Configure CFLAGS  #############################################
# ==============================================================================
CC          := icx
QOPT        :=
FQOPT       :=f
EQCOLON     :="="
TARGET      := gpairs

CFLAGS      += -g -O3
CFLAGS      += -qopt-report
CFLAGS      += -fiopenmp -fopenmp-targets=spir64
CFLAGS      += -I./

PREC ?= d
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

#TARGET := $(TARGET)

# ==============================================================================
# ############## Define make rules #############################################
# ==============================================================================

all: $(TARGET)

$(TARGET): $(SRC) gpairs.cpp
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $(TARGET)

clean:
	rm -rf *.o *.out *.optrpt *.yaml *~ gpairs

.PHONY: all clean
