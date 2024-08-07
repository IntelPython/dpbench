# Copyright (C) 2014-2015, 2018 Intel Corporation
#
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# Makefile for GNU make

# ==============================================================================
# ############## Configure CFLAGS  #############################################
# ==============================================================================
CC          := icpx
QOPT        :=
FQOPT       :=f
EQCOLON     :="="
TARGET      := gpairs

CFLAGS      += -O3
CFLAGS      += -fsycl
CFLAGS      += -I./
CFLAGS      += -DSYCL_USE_NATIVE_FP_ATOMICS

PREC ?= d
ifeq ($(PREC),d)
    CFLAGS += -D__DO_DOUBLE__
else
    CFLAGS += -D__DO_FLOAT__
endif

# ==============================================================================
# ############## Define make rules #############################################
# ==============================================================================

all: $(TARGET)

$(TARGET): data_gen.cpp main.cpp
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $(TARGET)

clean:
	rm -rf *.o *.out *.optrpt *.yaml *~ gpairs

.PHONY: all clean
