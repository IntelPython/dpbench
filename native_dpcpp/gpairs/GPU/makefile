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

CFLAGS      += -g -O3
CFLAGS      += -fsycl
CFLAGS      += -I./
CFLAGS      += -DSYCL_USE_NATIVE_FP_ATOMICS

PREC ?= d
ifeq ($(PREC),d)
else
    CFLAGS += -D__DO_FLOAT__
endif

# ==============================================================================
# ############## Define make rules #############################################
# ==============================================================================

all: $(TARGET)

comp: $(TARGET)_comp

$(TARGET): data_gen.cpp main.cpp gpairs.cpp
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $(TARGET)

$(TARGET)_comp: data_gen_comp.cpp main_comp.cpp gpairs_comp.cpp
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $(TARGET)_comp

clean:
	rm -rf *.o *.out *.optrpt *.yaml *~ gpairs gpairs_comp *.bin

.PHONY: all clean
