# Copyright (C) 2014-2015, 2018 Intel Corporation
#
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# Makefile for GNU make

# ==============================================================================
#  Content:
#      rambo formula example makefile
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
CC          := icpx
QOPT        :=
FQOPT       :=f
EQCOLON     :==
TARGET      := rambo

CFLAGS      += -g -O3
CFLAGS      += -fsycl
CFLAGS      += -I./
CFLAGS      += -std=c++14

# ==============================================================================
# ############## Define make rules #############################################
# ==============================================================================

all: rambo

comp: rambo

rambo: $(TARGET)

$(TARGET): $(SRC) rambo.cpp
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $(TARGET)

clean:
	rm -rf *.o *.out *.optrpt *.yaml *~ $(foreach acc,ha la ep,rambo) *.bin

.PHONY: all clean
