/*
 * Copyright (C) 2014-2015, 2018 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef __RDTSC_H
#define __RDTSC_H

#include <sys/time.h>
#include <stdio.h>
#include <unistd.h>

#if defined(__ICC)

typedef unsigned __int64 rdtsc_type;

static rdtsc_type timer_rdtsc(void)
{
    return __rdtsc();
}

#else

#if defined(__i386__)

typedef unsigned long long int rdtsc_type;

static rdtsc_type timer_rdtsc(void)
{
    rdtsc_type x;
    __asm__ volatile (".byte 0x0f, 0x31" : "=A" (x));
    return x;
}

#elif defined(__x86_64__)

typedef unsigned long long int rdtsc_type;

static rdtsc_type timer_rdtsc(void)
{
    unsigned hi, lo;
    __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
    return ( (rdtsc_type)lo)|( ((rdtsc_type)hi)<<32 );
}

#else

#error "THIS ARCH IS NOT SUPPORTED"

#endif

#endif



static double countGHz()
{
        struct timezone tz;
        struct timeval tvstart, tvstop;
        rdtsc_type rdtsc1, rdtsc2;
        double dsec;

        gettimeofday(&tvstart, &tz);
        rdtsc1 = timer_rdtsc();

        usleep(10000);

        rdtsc2 = timer_rdtsc();
        gettimeofday(&tvstop, &tz);

        dsec = (tvstop.tv_sec-tvstart.tv_sec) + (tvstop.tv_usec-tvstart.tv_usec)/(1000000.0);

        return (double)(rdtsc2-rdtsc1) / dsec / 1000000000.0;
}

static double getHz()
{
    double GHz;

    do
    {
        GHz = countGHz();
        //printf("%lf\n", GHz);
    } while(GHz < 0.1 || GHz > 10.0);

    // fprintf(stderr, "[GHz:%.2lf] ", GHz);

    return GHz * 1000000000.0;
}

#endif /* __RDTSC_H */
