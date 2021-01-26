#ifndef _DBSCAN_COMMON_H
#define _DBSCAN_COMMON_H

/*
Copyright (c) 2020, Intel Corporation

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
 * Neither the name of Intel Corporation nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <cstdio>
#include <string>
#include <cstdlib>
#include <ctime>
#include <sstream>
#include <cstring>
#include <chrono>

using namespace std;

struct Queue
{
    static const size_t defaultSize = 10;

    size_t *values;

    size_t capacity;
    size_t head;
    size_t tail;

    Queue(size_t cap = defaultSize)
    {
        capacity = cap;
        head = tail = 0;
        values = new size_t[capacity];
    }

    ~Queue()
    {
        delete [] values;
    }

    void resize(size_t newCapacity)
    {
        size_t *newValues = new size_t[newCapacity];

        memcpy(newValues, values, sizeof(size_t) * tail);

        delete [] values;

        capacity = newCapacity;
        values = newValues;
    }

    void push(size_t val)
    {
        if (tail == capacity)
        {
            resize(2 * capacity);
        }

        values[tail] = val;
        tail++;
    }

    inline size_t pop()
    {
        if (head < tail)
        {
            head++;
            return values[head - 1];
        }

        return -1;
    }

    inline bool empty()
    {
        return head == tail;
    }

    inline size_t getSize()
    {
        return tail - head;
    }
};

const int NOISE = -1;
const int UNDEFINED = -2;

#endif
