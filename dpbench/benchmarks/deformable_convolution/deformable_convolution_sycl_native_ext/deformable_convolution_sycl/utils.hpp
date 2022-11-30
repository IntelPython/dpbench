//
// Copyright 2022 Intel Corp.
//
// SPDX - License - Identifier : Apache 2.0
///

#include <array>

template<int Dims>
class TensorShape: public std::array<int, Dims>
{
public:
    template<class DimType>
    const int& operator[](DimType dim) const
    {
        return std::array<int, Dims>::operator[](static_cast<int>(dim));
    }

    template<class DimType>
    int& operator[](DimType dim)
    {
        return std::array<int, Dims>::operator[](static_cast<int>(dim));
    }

};

enum class CHW: int
{
    C,
    H,
    W,
};

enum class NCHW: int
{
    N,
    C,
    H,
    W,
};

enum class OIHW: int
{
    OC,
    IC,
    H,
    W,
};

enum class HWCK: int
{
    H,
    W,
    C,
    KH,
    KW
};

enum class CKHW: int
{
    C,
    KH,
    KW,
    H,
    W,
};

using Shape1D = TensorShape<1>;
using Shape2D = TensorShape<2>;
using Shape3D = TensorShape<3>;
using Shape4D = TensorShape<4>;
using Shape5D = TensorShape<5>;
using DType = float;

#define get_ptr_1d(data_ptr, max_dim_0, dim_0) (data_ptr + (dim_0))
#define get_ptr_2d(data_ptr, max_dim_0, max_dim_1, dim_0, dim_1) \
(\
    get_ptr_1d(data_ptr + (dim_0)*(max_dim_1), max_dim_1, dim_1) \
)
#define get_ptr_3d(data_ptr, max_dim_0, max_dim_1, max_dim_2, dim_0, dim_1, dim_2) \
(\
    get_ptr_2d(data_ptr + (dim_0)*(max_dim_1)*(max_dim_2), max_dim_1, max_dim_2, dim_1, dim_2) \
)
#define get_ptr_4d(data_ptr, max_dim_0, max_dim_1, max_dim_2, max_dim_3, dim_0, dim_1, dim_2, dim_3) \
(\
    get_ptr_3d(data_ptr + (dim_0)*(max_dim_1)*(max_dim_2)*(max_dim_3), max_dim_1, max_dim_2, max_dim_3, dim_1, dim_2, dim_3) \
)

#define get_ptr_5d(data_ptr, max_dim_0, max_dim_1, max_dim_2, max_dim_3, max_dim_4, dim_0, dim_1, dim_2, dim_3, dim_4) \
(\
    get_ptr_4d(data_ptr + (dim_0)*(max_dim_1)*(max_dim_2)*(max_dim_3)*(max_dim_4), \
               max_dim_1, max_dim_2, max_dim_3, max_dim_4, \
               dim_1, dim_2, dim_3, dim_4) \
)
