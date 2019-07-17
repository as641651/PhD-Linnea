#include <generator/generator.hpp>

template<typename Gen>
decltype(auto) operand_generator(Gen && gen)
{
    auto R = gen.generate({1999,2000}, generator::shape::upper_triangular{}, generator::property::random{}, generator::shape::not_square{});
    auto L = gen.generate({1999,1999}, generator::shape::self_adjoint{}, generator::shape::diagonal{}, generator::shape::lower_triangular{}, generator::shape::upper_triangular{}, generator::property::random{});
    auto A = gen.generate({2000,2000}, generator::property::random{});
    auto B = gen.generate({2000,2000}, generator::property::random{});
    auto y = gen.generate({2000,1}, generator::property::random{}, generator::shape::col_vector{}, generator::shape::not_square{});
    return std::make_tuple(R, L, A, B, y);
}