struct naive_armadillo
{
template<typename Type_R, typename Type_L, typename Type_A, typename Type_B, typename Type_y>
decltype(auto) operator()(Type_R && R, Type_L && L, Type_A && A, Type_B && B, Type_y && y)
{
    auto x = ((((R).t()*L*R+(A).t().i()*(B).t()*B*(A).i())).i()*(A).t().i()*(B).t()*B*(A).i()*y).eval();
    
typedef std::remove_reference_t<decltype(x)> return_t;
return return_t(x);

}
};