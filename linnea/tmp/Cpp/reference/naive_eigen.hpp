struct naive_eigen
{
template<typename Type_R, typename Type_L, typename Type_A, typename Type_B, typename Type_y>
decltype(auto) operator()(Type_R && R, Type_L && L, Type_A && A, Type_B && B, Type_y && y)
{
    auto x = ((((R).transpose()*L*R+(A).transpose().inverse()*(B).transpose()*B*(A).inverse())).inverse()*(A).transpose().inverse()*(B).transpose()*B*(A).inverse()*y).eval();
    
typedef std::remove_reference_t<decltype(x)> return_t;
return return_t(x);

}
};