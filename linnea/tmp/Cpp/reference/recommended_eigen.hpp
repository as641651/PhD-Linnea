struct recommended_eigen
{
template<typename Type_R, typename Type_L, typename Type_A, typename Type_B, typename Type_y>
decltype(auto) operator()(Type_R && R, Type_L && L, Type_A && A, Type_B && B, Type_y && y)
{
    auto x = (( ((( ((A).transpose()).partialPivLu().solve((B).transpose()) )*B*(A).inverse()+(R).transpose()*L*R)).partialPivLu().solve(( ((A).transpose()).partialPivLu().solve((B).transpose()) )) )*B*( (A).partialPivLu().solve(y) )).eval();
    
typedef std::remove_reference_t<decltype(x)> return_t;
return return_t(x);

}
};