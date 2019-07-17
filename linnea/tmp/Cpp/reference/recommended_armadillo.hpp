struct recommended_armadillo
{
template<typename Type_R, typename Type_L, typename Type_A, typename Type_B, typename Type_y>
decltype(auto) operator()(Type_R && R, Type_L && L, Type_A && A, Type_B && B, Type_y && y)
{
    auto x = (arma::solve((arma::solve((A).t(), (B).t(), arma::solve_opts::fast)*B*(A).i()+(R).t()*L*R), arma::solve((A).t(), (B).t(), arma::solve_opts::fast), arma::solve_opts::fast)*B*arma::solve(A, y, arma::solve_opts::fast)).eval();
    
typedef std::remove_reference_t<decltype(x)> return_t;
return return_t(x);

}
};