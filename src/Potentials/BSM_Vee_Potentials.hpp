#pragma once
#include "Angular/Wigner369j.hpp"
#include "DiracOperator/Operators/Vee.hpp"
#include "IO/InputBlock.hpp"
#include "Wavefunction/Wavefunction.hpp"

namespace BSM_Vee {
DiracSpinor V_Fv(const std::vector<DiracSpinor> &core, const DiracSpinor &Fv,
                 const std::string type, const int kappa_n, const double y,
                 const double mu);
DiracSpinor Bk_ab_v(const int k, const double mu, const bool betaalpha,
                    const std::string type, const DiracSpinor &Fa,
                    const DiracSpinor &Fc, const DiracSpinor &Fv);
DiracSpinor bk_bd_v(const int k, const double mu, const std::string type,
                    const DiracSpinor &Fb, const DiracSpinor &Fd,
                    const DiracSpinor &Fv);
DiracSpinor g0(const DiracSpinor &Fa);
DiracSpinor g5(const DiracSpinor &Fa);
DiracSpinor ig5(const DiracSpinor &Fa);
DiracSpinor i_g0_g5(const DiracSpinor &Fa);
double mod_sph_bessel_i(double n, double x);
double mod_sph_bessel_k(double n, double x);

} // namespace BSM_Vee