#pragma once
#include "IO/InputBlock.hpp"
#include "Wavefunction/Wavefunction.hpp"

namespace Module {
void update_Dv(double &Dv, const std::vector<DiracSpinor> core, DiracSpinor &Fv,
               const DiracSpinor &Fn, const Grid &gr, const double y,
               const double mu, const bool contact);
double d_ab(const Grid &gr, const DiracSpinor &Fa, const DiracSpinor &Fb);
double V_nv(const bool contact, const std::vector<DiracSpinor> core,
            const DiracSpinor &Fv, const DiracSpinor &Fn, const double y,
            const double mu);
void sps(const IO::InputBlock &input, const Wavefunction &wf);
double R_abcd_contact(const DiracSpinor &Fa, const DiracSpinor &Fb,
                      const DiracSpinor &Fc, const DiracSpinor &Fd);
double Rk_abcd(const double k, const double mu, const DiracSpinor &Fa,
               const DiracSpinor &Fb, const DiracSpinor &Fc,
               const DiracSpinor &Fd);
double Rk_abcd_massless(const double k, const DiracSpinor &Fa,
                        const DiracSpinor &Fb, const DiracSpinor &Fc,
                        const DiracSpinor &Fd);
std::vector<double> Bk_ab(const double k, const double mu,
                          const DiracSpinor &Fa, const DiracSpinor &Fb);
double mod_sph_bessel_i(double n, double x);
double mod_sph_bessel_k(double n, double x);
DiracSpinor i_gamma_5(const DiracSpinor &Fa);
void sps_testing(const Wavefunction &wf, const bool contact);
} // namespace Module
