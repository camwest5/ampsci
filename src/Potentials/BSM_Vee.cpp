#pragma once
#include "Angular/Wigner369j.hpp"
#include "DiracOperator/Operators/V_SP.hpp"
#include "IO/InputBlock.hpp"
#include "Wavefunction/Wavefunction.hpp"

namespace Vee {
DiracSpinor V_SP_Fv(const std::vector<DiracSpinor> &core, const DiracSpinor &Fv,
                    const int kappa_n, const double y, const double mu) {

  // For safety
  if (Fv.kappa() != -kappa_n) {
    return 0.0 * Fv;
  }

  DiracSpinor VFv(Fv.n(), Fv.kappa(), Fv.grid_sptr());

  for (auto Fa : core) {

    const auto Fdir = Fa.twojp1() * Bk_ac_v(0, mu, Fa, Fa, Fv);

    DiracSpinor Fexch(Fa.n(), Fa.kappa(), Fa.grid_sptr());

    for (int twok = std::abs(Fa.twoj() - Fv.twoj());
         twok <= Fa.twoj() + Fv.twoj(); twok += 2) {
      if ((Fa.twoj() + Fv.twoj() + twok) % 4 == 0) {
        const double k = 0.5 * twok;
        const double twokp1 = twok + 1;

        const auto Fexch_1 = Angular::Ck_kk(k, -Fv.kappa(), Fa.kappa()) *
                             Angular::Ck_kk(k, Fa.kappa(), -Fv.kappa()) *
                             bk_bd_v(k, mu, Fa, Fv, Fa);

        const auto Fexch_2 = Angular::Ck_kk(k, Fa.kappa(), Fv.kappa()) *
                             Angular::Ck_kk(k, -Fv.kappa(), -Fa.kappa()) *
                             Bk_ac_v(k, mu, Fa, Fv, Fa);

        Fexch += twokp1 * (Fexch_1 + Fexch_2);
      }
    }

    const auto phase = Angular::neg1pow_2(Fv.twoj() - Fa.twoj()) / Fv.twojp1();

    Fexch *= phase;

    VFv += Fdir - Fexch;
  }

  return y * VFv;
}

DiracSpinor Bk_ac_v(const int k, const double mu, const DiracSpinor &Fa,
                    const DiracSpinor &Fc, const DiracSpinor &Fv) {
  const auto &gr = Fa.grid();
  const auto &r = gr.r();

  // Modified spherical Bessel functions
  std::vector<double> i_k(gr.size());
  std::vector<double> k_k(gr.size());

  for (int i_gr = 0; i_gr < gr.size(); ++i_gr) {
    const auto x = mu * r[i_gr];
    i_k[i_gr] = mod_sph_bessel_i(k, x);
    k_k[i_gr] = mod_sph_bessel_k(k, x);
  }

  const auto g0Fc = g0(Fc);

  // Integrate
  std::vector<double> result(gr.size());

  for (int i_mid = 0; i_mid < gr.size(); ++i_mid) {
    double lower_ff =
        NumCalc::integrate(1.0, 0, i_mid, i_k, Fa.f(), g0Fc.f(), gr.drdu());

    double lower_gg =
        NumCalc::integrate(1.0, 0, i_mid, i_k, Fa.g(), g0Fc.g(), gr.drdu());

    // For r0 point
    if (i_mid == 0) {
      lower_ff = 0;
      lower_gg = 0;
    }

    const double upper_ff = NumCalc::integrate(1.0, i_mid, gr.size(), k_k,
                                               Fa.f(), g0Fc.f(), gr.drdu());

    const double upper_gg = NumCalc::integrate(1.0, i_mid, gr.size(), k_k,
                                               Fa.g(), g0Fc.g(), gr.drdu());

    result[i_mid] = (k_k[i_mid] * (lower_ff + lower_gg) +
                     i_k[i_mid] * (upper_ff + upper_gg)) *
                    gr.du();
  }

  return result * i_g0_g5(Fv);
}

DiracSpinor bk_bd_v(const int k, const double mu, const DiracSpinor &Fb,
                    const DiracSpinor &Fd, const DiracSpinor &Fv) {
  const auto &gr = Fb.grid();
  const auto &r = gr.r();

  // Modified spherical Bessel functions
  std::vector<double> i_k(gr.size());
  std::vector<double> k_k(gr.size());

  for (int i_gr = 0; i_gr < gr.size(); ++i_gr) {
    const auto x = mu * r[i_gr];
    i_k[i_gr] = mod_sph_bessel_i(k, x);
    k_k[i_gr] = mod_sph_bessel_k(k, x);
  }

  const auto ig0g5Fd = i_g0_g5(Fd);

  // Integrate
  std::vector<double> result(gr.size());

  for (int i_mid = 0; i_mid < gr.size(); ++i_mid) {
    double lower_ff =
        NumCalc::integrate(1.0, 0, i_mid, i_k, Fb.f(), ig0g5Fd.f(), gr.drdu());

    double lower_gg =
        NumCalc::integrate(1.0, 0, i_mid, i_k, Fb.g(), ig0g5Fd.g(), gr.drdu());

    // For r0 point
    if (i_mid == 0) {
      lower_ff = 0;
      lower_gg = 0;
    }

    const double upper_ff = NumCalc::integrate(1.0, i_mid, gr.size(), k_k,
                                               Fb.f(), ig0g5Fd.f(), gr.drdu());

    const double upper_gg = NumCalc::integrate(1.0, i_mid, gr.size(), k_k,
                                               Fb.g(), ig0g5Fd.g(), gr.drdu());

    result[i_mid] = (k_k[i_mid] * (lower_ff + lower_gg) +
                     i_k[i_mid] * (upper_ff + upper_gg)) *
                    gr.du();
  }

  return result * g0(Fv);
}

DiracSpinor g0(const DiracSpinor &Fa) {
  DiracSpinor Fb(Fa);
  Fb.g() = (-1.0 * Fa).g();
  return Fb;
}

DiracSpinor i_g0_g5(const DiracSpinor &Fa) {
  // Fb = i*γ5*γ0*Fa

  DiracSpinor Fb(Fa);
  Fb.f() = (-1.0 * Fa).g();
  Fb.g() = (-1.0 * Fa).f();
  return Fb;
}

double mod_sph_bessel_i(double n, double x) {
  gsl_sf_result i_k;
  const int gsl_status = gsl_sf_bessel_Inu_e(n + 0.5, x, &i_k);

  if (gsl_status == GSL_SUCCESS) {
    /*if (i_k.err / i_k.val > 0.01) {
      std::cout << "\nWARNING: error in i_k greater than 1\%\ni_k = " << i_k.val
                << " \u00b1 " << i_k.err << "\n";
    }*/
    return std::sqrt(M_PI / (2.0 * x)) * i_k.val;
  } else if (gsl_status == GSL_EOVRFLW) {
    return 0.0;
  }
  std::cout << "Need GSL_SUCCESS = " << GSL_SUCCESS
            << " or GSL_EOVRFLW = " << GSL_EOVRFLW;
  std::cout << "\n\ngsl_status = " << gsl_status << "\n";

  throw std::bad_function_call();
}

double mod_sph_bessel_k(double n, double x) {
  gsl_sf_result k_k;
  const int gsl_status = gsl_sf_bessel_Knu_e(n + 0.5, x, &k_k);

  if (gsl_status == GSL_SUCCESS) {
    /*
    if (k_k.err / k_k.val > 0.01) {
      std::cout << "\nWARNING: error in i_k greater than 1\%\n " << k_k.val
                << " \u00b1 " << k_k.err << "\n";
    }*/
    return std::sqrt(2.0 / (M_PI * x)) * k_k.val;
  } else if (gsl_status == GSL_EUNDRFLW) {
    return 0.0;
  }
  std::cout << "Need GSL_SUCCESS = " << GSL_SUCCESS
            << " or GSL_EUNDRFLW = " << GSL_EUNDRFLW;
  std::cout << "\n\ngsl_status = " << gsl_status << "\n";

  throw std::bad_function_call();
}

} // namespace Vee