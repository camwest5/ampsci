#include "Modules/sps.hpp"
#include "Angular/Wigner369j.hpp"
#include "DiracOperator/Operators/Ek.hpp"
#include "IO/InputBlock.hpp"
#include "Maths/NumCalc_quadIntegrate.hpp"
#include "Maths/SphericalBessel.hpp"
#include "Physics/PhysConst_constants.hpp" // For GHz unit conversion
#include "Wavefunction/Wavefunction.hpp"
#include "ampsci.hpp"
#include <cmath>
#include <gsl/gsl_sf.h>

namespace Module {

void sps(const IO::InputBlock &input, const Wavefunction &wf) {

  input.check({{"", "Introduces a new scalar-psuedoscalar electron-electron "
                    "interaction."},
               {"contact", "Consider μ->infty, i.e. a contact force [true]"},
               {"test", "Run module testing [false]"}});
  // If we are just requesting 'help', don't run module:
  if (input.has_option("help")) {
    return;
  }

  const bool contact = input.get<bool>("contact", true);

  if (input.get<bool>("test", false) == true) {
    sps_testing(wf, contact);
    return;
  }

  const double y_sps = 1;

  const double mu = contact == true ? 1 : 1;

  std::cout << "Fv\tDv\n";

  for (auto Fv : wf.valence()) {
    double Dv = 0;

    for (auto Fn : wf.core()) {
      if ((Fn.twoj() == Fv.twoj()) && (Fn != Fv)) {
        // <v|d|n>
        const auto d_vn = d_ab(wf.grid(), Fv, Fn);

        // <n|V|v> = Σ_a (u_nava - u_anva)
        const auto V_nv_core = V_nv(contact, wf.core(), Fv, Fn, y_sps, mu);

        Dv += 2.0 * d_vn * V_nv_core / (Fv.en() - Fn.en());
      }
    }

    for (auto Fn : wf.valence()) {
      if ((Fn.twoj() == Fv.twoj()) && (Fn != Fv)) {
        // <v|d|n>
        const auto d_vn = d_ab(wf.grid(), Fv, Fn);

        // <n|V|v> = Σ_a (u_nava - u_anva)
        const auto V_nv_core = V_nv(contact, wf.core(), Fv, Fn, y_sps, mu);

        Dv += 2.0 * d_vn * V_nv_core / (Fv.en() - Fn.en());
      }
    }

    std::cout << Fv.shortSymbol() << "\t" << Dv << "\n";
  }
}

double d_ab(const Grid &gr, const DiracSpinor &Fa, const DiracSpinor &Fb) {
  return DiracOperator::E1(gr).fullME(Fa, Fb);

  /* Manual - seems to match E1

  const auto direct_d_vn =
      -Angular::threej_2(Fv.twoj(), Fv.twoj(), 2, 1, -1, 0) *
      Angular::Ck_kk(1, Fv.kappa(), Fn.kappa()) *
      d.radialIntegral(Fv, Fn);

  */
}

double V_nv(const bool contact, const std::vector<DiracSpinor> core,
            const DiracSpinor &Fv, const DiracSpinor &Fn, const double y,
            const double mu) {

  auto u_nava = 0.0;
  auto u_anva = 0.0;

  for (auto Fa : core) {
    if (Fn.kappa() == -Fv.kappa()) {
      const auto R0_nava = contact == true ? R_abcd_contact(Fn, Fa, Fv, Fa) :
                                             Rk_abcd(0, mu, Fn, Fa, Fv, Fa);
      u_nava += R0_nava * Fa.twojp1();
    }

    const int ja = 0.5 * Fa.twoj();
    const int jv = 0.5 * Fv.twoj();

    for (int k = std::abs(ja - jv); k <= ja + jv; ++k) {
      if ((ja + jv + k) % 2 == 0) {
        const auto A_anva = (2.0 * k + 1) *
                            Angular::Ck_kk(k, Fa.kappa(), -Fv.kappa()) *
                            Angular::Ck_kk(k, Fn.kappa(), Fa.kappa());
        const auto R_anva = contact == true ? R_abcd_contact(Fa, Fn, Fv, Fa) :
                                              Rk_abcd(k, mu, Fa, Fn, Fv, Fa);
        u_anva += A_anva * R_anva;
      }
    }

    if ((jv - ja) % 2 == 1) {
      u_anva *= -1.0;
    }

    u_anva *= 1.0 / Fv.twojp1();
  }

  return (u_nava - u_anva) * y * mu;
}

double Rk_abcd(const double k, const double mu, const DiracSpinor &Fa,
               const DiracSpinor &Fb, const DiracSpinor &Fc,
               const DiracSpinor &Fd) {
  // Compare with yk_ab to find r> and r< functions
  // Then create radial operator?
  // Find the Rk_abcd implementation in code.

  const auto i0 = std::max(Fa.min_pt(), Fc.min_pt());
  const auto imax = std::min(Fa.max_pt(), Fc.max_pt());

  const auto screening_function = Bk_ab(k, mu, Fb, Fd);

  const auto ig5_Fc = i_gamma_5(Fc);

  const auto Rff = NumCalc::integrate(1.0, i0, imax, Fa.f(), ig5_Fc.f(),
                                      screening_function, Fa.grid().drdu());

  const auto Rgg = NumCalc::integrate(1.0, i0, imax, Fa.g(), ig5_Fc.g(),
                                      screening_function, Fa.grid().drdu());

  return (Rff + Rgg) * Fa.grid().du();
}

double R_abcd_contact(const DiracSpinor &Fa, const DiracSpinor &Fb,
                      const DiracSpinor &Fc, const DiracSpinor &Fd) {

  const auto ig5_Fc = i_gamma_5(Fc);
  const auto &gr = Fa.grid();

  std::vector<double> integrand(gr.size());
  for (double i = 0; i < gr.size(); ++i) {
    integrand[i] = (Fa.f(i) * ig5_Fc.f(i) + Fa.g(i) * ig5_Fc.g(i)) *
                   (Fb.f(i) * Fd.f(i) + Fb.g(i) * Fd.g(i));
  }

  const auto i0 = std::max(std::max(Fa.min_pt(), Fb.min_pt()),
                           std::max(Fc.min_pt(), Fd.min_pt()));
  const auto imax = std::min(std::min(Fa.max_pt(), Fb.max_pt()),
                             std::min(Fc.max_pt(), Fd.max_pt()));

  // const auto df = NumCalc::derivative(integrand, gr.drdu(), gr.du());

  return NumCalc::integrate(1.0, i0, imax, integrand, gr.drdu()) * gr.du();
}

std::vector<double> Bk_ab(const double k, const double mu,
                          const DiracSpinor &Fa, const DiracSpinor &Fb) {
  const auto &gr = Fa.grid();
  const auto &r = gr.r();
  const auto i0 = std::max(Fa.min_pt(), Fb.min_pt());
  const auto imax = std::min(Fa.max_pt(), Fb.max_pt());

  // Modified spherical Bessel functions
  std::vector<double> i_k(gr.size());
  std::vector<double> k_k(gr.size());

  for (int i_gr = 0; i_gr < gr.size(); ++i_gr) {
    const auto x = mu * r[i_gr];
    i_k[i_gr] = std::sqrt(M_PI / (2.0 * x)) * gsl_sf_bessel_Inu(k + 0.5, x);
    k_k[i_gr] = std::sqrt(2.0 / (M_PI * x)) * gsl_sf_bessel_Knu(k + 0.5, x);

    // For testing
    //i_k[i_gr] = 1.0;
    //k_k[i_gr] = 1.0;
  }

  // Integrate
  std::vector<double> result(gr.size());

  for (int i_mid = 0; i_mid < gr.size(); ++i_mid) {
    const double lower_ff =
        NumCalc::integrate(1.0, i0, i_mid, i_k, Fa.f(), Fb.f(), gr.drdu());

    const double lower_gg =
        NumCalc::integrate(1.0, i0, i_mid, i_k, Fa.g(), Fb.g(), gr.drdu());

    const double upper_ff =
        NumCalc::integrate(1.0, i_mid, imax, k_k, Fa.f(), Fb.f(), gr.drdu());
    const double upper_gg =
        NumCalc::integrate(1.0, i_mid, imax, k_k, Fa.g(), Fb.g(), gr.drdu());

    result[i_mid] = (k_k[i_mid] * (lower_ff + lower_gg) +
                     i_k[i_mid] * (upper_ff + upper_gg)) *
                    gr.du();
  }

  return result;
}

DiracSpinor i_gamma_5(const DiracSpinor &Fa) {
  // Fb = i*γ5*Fa
  DiracSpinor Fb(Fa);
  Fb.f() = (-1 * Fa).g();
  Fb.g() = Fa.f();
  return Fb;
}

void sps_testing(const Wavefunction &wf, const bool contact) {

  // Check E1
  const auto Fv0 = wf.valence()[0];
  const auto Fv1 = wf.valence()[1];
  const auto Fa0 = wf.core()[0];
  const auto Fa1 = wf.core()[1];

  // <v|d|v> - should be 0
  const auto vdv = d_ab(wf.grid(), Fv0, Fv0);

  // <v0|d|v1> and <v1|d|v0> - should be symmetric
  const auto v0dv1 =
      d_ab(wf.grid(), Fv0, Fv1); // should be 5.144 for Fr (<7s_1/2|d|7p_1/2>)
  const auto v1dv0 = d_ab(wf.grid(), Fv1, Fv0);

  // <v0||d||v1> - RME should be 5.144 for Fr (<7s_1/2||d||7p_1/2), https://arxiv.org/pdf/2212.11490 Table II
  const auto z_ab =
      v0dv1 / (std::pow(-1.0, (Fv0.twoj() - 1) * 0.5) *
               Angular::threej_2(Fv0.twoj(), 2, Fv1.twoj(), -1, 0, 1));

  // Check it's obeying angular selection rules?

  std::cout << "\nDipole operator checks for v0 = " << Fv0.symbol()
            << " and v1 = " << Fv1.symbol() << "\n<v0|d|v0> = " << vdv
            << "\t\t=0?\n<v0|d|v1> = " << v0dv1
            << "\t=<v1|d|v0>?\n<v1|d|v0> = " << v1dv0
            << "\t=<v0|d|v1>?\n<v0||d||v1> = " << z_ab << "\tif Fr, =5.144?\n";

  DiracSpinor F1(Fv0);
  std::vector<double> ones(wf.grid().size(), 1.0 / std::sqrt(2.0));
  F1.f() = ones;
  F1.g() = ones;

  if (contact) {
    // Check R_abcd_contact

    // R_aaaa = R_abab = R_abac = 0 due to γ5
    const auto R_aaaa = R_abcd_contact(Fv0, Fv0, Fv0, Fv0);
    const auto R_abab = R_abcd_contact(Fv0, Fv1, Fv0, Fv1);
    const auto R_abac = R_abcd_contact(Fv0, Fv1, Fv0, Fa0);
    const auto R_g1a1b = R_abcd_contact(i_gamma_5(F1), Fv0, F1, Fv1);

    // R_(γ5*1)a1a = R_(γ5*a)1a1 = 1
    const auto R_g1a1a = R_abcd_contact(i_gamma_5(F1), Fv0, F1, Fv0);
    const auto R_ga1a1 = R_abcd_contact(i_gamma_5(Fv0), F1, Fv0, F1);

    std::cout << "\nRadial contact integration orthonormality checks\nR_αααα = "
              << R_aaaa << "\t=0?\nR_abab = " << R_abab
              << "\t=0?\nR_abac = " << R_abac
              << "\t=0?\nR_(γ1)a1b = " << R_g1a1b << "\t=0?\n"
              << "R_(γ1)a1a = " << R_g1a1a << "\t=1?\nR_(γa)1a1 = " << R_ga1a1
              << "\t=1?\n\n";

  } else {
    // Bessel screening function checks

    // Bk_ab returns vector.
    const auto B0_11 = Bk_ab(0.0, 1.0, Fv0, Fv0);

    // Check 10 values
    std::cout << "\nBessel function integration checks with λ=0 and "
                 "fa=fb=ga=gb=1\nr          f1   g1   numeric  analytic\n";
    for (int i = 0; i < wf.grid().size();
         i += std::round(wf.grid().size() / 20)) {
      const auto ri = wf.grid().r(i);
      const auto i0 = gsl_sf_bessel_Inu(0.0, ri);
      const auto k0 = gsl_sf_bessel_Knu(0.0, ri);

      const auto B0_11_actual =
          k0 * gsl_sf_Shi(ri) - i0 * gsl_sf_expint_Ei(-ri);

      fmt::print("{:10.6f} {:4.2f} {:4.2f} {:4.2e} {:4.2f}\n", ri, F1.f(i),
                 F1.g(i), B0_11[i], B0_11_actual);
    }
  }

  // B0_11(r) = k0(r)\int_0^r dr' i0(r) + i0(r)\int_r^\infty dr' k0(r)
  // B0_11(r) = k0(r)Shi[r] - i0(r)Ei[-r]

  // Radial integral checks

  // V_nv checks

  // Dv checks
}

} // namespace Module