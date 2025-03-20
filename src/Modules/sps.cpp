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

  const auto e_handler = gsl_set_error_handler_off();

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

    for (auto Fn : wf.basis()) {
      if ((Fn.twoj() == Fv.twoj()) && (Fn != Fv)) {
        // <v|d|n>
        const auto d_vn = d_ab(wf.grid(), Fv, Fn);

        // <n|V|v> = Σ_a (u_nava - u_anva)
        const auto Vsps = V_nv(contact, wf.core(), Fv, Fn, y_sps, mu);

        Dv += 2.0 * d_vn * Vsps / (Fv.en() - Fn.en());
      }
    }

    std::cout << Fv.shortSymbol() << "\t" << Dv << "\n";
  }

  gsl_set_error_handler(e_handler);
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

  return (u_nava - u_anva) * y;
}

double Rk_abcd(const double k, const double mu, const DiracSpinor &Fa,
               const DiracSpinor &Fb, const DiracSpinor &Fc,
               const DiracSpinor &Fd) {
  // Compare with yk_ab to find r> and r< functions
  // Then create radial operator?
  // Find the Rk_abcd implementation in code.

  //const auto i0 = std::max(Fa.min_pt(), Fc.min_pt());
  //const auto imax = std::min(Fa.max_pt(), Fc.max_pt());

  const auto screening_function = Bk_ab(k, mu, Fb, Fd);

  const auto ig5_Fc = i_gamma_5(Fc);

  const auto Rff =
      NumCalc::integrate(1.0, 0, Fa.grid().size(), Fa.f(), ig5_Fc.f(),
                         screening_function, Fa.grid().drdu());

  const auto Rgg =
      NumCalc::integrate(1.0, 0, Fa.grid().size(), Fa.g(), ig5_Fc.g(),
                         screening_function, Fa.grid().drdu());

  return (Rff + Rgg) * Fa.grid().du() * mu;
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
  //const auto i0 = std::max(Fa.min_pt(), Fb.min_pt());
  //const auto imax = std::min(Fa.max_pt(), Fb.max_pt());

  // Modified spherical Bessel functions
  std::vector<double> i_k(gr.size());
  std::vector<double> k_k(gr.size());

  for (int i_gr = 0; i_gr < gr.size(); ++i_gr) {
    const auto x = mu * r[i_gr];
    i_k[i_gr] = mod_sph_bessel_i(k, x);
    k_k[i_gr] = mod_sph_bessel_k(k, x);

    // For testing
    //i_k[i_gr] = 1.0;
    //k_k[i_gr] = 1.0;
  }

  // Integrate
  std::vector<double> result(gr.size());

  for (int i_mid = 0; i_mid < gr.size(); ++i_mid) {
    double lower_ff =
        NumCalc::integrate(1.0, 0, i_mid, i_k, Fa.f(), Fb.f(), gr.drdu());

    double lower_gg =
        NumCalc::integrate(1.0, 0, i_mid, i_k, Fa.g(), Fb.g(), gr.drdu());

    // For r0 point
    if (i_mid == 0) {
      lower_ff = 0;
      lower_gg = 0;
    }

    const double upper_ff = NumCalc::integrate(1.0, i_mid, gr.size(), k_k,
                                               Fa.f(), Fb.f(), gr.drdu());

    const double upper_gg = NumCalc::integrate(1.0, i_mid, gr.size(), k_k,
                                               Fa.g(), Fb.g(), gr.drdu());

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

double mod_sph_bessel_i(double n, double x) {
  gsl_sf_result i_k;
  const int gsl_status = gsl_sf_bessel_Inu_e(n + 0.5, x, &i_k);

  if (gsl_status == GSL_SUCCESS) {
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
    return std::sqrt(2.0 / (M_PI * x)) * k_k.val;
  } else if (gsl_status == GSL_EUNDRFLW) {
    return 0.0;
  }
  std::cout << "Need GSL_SUCCESS = " << GSL_SUCCESS
            << " or GSL_EUNDRFLW = " << GSL_EUNDRFLW;
  std::cout << "\n\ngsl_status = " << gsl_status << "\n";

  throw std::bad_function_call();
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

  // Check that the radial contact integral Rδ_abcd is returning expected orthonormality

  // R_aaaa = R_abab = R_abac = 0 due to γ5
  const auto R_aaaa = R_abcd_contact(Fv0, Fv0, Fv0, Fv0);
  const auto R_abab = R_abcd_contact(Fv0, Fv1, Fv0, Fv1);
  const auto R_abac = R_abcd_contact(Fv0, Fv1, Fv0, Fa0);
  const auto R_g1a1b = R_abcd_contact(i_gamma_5(F1), Fv0, F1, Fv1);

  // R_(γ5*1)a1a = R_(γ5*a)1a1 = 1
  const auto R_g1a1a = R_abcd_contact(i_gamma_5(F1), Fv0, F1, Fv0);
  const auto R_ga1a1 = R_abcd_contact(i_gamma_5(Fv0), F1, Fv0, F1);

  std::cout << "\nRadial contact approximation checks\nR_aaaa = " << R_aaaa
            << "\t=0?\nR_abab = " << R_abab << "\t=0?\nR_abac = " << R_abac
            << "\t=0?\nR_(γ1)a1b = " << R_g1a1b << "\t=0?\n"
            << "R_(γ1)a1a = " << R_g1a1a << "\t=1?\nR_(γa)1a1 = " << R_ga1a1
            << "\t=1?\n\n";

  // Check that λ = 0 Bessel function integral Bλ_ab produces expected results
  // Set (fafb + gagb) = 1 for all r
  // Then,
  // B0_11(r) = k0(r) int_0^r dr' i0(r') + i0(r) int_r^inf k0(r') dr'
  //          =       k0(r) * Shi[r]     -      i0(r) * Ei[-r]

  const auto B0_11 = Bk_ab(0.0, 1.0, F1, F1);

  // Check 10 values
  std::cout << "Bessel function integration (Bλ_ab) checks with λ=0 and "
               "(fafb + gagb) = 1 for all r\n         r    f    g numeric "
               "  exact \n";
  for (int i = 0; i < wf.grid().size();
       i += std::round(wf.grid().size() / 10)) {
    const auto ri = wf.grid().r(i);
    const auto i0 = mod_sph_bessel_i(0.0, ri);
    const auto k0 = mod_sph_bessel_k(0.0, ri);

    // B0_11(r) = k0(r)\int_0^r dr' i0(r) + i0(r)\int_r^\infty dr' k0(r)
    // B0_11(r) = k0(r)Shi[r] - i0(r)Ei[-r]
    const auto B0_11_exact =
        k0 * (gsl_sf_Shi(ri) - gsl_sf_Shi(0.000001)) -
        i0 * (gsl_sf_expint_Ei(-ri) - gsl_sf_expint_Ei(-150));

    fmt::print("{:10.6f} {:4.2f} {:4.2f} {:7.4f} {:7.4f}\n", ri, F1.f(i),
               F1.g(i), B0_11_exact, B0_11[i]);
  }

  // Check that radial integral is integrating Bk_ab as expected.
  // With (-fagc + fcga) = (fbfd + gbgd) = 1, should have
  // R0_(γ1)111 = \int_0^inf B0_bd

  const auto R0_g1111 = Rk_abcd(0.0, 1.0, i_gamma_5(F1), F1, F1, F1);
  const auto R0_g1111_manual =
      NumCalc::integrate(1.0, 0.0, wf.grid().size(), B0_11, wf.grid().drdu()) *
      wf.grid().du();

  std::cout
      << "\nRadial function integration with λ=0 and (-fagc + fcga)=(fbfd "
         "+ gbgd)=1 for all r\nR0_(γ1)1111\t = "
      << R0_g1111 << "\nint_0^inf B0_11\t = " << R0_g1111_manual
      << "\nAre these equal?\n";

  // Check radial integral's orthonormality is as expected. Same as contact case
  // Consider λ = 1 and μ = 1

  // R1_aaaa = R1_abab = R1_abac = 0 due to γ5
  const auto R1_aaaa = Rk_abcd(1.0, 1.0, Fv0, Fv0, Fv0, Fv0);
  const auto R1_abab = Rk_abcd(1.0, 1.0, Fv0, Fv1, Fv0, Fv1);
  const auto R1_abac = Rk_abcd(1.0, 1.0, Fv0, Fv1, Fv0, Fa0);
  const auto R1_g1a1b = Rk_abcd(1.0, 1.0, i_gamma_5(F1), Fv0, F1, Fv1);

  std::cout
      << "\nRadial integration orthonormality checks with λ = μ = 1\nR1_aaaa = "
      << R1_aaaa << "\t=0?\nR1_abab = " << R1_abab
      << "\t=0?\nR1_abac = " << R1_abac << "\t=0?\nR1_(γ1)a1b = " << R1_g1a1b
      << "\t=0?\n";

  // V_nv checks

  // Compare contact approximation with large μ

  std::cout << "\nV_nv matrix elements method agreement check. \nFull "
               "implementation should approach contact limit for μ>>0\n  mu"
               "   limit    full k0(150μ) i0(150μ)\n";

  const auto max_mu = 6.0;
  for (double mu = 0.0000001; mu < max_mu; mu += max_mu / 10) {
    const auto V_v0v1_full = V_nv(false, wf.core(), Fv0, Fv1, 1.0, mu);
    const auto V_v0v1_contact = V_nv(true, wf.core(), Fv0, Fv1, 1.0, mu);

    const auto i0_max = mod_sph_bessel_i(0.0, mu * 150.0);
    const auto k0_max = mod_sph_bessel_k(0.0, mu * 150.0);

    fmt::print("{:4.2f} {:7.4f} {:7.4f} {:8.1e} {:8.1e}\n", mu, V_v0v1_contact,
               V_v0v1_full, i0_max, k0_max);
  }

  // Dv checks
}

} // namespace Module