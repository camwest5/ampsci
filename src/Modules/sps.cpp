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
               {"min_mu", "Minimum mediator mass to consider [1e-6]"},
               {"max_mu", "Maximum mediator mass to consider [20]"},
               {"N_mu", "Number of masses to consider [100]"},
               {"test", "Run module testing [false]"}});
  // If we are just requesting 'help', don't run module:
  if (input.has_option("help")) {
    return;
  }

  const auto e_handler = gsl_set_error_handler_off();

  const bool contact = input.get<bool>("contact", true);
  const double min_mu = input.get<double>("min_mu", 1e-6);
  const double max_mu = input.get<double>("max_mu", 100.0);
  const double N_mu = input.get<double>("N_mu", 100.0);

  if (input.get<bool>("test", false) == true) {
    sps_testing(wf, contact);
    return;
  }

  const double y_sps = 1;
  const auto Fv = wf.valence()[0];

  std::cout << "\nAtomic electric dipole moment for the " << Fv.symbol()
            << " state with S-PS interaction (mediator mass = μ).\n  μ (m_e)  "
               "      Dv     i0_max     k0_max\n";

  // Currently looks at one valence state - ground
  for (double log_mu = log(min_mu); log_mu < log(max_mu);
       log_mu += std::abs(log(max_mu) - log(min_mu)) / N_mu) {

    const auto mu = std::exp(log_mu);
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

    const auto i0_max = mod_sph_bessel_i(0.0, mu * wf.grid().rmax());
    const auto k0_max = mod_sph_bessel_k(0.0, mu * wf.grid().rmax());

    fmt::print("{:9.5f} {:9.5f} {:10.1e} {:10.1e}\n", mu, Dv, i0_max, k0_max);
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

  // For safety
  if (Fv.twoj() != Fn.twoj()) {
    return 0.0;
  }

  auto u_nava = 0.0;
  auto u_anva = 0.0;

  for (auto Fa : core) {
    if (Fn.kappa() == -Fv.kappa()) {
      const auto R0_nava = contact == true ?
                               R_abcd_contact(mu, Fn, Fa, Fv, Fa) :
                           mu == 0.0 ? Rk_abcd_massless(0, Fn, Fa, Fv, Fa) :
                                       Rk_abcd(0, mu, Fn, Fa, Fv, Fa);
      u_nava += R0_nava * Fa.twojp1();
    }

    for (int twok = std::abs(Fa.twoj() - Fv.twoj());
         twok <= Fa.twoj() + Fv.twoj(); twok += 2) {
      if ((Fa.twoj() + Fv.twoj() + twok) % 4 == 0) {
        const double k = 0.5 * twok;
        const auto A_anva = (2.0 * k + 1) *
                            Angular::Ck_kk(k, Fa.kappa(), -Fv.kappa()) *
                            Angular::Ck_kk(k, Fn.kappa(), Fa.kappa());
        const auto R_anva = contact == true ?
                                R_abcd_contact(mu, Fa, Fn, Fv, Fa) :
                            mu == 0.0 ? Rk_abcd_massless(k, Fa, Fn, Fv, Fa) :
                                        Rk_abcd(k, mu, Fa, Fn, Fv, Fa);
        /* std::cout << "λ = " << k << "\tκa = " << Fa.kappa()
                  << "\tja = " << Fa.twoj() * 0.5
                  << "\tjv = " << Fv.twoj() * 0.5 << "\tFa = " << Fa.symbol()
                  << "\n";
        std::cout << "A_anva = " << A_anva << "\n"; */
        u_anva += A_anva * R_anva;
      }
    }

    u_anva *= std::pow(-1.0, 0.5 * (Fv.twoj() - Fa.twoj())) / Fv.twojp1();
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

double R_abcd_contact(const double mu, const DiracSpinor &Fa,
                      const DiracSpinor &Fb, const DiracSpinor &Fc,
                      const DiracSpinor &Fd) {

  const auto &gr = Fa.grid();
  const auto &r = gr.r();
  const auto ig5_Fc = i_gamma_5(Fc);

  // Delta case

  std::vector<double> integrand(gr.size());

  for (int i = 0; i < gr.size(); ++i) {
    integrand[i] = (Fa.f(i) * ig5_Fc.f(i) + Fa.g(i) * ig5_Fc.g(i)) *
                   (Fb.f(i) * Fd.f(i) + Fb.g(i) * Fd.g(i)) /
                   (gr.r(i) * gr.r(i));
  }

  return NumCalc::integrate(1.0, 0, gr.size(), integrand, gr.drdu()) * gr.du() /
         (mu * mu);

  /*
  // Bessel approx case 

  //const auto i0 = std::max(Fa.min_pt(), Fb.min_pt());
  //const auto imax = std::min(Fa.max_pt(), Fb.max_pt());

  // Integrate
  std::vector<double> result(gr.size());

  //Slower, more stable way
  std::vector<double> ik(gr.size());

  for (int i_mid = 0; i_mid < gr.size(); ++i_mid) {
    for (int i_gr = 0; i_gr < gr.size(); ++i_gr) {
      ik[i_gr] = std::exp(-mu * std::abs(r[i_mid] - r[i_gr])) /
                 (2 * mu * mu * r[i_gr] * r[i_mid]);
    }
    double B_ff =
        NumCalc::integrate(1.0, 0, gr.size(), ik, Fb.f(), Fd.f(), gr.drdu());

    double B_gg =
        NumCalc::integrate(1.0, 0, gr.size(), ik, Fb.g(), Fd.g(), gr.drdu());

    result[i_mid] = (B_ff + B_gg) * gr.du();
  }

  Faster, less stable way
  std::vector<double> i_k(gr.size());
  std::vector<double> k_k(gr.size());

  for (int i_gr = 0; i_gr < gr.size(); ++i_gr) {
    const auto x = mu * r[i_gr];
    //i_k[i_gr] = mod_sph_bessel_i(k, x);
    //k_k[i_gr] = mod_sph_bessel_k(k, x);

    i_k[i_gr] = std::exp(x) / (2 * x);
    k_k[i_gr] = std::exp(-x) / x;

    // For testing
    //i_k[i_gr] = 1.0;
    //k_k[i_gr] = 1.0;
  }

  for (int i_mid = 0; i_mid < gr.size(); ++i_mid) {
    double lower_ff =
        NumCalc::integrate(1.0, 0, i_mid, i_k, Fb.f(), Fd.f(), gr.drdu());

    double lower_gg =
        NumCalc::integrate(1.0, 0, i_mid, i_k, Fb.g(), Fd.g(), gr.drdu());

    // For r0 point
    if (i_mid == 0) {
      lower_ff = 0;
      lower_gg = 0;
    }

    const double upper_ff = NumCalc::integrate(1.0, i_mid, gr.size(), k_k,
                                               Fb.f(), Fd.f(), gr.drdu());

    const double upper_gg = NumCalc::integrate(1.0, i_mid, gr.size(), k_k,
                                               Fb.g(), Fd.g(), gr.drdu());

    result[i_mid] = (k_k[i_mid] * (lower_ff + lower_gg) +
                     i_k[i_mid] * (upper_ff + upper_gg)) *
                    gr.du();
  } 

  

  const auto B_bd = result;

  const auto Rff = NumCalc::integrate(1.0, 0, Fa.grid().size(), Fa.f(),
                                      ig5_Fc.f(), B_bd, Fa.grid().drdu());

  const auto Rgg = NumCalc::integrate(1.0, 0, Fa.grid().size(), Fa.g(),
                                      ig5_Fc.g(), B_bd, Fa.grid().drdu());

  return (Rff + Rgg) * Fa.grid().du() * mu;

  */
}

double Rk_abcd_massless(const double k, const DiracSpinor &Fa,
                        const DiracSpinor &Fb, const DiracSpinor &Fc,
                        const DiracSpinor &Fd) {
  const auto screening_function = Coulomb::yk_ab(k, Fb, Fd);

  const auto ig5_Fc = i_gamma_5(Fc);

  const auto Rff =
      NumCalc::integrate(1.0, 0, Fa.grid().size(), Fa.f(), ig5_Fc.f(),
                         screening_function, Fa.grid().drdu());

  const auto Rgg =
      NumCalc::integrate(1.0, 0, Fa.grid().size(), Fa.g(), ig5_Fc.g(),
                         screening_function, Fa.grid().drdu());

  return (Rff + Rgg) * Fa.grid().du() / (2 * k + 1);
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

    //i_k[i_gr] = std::exp(x) / (2 * x);
    //k_k[i_gr] = std::exp(-x) / x;

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

void sps_testing(const Wavefunction &wf, const bool contact) {

  if (wf.valence().size() < 2) {
    std::cout << "\nERROR: Cannot run SPS testing with <2 valence states.\n";
    return;
  } else if (wf.core().size() < 2) {
    std::cout << "\nERROR: Cannot run SPS testing with <2 core states.\n";
    return;
  }

  // Pick Fv/a0 and Fv/a1 to be the lowest nκ n(-κ) pair of valence/core states

  DiracSpinor Fv0(wf.valence()[0]);
  DiracSpinor Fv1(wf.valence()[1]);
  DiracSpinor Fa0(wf.core()[0]);
  DiracSpinor Fa1(wf.core()[1]);

  if (Fv0.kappa() != -Fv1.kappa()) {
    bool flag = false;
    for (int i = 0; i < wf.valence().size(); ++i) {
      for (int j = 0; j < wf.valence().size(); ++j) {
        Fv0 = wf.valence()[i];
        Fv1 = wf.valence()[j];

        if (Fv0.kappa() == -Fv1.kappa()) {
          flag = true;
          break;
        }
      }
      if (flag == true) {
        break;
      }
    }
    if (flag == false) {
      std::cout
          << "\n ERROR: No valence states exist with nonzero V_nv. To run "
             "testing, ensure that two valence states exist such that κ1 = "
             "-κ2.\n";
      return;
    }
  }

  // Create spinor F1 satisfying (ff + gg) = 1 (i.e, f = g = 1/sqrt(2))
  DiracSpinor F1(Fv0);
  std::vector<double> ones(wf.grid().size(), 1.0 / std::sqrt(2.0));
  F1.f() = ones;
  F1.g() = ones;

  // Create spinor Fr satisfying (ff + gg) = r^2 (i.e., f = g = r/sqrt(2))
  DiracSpinor Fr(F1);

  for (int i = 0; i < wf.grid().size(); ++i) {
    Fr.f(i) = ones[i] * wf.grid().r(i);
    Fr.g(i) = ones[i] * wf.grid().r(i);
  }

  std::cout << "\nRunning checks on the SPS module with the following "
               "states\nname  state  κ   j\n"
            << " Fv0 " << Fv0.symbol(true) << " " << Fv0.kappa() << " "
            << Fv0.twoj() * 0.5 << "\n Fv1 " << Fv1.symbol(true) << " "
            << Fv1.kappa() << " " << Fv1.twoj() * 0.5 << "\n Fa0 "
            << Fa0.symbol() << " " << Fa0.kappa() << " " << Fa0.twoj() * 0.5
            << "\n Fa1 " << Fa1.symbol() << " " << Fa1.kappa() << " "
            << Fa1.twoj() * 0.5;

  std::cout << "\nAnd the special states 'F1' with f = g = 1/sqrt(2) such that "
               "F1*F1 = ff+gg "
               "= 1,\n\tand 'γF1' where f = -g = 1/sqrt(2) such that γF1*F1 = "
               "-fg+gf = 1,\n\tand 'Fr' where f = g = r/sqrt(2) such that "
               "Fr*Fr = ff+gg = r^2\n";

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

  std::cout << "\nDipole operator checks for v0 = " << Fv0.symbol()

            << " and v1 = " << Fv1.symbol() << "\n<v0|d|v0> = " << vdv
            << "\t\t=0?\n<v0|d|v1> = " << v0dv1
            << "\t=<v1|d|v0>?\n<v1|d|v0> = " << v1dv0
            << "\t=<v0|d|v1>?\n<v0||d||v1> = " << z_ab << "\tif Fr, =5.144?\n";

  // Check that the radial contact integral Rδ_abcd is returning expected orthonormality

  // R_aaaa = R_abab = R_abac = 0 due to γ5
  const auto R_aaaa = R_abcd_contact(1.0, Fv0, Fv0, Fv0, Fv0);
  const auto R_abab = R_abcd_contact(1.0, Fv0, Fv1, Fv0, Fv1);
  const auto R_abac = R_abcd_contact(1.0, Fv0, Fv1, Fv0, Fa0);
  const auto R_1a1b = R_abcd_contact(1.0, Fr, Fv0, Fr, Fv1);

  // Need another special function

  // R_(γ5*1)a1a = R_(γ5*a)1a1 = 1
  const auto R_g1a1a = R_abcd_contact(1.0, i_gamma_5(Fr), Fv0, Fr, Fv0);
  const auto R_ga1a1 = R_abcd_contact(1.0, i_gamma_5(Fv0), Fr, Fv0, Fr);

  std::cout << "\nRadial contact approximation checks\nR_aaaa = " << R_aaaa
            << "\t=0?\nR_abab = " << R_abab << "\t=0?\nR_abac = " << R_abac
            << "\t=0?\nR_1a1b = " << R_1a1b << "\t=0?\n"
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
      << R0_g1111 << "\nExact \t = " << R0_g1111_manual
      << "\nAre these equal?\n";

  // Check radial integral's orthonormality is as expected. Same as contact case
  // Consider λ = 1 and μ = 1

  // R1_aaaa = R1_abab = R1_abac = 0 due to γ5
  const auto R1_aaaa = Rk_abcd(1.0, 1.0, Fv0, Fv0, Fv0, Fv0);
  const auto R1_abab = Rk_abcd(1.0, 1.0, Fv0, Fv1, Fv0, Fv1);
  const auto R1_abac = Rk_abcd(1.0, 1.0, Fv0, Fv1, Fv0, Fa0);
  const auto R1_1a1b = Rk_abcd(1.0, 1.0, F1, Fv0, F1, Fv1);

  std::cout
      << "\nRadial integration orthonormality checks with λ = μ = 1\nR1_aaaa = "
      << R1_aaaa << "\t=0?\nR1_abab = " << R1_abab
      << "\t=0?\nR1_abac = " << R1_abac << "\t=0?\nR1_1a1b = " << R1_1a1b
      << "\t=0?\n";

  // V_nv checks

  // Compare contact approximation with large μ

  const auto max_mu = 1000.0;
  const auto min_mu = 1e-6;

  // Test ground with all core states
  std::cout
      << "\nTesting V_nv in limits for all valence |v> and basis |n> "
         "states.\nShowing >10% discrepancies.\n\nFor the exact cases,\nμ->0 = "
      << min_mu << "\nμ->∞ = " << max_mu;

  std::cout << "\n\nCheck OK:\n|v>  κv   |n>  κn     massless exact (μ->0) "
               " rel diff exact "
               "(μ->∞)      contact  rel diff\n";

  for (auto Fv : wf.valence()) {
    for (auto Fn : wf.basis()) {
      if (Fn.twoj() == Fv.twoj()) {
        const auto V_massless = V_nv(false, wf.core(), Fv, Fn, 1.0, 0.0);
        const auto V_exact_min = V_nv(false, wf.core(), Fv, Fn, 1.0, min_mu);
        const auto V_exact_max = V_nv(false, wf.core(), Fv, Fn, 1.0, max_mu);
        const auto V_contact = V_nv(true, wf.core(), Fv, Fn, 1.0, max_mu);

        const auto diff_massless =
            std::abs((V_massless - V_exact_min) / V_massless);
        const auto diff_contact =
            std::abs((V_contact - V_exact_max) / V_contact);

        if ((diff_massless > 0.1) || (diff_contact > 0.1)) {
          fmt::print("{:3s} {:3}  {:4s} {:3} {:12.3e} {:12.3e} {:9.3f} "
                     "{:12.3e} {:12.3e} {:9.3f}\n",
                     Fv.shortSymbol(), Fv.kappa(), Fn.shortSymbol(), Fn.kappa(),
                     V_massless, V_exact_min, diff_massless, V_exact_max,
                     V_contact, diff_contact);
        }
      }
    }
    std::cout << "\n";
  }

  std::cout << "\nCalculating V_nv matrix elements for range of μ for <"
            << Fv1.symbol() << "|V|" << Fv0.symbol()
            << ">\n         mu"
               "    massless        full     contact  i0(150μ)  k0(150μ)\n";

  const auto V_v0v1_massless = V_nv(false, wf.core(), Fv0, Fv1, 1.0, 0.0);

  for (double log_mu = log(min_mu); log_mu < log(max_mu);
       log_mu += std::abs(log(max_mu) - log(min_mu)) / 100) {
    const auto mu = exp(log_mu);
    const auto V_v0v1_contact = V_nv(true, wf.core(), Fv0, Fv1, 1.0, mu);

    auto V_v0v1_full = V_nv(false, wf.core(), Fv0, Fv1, 1.0, mu);

    const auto i0_max = mod_sph_bessel_i(0.0, mu * 150.0);
    const auto k0_max = mod_sph_bessel_k(0.0, mu * 150.0);
    /*if (i0_max == 0 & k0_max == 0) {
      V_v0v1_full = 0.0;
    } */

    fmt::print("{:11.6f}  {:10.3e}  {:10.3e}  {:10.3e}  {:8.1e}  {:8.1e}\n", mu,
               V_v0v1_massless, V_v0v1_full, V_v0v1_contact, i0_max, k0_max);
  }
}

// Dv checks

} // namespace Module