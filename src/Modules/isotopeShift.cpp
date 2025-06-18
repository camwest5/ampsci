
#include "Modules/isotopeShift.hpp"
#include "Angular/Wigner369j.hpp"
#include "DiracOperator/Operators/RadialF.hpp"
#include "DiracOperator/include.hpp"
#include "ExternalField/MixedStates.hpp"
#include "ExternalField/TDHF.hpp"
#include "ExternalField/TDHFbasis.hpp"
#include "IO/InputBlock.hpp"
#include "Physics/PhysConst_constants.hpp"
#include "Wavefunction/Wavefunction.hpp"
#include "ampsci.hpp"
#include <cmath>

#include "fmt/ostream.hpp"
#include "qip/Array.hpp"
#include "qip/Vector.hpp"
#include <gsl/gsl_fit.h>
#include <gsl/gsl_multifit.h>
#include <gsl/gsl_statistics_double.h>

#include <iostream>

namespace Module {

void fieldShift(const IO::InputBlock &input, const Wavefunction &wf) {
  input.check({{"", "Calculates field shift using MBPT, including 2nd-order "
                    "(quadratic) field shift, via the TDHF/RPA(basis) method"},
               {"RPA", "Include RPA? true/Ffalse [true]"},
               {"num_steps", "Number of steps to compute changes in <r^2> for; "
                             "should be even, >=2 [20]"},
               {"dr", "Smallest change in rms charge radius (fm) [0.0001]"}});

  // If we are just requesting 'help', don't run module:
  if (input.has_option("help")) {
    return;
  }

  using namespace qip::overloads;

  IO::ChronoTimer timer("isotopeShift");

  const auto num_steps_tmp = std::max(input.get("num_steps", 20ul), 2ul);
  const auto delta_r = input.get("dr", 0.0001);
  const auto rpaQ = input.get("RPA", true);

  // Initial (reference) nuclear parameters:
  const auto &nuc0 = wf.nucleus();
  const auto r_rms0 = nuc0.r_rms();
  const auto r20 = r_rms0 * r_rms0;
  const auto r40 = DiracOperator::fieldshift::r4(wf.grid(), wf.nucleus());

  std::cout
      << "\nCalculating shift in valence state energies using.\n"
      << "dE^(1) = F d<r^2> + G^2 d<r^2>^2 + G^4 d<r^4>\n"
      << "[nb: E is binding energy (different sign from someother works)]\n";

  // For second-order G^2 correction,
  // See, e.g., Eq.(8) PhysRevA.103.L030801 (2021)

  std::cout << "\nInitial nuclear parameters:\n"
            << "r_rms_0 = " << r_rms0 << " fm, "
            << "<r^2>_0 = " << r20 << " fm^2, "
            << "<r^4>_0 = " << r40 << " fm^4\n";

  // Grid for delta_rrms: use logarithmic grid (should be fit around 0)
  const auto gtype = GridType::logarithmic;
  // const auto gtype = GridType::linear;
  const auto range = gtype == GridType::logarithmic ?
                         qip::logarithmic_range<double, std::size_t> :
                         qip::uniform_range<double, std::size_t>;

  // Even grids on both sides of delta_r = 0
  const auto max_dr = delta_r * int(num_steps_tmp) / 2.0;
  const auto drs_t = range(delta_r, max_dr, num_steps_tmp / 2);
  const auto drs = qip::merge(-1.0 * qip::reverse(drs_t), drs_t);
  const auto num_steps = drs.size();

  // Arrays to store data for fitting:
  qip::Array F1_data(wf.valence().size(), num_steps);
  qip::Array G2_data(wf.valence().size(), num_steps);
  std::vector<double> dr4s(num_steps);

  std::cout << "\ndE(1) (GHz)\n";
  fmt::print("{:6s} {:8s} {:8s}", "r_rms", " d<r2>", " d<r4>");
  for (const auto &Fv : wf.valence()) {
    fmt::print("  {:10s}", Fv.shortSymbol());
  }
  std::cout << std::endl;
  std::stringstream gout, gout2;

  // Loop over nuclear charge radii:
  auto nuc = nuc0;
  std::vector<double> bs;
  for (std::size_t i = 0ul; i < drs.size(); ++i) {
    if (std::abs(drs.at(i)) < 1.0e-10)
      continue;

    nuc.set_rrms(r_rms0 + drs.at(i));

    const auto r4 = DiracOperator::fieldshift::r4(wf.grid(), nuc);
    const auto r22 = qip::pow<4>(nuc.r_rms());
    bs.push_back(r4 / r22);

    // Consistancy check:
    // const auto r2 = DiracOperator::fieldshift::r2(wf.grid(), nuc);
    // std::cout << std::sqrt(r2) << " " << nuc.r_rms() << "\n";

    DiracOperator::fieldshift fis(wf.grid(), nuc0, nuc);

    const auto r_rms = nuc.r_rms();
    const auto delta_r2 = fis.dr2();
    const auto delta_r4 = fis.dr4();
    dr4s.at(i) = delta_r4;

    // Always use TDHF now
    ExternalField::TDHF dVfis(&fis, wf.vHF());

    if (rpaQ)
      dVfis.solve_core(0.0, 100, false);

    fmt::print("{:6.4f} {:8.5f} {:8.5f}", r_rms, delta_r2, delta_r4);
    fmt::print(gout, "{:6.4f} {:8.5f} {:8.2e}", r_rms, delta_r2,
               delta_r2 * delta_r2);
    fmt::print(gout2, "{:6.4f} {:8.5f} {:8.2e}", r_rms, delta_r2,
               delta_r2 * delta_r2);

    //-------------------------------------------------------
    // Loop over valence states:
    for (std::size_t j = 0ul; j < wf.valence().size(); ++j) {
      const auto &Fv = wf.valence().at(j);

      const auto k = fis.rme3js(Fv.twoj(), Fv.twoj()); // RME -> ME

      // First-order correction:
      auto F_rme = fis.reducedME(Fv, Fv) + dVfis.dV(Fv, Fv);
      const auto dE = F_rme * k * PhysConst::Hartree_GHz;

      fmt::print(" {:11.4e}", dE);

      F1_data(j, i) = dE;

      // Second-order G^2 correction:
      {

        // Use TDHF method:
        const auto hFv =
            fis.reduced_rhs(Fv.kappa(), Fv) + dVfis.dV_rhs(Fv.kappa(), Fv);

        const auto dFv1 = dVfis.solve_dPsi(Fv, 0.0, ExternalField::dPsiType::X,
                                           Fv.kappa(), wf.Sigma());

        // .. and using SOS method:
        // If including correlations, should have a spectrum for MBPT
        const auto &basis = wf.spectrum().empty() ? wf.basis() : wf.spectrum();
        // Note: negative energy states are required here!!
        const auto dFv2 =
            ExternalField::solveMixedState_basis(Fv, hFv, 0.0, basis);

        const auto G2_rme = fis.reducedME(Fv, dFv1) + dVfis.dV(Fv, dFv1);
        const auto G2_rme2 = fis.reducedME(Fv, dFv2) + dVfis.dV(Fv, dFv2);
        const auto dE2 = G2_rme * k * k * PhysConst::Hartree_GHz;

        // Just ude TDHF for the fit: more stable
        G2_data(j, i) = dE2;

        fmt::print(gout, " {:11.4e}", G2_rme);
        fmt::print(gout2, " {:11.4e}", G2_rme2);
      }
    }
    if (dVfis.last_eps() > 1.0e-8)
      std::cout << " ***";
    std::cout << std::endl;
    fmt::print(gout, "\n");
    fmt::print(gout2, "\n");
  }

  const auto b = qip::mean(bs);
  const auto db = qip::sem(bs, 1);
  std::cout << "\n<r4> = b * (<r^2>^2)\n";
  fmt::print("b = {:.6f} +/- {:.6f}\n", b, db);

  // Print the G2 results below:
  {
    std::cout << "\ndE(2) (GHz) - TDHF method\n";
    fmt::print("{:6s} {:8s} {:8s}", "r_rms", " d<r2>", " d<r2>^2");
    for (const auto &Fv : wf.valence()) {
      fmt::print("  {:10s}", Fv.shortSymbol());
    }
    std::cout << std::endl;
    std::cout << gout.str();

    std::cout << "\ndE(2) (GHz) - SOS method (as a check)\n"
                 " * note: Requires large spectrum + negative energy states!\n";
    fmt::print("{:6s} {:8s} {:8s}", "r_rms", " d<r2>", " d<r2>^2");
    for (const auto &Fv : wf.valence()) {
      fmt::print("  {:10s}", Fv.shortSymbol());
    }
    std::cout << std::endl;
    std::cout << gout2.str();
  }

  //----------------------------------------------------------------------------
  // Perform fit to dE = F * d<r^2> to extract F
  // AND, independantly for QFS
  // fit to dE(2) = G2 * (d<r^2>)^2 to extract G2
  std::cout << "\n-------------------------------------\n";
  std::cout << "Fit two independent linear fits:\n"
            << "dE(1) = F  * d<r^2>     : extract F\n"
            << "dE(2) = G2 * (d<r^2>)^2 : extract G2 (QFS)\n";

  // Used for the linear fits:
  const auto dr2s = (r_rms0 + drs) * (r_rms0 + drs) - r20;
  const auto dr2s2 = dr2s * dr2s;

  // Store G2s (re-use later)
  std::vector<std::pair<double, double>> G2s;

  // For transitions:
  std::stringstream out_trans;
  std::optional<double> F0, G0;
  std::string v0;

  std::cout << "\nv     F (GHz/fm^2)   err*       G2 (GHz/fm^4)   err\n";
  for (auto i = 0ul; i < wf.valence().size(); ++i) {
    const auto &Fv = wf.valence().at(i);
    auto G2_data_v = G2_data.row(i);
    auto F1_data_v = F1_data.row(i);

    // Linear fit, use GSL
    // Not strictly necisary, but gives estimate of numerical error
    // https://www.gnu.org/software/gsl/doc/html/lls.html

    double F{0.0}, G{0.0}, covF{0.0}, covG{0.0}, sumsq{0.0};
    // Fit for F
    gsl_fit_mul(dr2s.data(), 1, F1_data_v.data(), 1, dr2s.size(), &F, &covF,
                &sumsq);
    // Fit for G_2
    gsl_fit_mul(dr2s2.data(), 1, G2_data_v.data(), 1, dr2s2.size(), &G, &covG,
                &sumsq);

    fmt::print("{:4s} {:13.6e}   {:7.1e}   {:13.6e}    {:7.1e}\n",
               Fv.shortSymbol(), F, std::sqrt(covF), G, std::sqrt(covG));

    // Store first F0, G0: used for transitions
    if (!F0) {
      F0 = F;
      G0 = G;
      v0 = Fv.shortSymbol();
    }

    G2s.push_back({G, std::sqrt(covG)});
    fmt::print(out_trans, "{:4s} - {:4s} {:13.6e}   {:13.6e}\n",
               Fv.shortSymbol(), v0, F - *F0, G - *G0);
  }

  std::cout << "\nTransitions: F (GHz/fm^2)   G2 (GHz/fm^4)\n";
  std::cout << out_trans.str() << "\n";

  std::cout << "(* err is just from the fit: "
            << "it is a minimum numerical error only)\n";
  std::cout << "(This F includes the G^4 term)\n";

  //----------------------------------------------------------------------------
  // Perform fit to dE = F2 * d<r^2> + G4 * d<r^4> to extract F2, G4
  std::cout << "\n-------------------------------------\n";
  std::cout << "Fit to dE = F2 * d<r^2> + G4 * d<r^4> : extract F2, G4\n";

  // Fit for F2 and G4 at same time:

  // For the transitions:
  std::stringstream out_trans_4;
  std::optional<double> F20, G20, G40, E0;
  std::cout << "\nv     F2 (GHz/fm^2)  err*     G2 (GHz/fm^4)  err*     G4 "
               "(GHz/fm^4)  err\n";
  for (auto i = 0ul; i < wf.valence().size(); ++i) {
    const auto &Fv = wf.valence().at(i);
    auto dE = F1_data.row(i);

    // Allocate matrices for the independent variables and
    // vector for the dependent variable
    const auto n = dE.size();
    gsl_matrix *X = gsl_matrix_alloc(n, 2);
    gsl_vector *y = gsl_vector_alloc(n);
    gsl_vector *c = gsl_vector_alloc(2);
    gsl_matrix *cov = gsl_matrix_alloc(2, 2);

    // Populate X matrix and y vector
    for (size_t j = 0; j < n; ++j) {
      gsl_matrix_set(X, j, 0, dr2s[j]);
      gsl_matrix_set(X, j, 1, dr4s[j]);
      gsl_vector_set(y, j, dE[j]);
    }

    // Perform the linear regression
    gsl_multifit_linear_workspace *work = gsl_multifit_linear_alloc(n, 2);
    double chisq{0.0};
    gsl_multifit_linear(X, y, c, cov, &chisq, work);

    // Extract the fitted coefficients
    double F2 = gsl_vector_get(c, 0);
    double G4 = gsl_vector_get(c, 1);
    double cov_F2 = gsl_matrix_get(cov, 0, 0);
    double cov_G4 = gsl_matrix_get(cov, 1, 1);

    // calculated before:
    const auto [G2, dG2] = G2s.at(i);

    // Store first F0, G0
    if (!F20) {
      F20 = F2;
      G20 = G2;
      G40 = G4;
      E0 = Fv.en();
      v0 = Fv.shortSymbol();
    }

    // Print the results
    fmt::print("{:4s} {:13.6e}   {:5.0e}   {:13.6e}   {:5.0e}   {:13.6e}   "
               "{:5.0e}\n",
               Fv.shortSymbol(), F2, std::sqrt(cov_F2), G2, dG2, G4,
               std::sqrt(cov_G4));

    const auto ww = (Fv.en() - *E0) * PhysConst::Hartree_GHz;
    fmt::print(out_trans_4,
               "{:4s} - {:4s} : {:13.6e}   {:13.6e}   {:13.6e}   {:13.6e}\n",
               Fv.shortSymbol(), v0, ww, F2 - *F20, G2 - *G20, G4 - *G40);

    // Free memory
    gsl_multifit_linear_free(work);
    gsl_matrix_free(X);
    gsl_vector_free(y);
    gsl_vector_free(c);
    gsl_matrix_free(cov);
  }

  std::cout
      << "\nTransitions :  Omega (GHz)     F2 (GHz/Fm^2)   G2 (GHz/fm^4)   "
         "G4 (GHz/fm^4)\n";
  std::cout << out_trans_4.str() << "\n";
}

//==============================================================================
void fieldShift_direct(const IO::InputBlock &input, const Wavefunction &wf) {

  input.check(
      {{"", "Calculates field shift: F = d(E)/d(<r^2>) by direct calculation. "
            "Note: copies the same correlation potential; this is OK, but not "
            "exact (i.e., neglects the SR contribution)"},
       {"core_relaxation", "Include Core relaxation (equiv to RPA)? [true]"},
       {"print", "Print each step to screen? [true]"},
       {"write", "Write dE(r^2) to file? [false]"},
       {"minmax_delta", "Minimum relative shift in r [1.0e-5, 1.0e-3]"},
       {"num_steps", "Number of steps for fit (for each sign)? [5]"},
       {"grid", "Logarithmic or linear grid for dr2 [logarithmic]"}});
  // If we are just requesting 'help', don't run module:
  if (input.has_option("help")) {
    return;
  }

  const auto core_relax = input.get("core_relaxation", true);
  const auto print = input.get("print", true);
  const auto write = input.get("write", false);

  const auto [min_d, max_d] =
      input.get("minmax_delta", std::array{1.0e-5, 1.0e-3});

  const auto num_steps = input.get<unsigned long>("num_steps", 5);

  const auto grid_type = input.get("grid", std::string{"logarithmic"});

  Wavefunction wfB(wf.grid_sptr(), wf.nucleus(), wf.alpha() / PhysConst::alpha);

  std::cout << "Calculating field shift corrections for \n"
            << wf.atom() << ", " << wf.nucleus() << "\n"
            << "By fitting de = F<dr^2> for small delta r\n"
            << "Directly re-solves Hartree-Fock at each step\n"
            << "(Note: does not re-calculate Sigma!)\n";

  wfB.copySigma(
      wf.Sigma()); // Not quite correct, as wfB's sigma should be determined 'from scratch'

  const auto core_string = wf.coreConfiguration();
  const auto val_string = DiracSpinor::state_config(wf.valence());
  const auto r0 = wf.get_rrms();

  std::cout << "\n";
  wfB.solve_core("HartreeFock", 0.0, core_string, 0.0, true);
  if (!core_relax) {
    std::cout << "Not including Core relaxtion\n";
  } else {
    std::cout << "Including Core relaxtion\n";
  }

  // Optionally write dE(R) to file for plotting
  std::ofstream of;
  if (write) {
    of.open(wf.identity() + "_FS.txt");
    of << "dr dr2 ";
    for (auto &v : wf.valence()) {
      of << " " << v;
    }
    of << "\n";
  }

  const auto gtype = qip::ci_wc_compare(grid_type, "log*") ?
                         GridType::logarithmic :
                         GridType::linear;

  const auto drs_t =
      gtype == GridType::logarithmic ?
          qip::logarithmic_range(r0 * min_d, r0 * max_d, num_steps) :
          qip::uniform_range(r0 * min_d, r0 * max_d, num_steps);
  using namespace qip::overloads;
  const auto drs = qip::merge(-1.0 * qip::reverse(drs_t), drs_t);

  if (print) {
    std::cout << "\n   r_rms (fm)     del(r)     del(r^2)     dE (GHz)   F "
                 "(GHz/fm^2)\n";
  } else {
    std::cout << "\nRunning...\n";
  }

  // Store data - used for linear fit to get F
  std::vector<std::vector<std::pair<double, double>>> data(wf.valence().size());

  for (const auto del : drs) {
    const auto rB = r0 + del;
    const auto dr2 = rB * rB - r0 * r0;

    auto nuc_b = wf.nucleus();
    nuc_b.set_rrms(rB);

    wfB.update_Vnuc(Nuclear::formPotential(nuc_b, wf.grid().r()));

    if (write) {
      of << rB - r0 << " " << dr2;
    }

    if (core_relax)
      wfB.solve_core("HartreeFock", 0.0, core_string, 0.0, false);
    wfB.solve_valence(val_string, false);
    wfB.hartreeFockBrueckner(false);

    for (auto i = 0ul; i < wfB.valence().size(); ++i) {
      const auto &Fv = wfB.valence().at(i);
      const auto &Fv0 = *wf.getState(Fv.n(), Fv.kappa());
      const auto dE = (Fv.en() - Fv0.en()) * PhysConst::Hartree_GHz;
      const auto tF = dE / dr2;
      if (print)
        printf("%4s  %7.5f  %+8.6f  %11.4e  %11.4e  %10.3e\n",
               Fv.shortSymbol().c_str(), rB, rB - r0, dr2, dE, tF);
      auto &data_v = data[i];
      data_v.emplace_back(dr2, dE);

      if (write) {
        of << " " << dE;
      }
    }

    if (write) {
      of << "\n";
    }
    if (print) {
      std::cout << "\n";
    }
  }

  std::cout << "\n";

  // Fit straight line to data
  std::cout << "\nv     F (GHz/fm^2)   err*\n";
  for (auto i = 0ul; i < wfB.valence().size(); ++i) {
    const auto &Fv = wfB.valence().at(i);
    auto &data_v = data[i];

    // Linear fit
    // https://www.gnu.org/software/gsl/doc/html/lls.html
    double c1{0.0}, cov11{0.0}, sumsq{0.0};
    gsl_fit_mul(&data_v[0].first, 2, &data_v[0].second, 2, data_v.size(), &c1,
                &cov11, &sumsq);

    fmt::print("{:4s} {:13.6e}   {:7.1e}\n", Fv.shortSymbol(), c1,
               std::sqrt(cov11));
  }
  std::cout << "\n";
  std::cout << "(* err is just from the fit: "
            << "it is a minimum numerical error only)\n";
}

void isotopeShift(const IO::InputBlock &input, const Wavefunction &wf) {
  using namespace qip::overloads;
  input.check({{"", "Determines isotope shift"},
               {"A2", "Second isotope's mass number"},
               {"new_correlations",
                "Create new correlations for second isotope? [true]"},
               {"plusminus",
                "Range of isotope mass numbers around A2 to include [0]"}});
  // If we are just requesting 'help', don't run module:
  if (input.has_option("help")) {
    return;
  }

  const auto A2 = input.get<int>("A2");
  const int range = input.get<int>("plusminus", 0);

  // Read original input file again, much like in main()
  auto new_input =
      IO::InputBlock("ampsci", input.path(), std::fstream(input.path()));

  const auto mass_shift = new_input.get({"HartreeFock"}, "mass_shift", false);
  // Create range of As
  auto wf2s_size =
      std::abs(wf.Anuc() - A2.value()) > range ? range * 2 + 1 : range * 2;

  std::vector<Wavefunction> wf2s;

  for (int i = 0; i < wf2s_size; i++) {

    int new_A = i + A2.value() - range;

    if (new_A == wf.Anuc()) {
      continue;
    }

    new_input.merge("Atom{A = " + std::to_string(new_A) + ";}");

    // Currently just adds new_A without removing previous. OK because it reads the last,
    // but would be safer to remove original

    // Remove all modules
    const auto blocks_copy = new_input.blocks();

    for (const auto block : blocks_copy) {
      auto name = block.name();

      if (name.substr(0, 8) == "Module::") {
        new_input.remove_block(name);
      }
    }

    // Create second wavefunction
    std::cout << "\nCreating wavefunction for A = " << new_A << ".\n";

    // Create second wavefunction
    wf2s.push_back(ampsci(new_input));
  }

  std::cout << "\nCalculating isotope shift contributions between reference "
               "isotope\n  "
            << wf.atom() << " " << wf.nucleus() << "\n and the isotopes\n";

  const auto r0 = wf.get_rrms();

  // Each constant estimate for each state for each isotope (state<isotope<constant>>)
  // Will take mean for each state at the end
  std::vector<std::vector<double>> F(wf.valence().size());
  std::vector<std::vector<double>> Fdir(wf.valence().size());
  std::vector<std::vector<double>> Ksms(wf.valence().size());
  std::vector<std::vector<double>> Knms(wf.valence().size());

  for (const auto wf2 : wf2s) {

    // Calculate field shift via operator method (can't use ΔE since SMS is indistinguishable)
    const auto r2 = wf2.get_rrms();
    const auto drr = r2 * r2 - r0 * r0;

    // Calculate field shift via <v|dV|v>
    const DiracOperator::RadialF dV(wf2.vnuc() - wf.vnuc());
    ExternalField::TDHF tdhf(&dV, wf.vHF());
    std::cout << "\nCalculating field shift parameters for " << wf2.atom()
              << ", r_rms = " << wf2.get_rrms() << ":\n";
    tdhf.solve_core(0, 100, true);

    std::cout << "  " << wf2.atom() << " " << wf2.nucleus() << "\n";
    std::cout << "\nA'   state          En (au)       ΔE (MHz)      NMS (MHz)  "
                 "    SMS (MHz)       FS (MHz)  IS to "
              << wf.valence()[0].shortSymbol() << " (MHz)\n";
    for (auto i = 0ul; i < wf2.valence().size(); ++i) {
      const auto &Fv0 = wf.valence()[i];
      const auto &Fv2 = wf2.valence()[i];

      // Apply normal mass shift (currently *0.0 in HartreeFock.cpp)
      // Check sign - this looks consistent with Dzuba (2005)
      const auto NMS0 = Fv0.en() / (wf.Anuc() * PhysConst::u_NMU + 1);
      const auto NMS2 = Fv2.en() / (wf2.Anuc() * PhysConst::u_NMU + 1);

      // Find field shift - default included at HF level
      const auto factor = dV.rme3js(Fv0.twoj(), Fv0.twoj());
      auto FS = factor * (dV.reducedME(Fv0, Fv0) + tdhf.dV(Fv0, Fv0)) *
                PhysConst::Hartree_MHz;

      const auto E0 = Fv0.en() + NMS0;
      const auto E2 = Fv2.en() + NMS2;

      const auto dE = E2 - E0;

      double dE_ground;

      if (i == 0) {
        dE_ground = dE;
      }

      const auto IS_to_ground = (dE - dE_ground) * PhysConst::Hartree_MHz;
      const auto NMS = (NMS2 - NMS0) * PhysConst::Hartree_MHz;
      double SMS;

      if (mass_shift == true) {
        SMS = (Fv2.en() - Fv0.en()) * PhysConst::Hartree_MHz - FS;
        Ksms[i].push_back((SMS / 1000) /
                          ((1.0 / wf2.Anuc()) - (1.0 / wf.Anuc())));

      } else {
        double tvv = 0;

        // Sum over core states
        for (auto k = 0ul; k < wf.core().size(); k++) {

          // Does A0 and A2 wavefunctions at the same time for each state, but this is not efficient if running multiple isotopes (only need to do wf0 once).
          auto Fa = wf.core()[k];

          // Reduced matrix element <v||C^1||a>
          double RME = Angular::Ck_kk(1, Fv0.kappa(), Fa.kappa());

          // Doesn't include factor -i, not important here but note for future use
          double Pva = DiracOperator::p().radialIntegral(Fv0, Fa);

          tvv +=
              (1.0 / Fv0.twojp1()) * std::abs(RME * RME) * std::abs(Pva * Pva);
        }
        tvv *= -1.0;

        const auto current_Ksms =
            tvv * PhysConst::Hartree_GHz / PhysConst::u_NMU;
        Ksms[i].push_back(current_Ksms);
        SMS =
            current_Ksms * ((1.0 / wf2s[i].Anuc()) - (1.0 / wf.Anuc())) * 1000;
      }

      Knms[i].push_back((NMS / 1000) /
                        ((1.0 / (wf2.Anuc())) - (1.0 / (wf.Anuc()))));

      F[i].push_back(FS / drr);
      Fdir[i].push_back((Fv2.en() - Fv0.en()) * PhysConst::Hartree_MHz / drr);

      fmt::print(
          "{:3} {:4} {:.13f} {:14.4f} {:14.4f} {:14.4f} {:14.4f} {:16.4f}\n",
          wf2.Anuc(), Fv2.symbol().c_str(), E2, dE * PhysConst::Hartree_MHz,
          NMS, SMS, FS, IS_to_ground);
    }
  }

  std::cout << "\nMass and field shift constants for " << wf.atomicSymbol()
            << "\n\n";
  std::cout << " state Knms (GHz amu) Ksms (GHz amu)   F (MHz/fm^2) ΔE/Δ<r^2> "
               "(MHz/fm^2)\n";

  // Average and print out constants
  for (auto i = 0ul; i < wf.valence().size(); ++i) {
    const auto Knms_avg =
        std::accumulate(Knms[i].begin(), Knms[i].end(), 0.0) / Knms[i].size();
    const auto Knms_range = std::max_element(Knms[i].begin(), Knms[i].end()) -
                            std::min_element(Knms[i].begin(), Knms[i].end());

    const auto Ksms_avg =
        std::accumulate(Ksms[i].begin(), Ksms[i].end(), 0.0) / Ksms[i].size();
    const auto Ksms_range = std::max_element(Ksms[i].begin(), Ksms[i].end()) -
                            std::min_element(Ksms[i].begin(), Ksms[i].end());

    const auto F_avg =
        std::accumulate(F[i].begin(), F[i].end(), 0.0) / F[i].size();
    const auto F_range = std::max_element(F[i].begin(), F[i].end()) -
                         std::min_element(F[i].begin(), F[i].end());

    const auto Fdir_avg =
        std::accumulate(Fdir[i].begin(), Fdir[i].end(), 0.0) / Fdir[i].size();
    const auto Fdir_range = std::max_element(Fdir[i].begin(), Fdir[i].end()) -
                            std::min_element(Fdir[i].begin(), Fdir[i].end());

    fmt::print("{:6} {:14.4f} {:14.4f} {:14.4f} {:20.4f}\n",
               wf.valence()[i].symbol().c_str(), Knms_avg, Ksms_avg, F_avg,
               Fdir_avg);
  }
  if (mass_shift == false) {
    std::cout << "\n*SMS and Ksms to first order only.\n";
  } else {
    std::cout << "\n*ΔE/Δ<r^2> calculations of F include SMS which may reduce "
                 "accuracy.\n";
  }

  /*
  //std::cout << "\nA               del(r^2)     dE (MHz) F (MHz/fm^2)    "
  //        << "FS (MHz)   NMS (MHz)  SMS* (MHz)   IS (MHz)\n";

  // Find mass shift constants for all valence states using reference isotope
  // std::vector<double> Ksms(wf.valence().size());
  std::vector<double> Knms(wf.valence().size());

  for (auto i = 0ul; i < wf.valence().size(); i++) {
    const auto &Fv = wf.valence()[i];

    
    // Specific mass shift - currently first order, <v|T|v> = tvv
    double tvv = 0;

    // Sum over core states
    for (auto k = 0ul; k < wf.core().size(); k++) {

      // Does A0 and A2 wavefunctions at the same time for each state, but this is not efficient if running multiple isotopes (only need to do wf0 once).
      auto Fa = wf.core()[k];

      // Reduced matrix element <v||C^1||a>
      double RME = Angular::Ck_kk(1, Fv.kappa(), Fa.kappa());

      // Doesn't include factor -i, not important here but note for future use
      double Pva = DiracOperator::p().radialIntegral(Fv, Fa);

      tvv += (1.0 / Fv.twojp1()) * std::abs(RME * RME) * std::abs(Pva * Pva);
    }
    tvv *= -1.0;

    // std::cout << "\n" << Fv.symbol() << " " << tvv;

    // In GHz amu
    Ksms[i] = tvv * PhysConst::Hartree_GHz / PhysConst::u_NMU;
    

    // Normal mass shift
    // In GHz amu
    Knms[i] = Fv.en() * PhysConst::Hartree_GHz / PhysConst::u_NMU;
  }

  // Determine remaining terms between isotopes

  // Find delta<r^2>
  const auto r0 = wf.get_rrms();
  for (int i = 0; i < wf2s.size(); i++) {
    const auto r2 = wf2s[i].get_rrms();
    const auto drr = r2 * r2 - r0 * r0;

    // Calculate field shift via <v|dV|v>
    const DiracOperator::RadialF dV(wf2s[i].vnuc() - wf.vnuc());
    ExternalField::TDHF tdhf(&dV, wf.vHF());

    std::cout << "\nCalculating field shift parameters for " << wf2s[i].atom()
              << ", r_rms = " << wf2s[i].get_rrms() << ":\n";
    tdhf.solve_core(0, 100, true);

    // Mass shift contributions

    // Artificial -1 * has been inserted to conform with Dzuba (2005), but I can't see why.
    // Convention here is that all subtractions are: i - a, where a is reference.
    // This yields a matching FS, but not NMS! Are their energies flipped?

    // Normal mass shifts between A and A' for all states (in MHz)
    const auto NMS =
        -1 * Knms * ((1.0 / wf2s[i].Anuc()) - (1.0 / wf.Anuc())) * 1000;

    // Specific mass shifts between A and A' for all states (in MHz)
    //const auto SMS = Ksms * ((1.0 / wf2s[i].Anuc()) - (1.0 / wf.Anuc())) * 1000;

    double FS_ground;
    double NMS_ground;
    double SMS_ground;

    std::cout
        << "\nIsotope shift parameters and energy contributions between A = "
        << wf.Anuc() << " and A':";
    std::cout
        << "\nA'   state          En (au) Knms (GHz amu) Ksms (GHz amu) "
           "Fdir (MHz/fm^2) F (MHz/fm^2)     NMS (MHz)    SMS* (MHz)    FS "
           "(MHz) IS to "
        << wf.valence()[0].shortSymbol() << " (MHz)\n";

    for (auto j = 0ul; j < wf.valence().size(); j++) {

      const auto &Fv = wf.valence()[j];

      // Not currently using this approach - this determines dE just by taking the difference
      // Might be worth keeping for future reference

      double dE =
          (wf2s[i].valence()[j].en() - Fv.en()) *
          PhysConst::Hartree_MHz; // check whether this is the right way round
      double F_direct = dE / drr;

      // Field shift
      const auto factor = dV.rme3js(Fv.twoj(), Fv.twoj());
      auto FS = factor * (dV.reducedME(Fv, Fv) + tdhf.dV(Fv, Fv)) *
                PhysConst::Hartree_MHz;

      // Normal mass shift - which energy should be used? This formula is from Dzuba (2005) and also Viatkina (2023) (where the negative is introduced)

      // Field shift parameter
      double F = FS / drr;

      // Ground state parameters
      if (j == 0) {
        FS_ground = FS;
        NMS_ground = NMS[j];
        SMS_ground = SMS[j];
      }

      // IS parameters between excited and ground states
      const auto dFS = FS - FS_ground;
      const auto dNMS = NMS[j] - NMS_ground;
      const auto dSMS = SMS[j] - SMS_ground;
      const auto IS = dFS + dNMS + dSMS;

      fmt::print("{:3} {:4} {:.13f} {:14.4f} {:14.4f} {:14.4f} {:13.4f} "
                 "{:13.4f} {:13.4f} "
                 "{:11.4f} {:15.4f}\n",
                 wf2s[i].Anuc(), Fv.symbol().c_str(), wf2s[i].valence()[j].en(),
                 Knms[j], Ksms[j], F_direct, F, NMS[j], SMS[j], FS, IS);
    }
    std::cout << "\n*SMS is only to first order";
  }
  std::cout << "\n\nNote: NMS, SMS and FS are the differences between "
               "isotopes for each state. \nSubtract between states to find "
               "contribution to isotope shifts, as in last column.\n";

  */
}

// Analytically evaluate the normal mass shift NMS in the relativitistic case
void massShift(const IO::InputBlock &input, const Wavefunction &wf) {

  const auto &orbitals = wf.valence();

  std::cout << "     state  "
            << "k   Rinf its   eps     delEnms (au)   delEnms (/cm)      En "
               "(au)         En (/cm)\n";

  int i = 0;
  for (const auto &phi : orbitals) {

    // Mass of electron
    const double m_e = 1;

    // Mass of nucleus - naive approximation
    const double m_A = wf.Anuc() * PhysConst::m_p;

    // Energy correction (NMS)
    const double dEnms = (-m_e / (m_A + m_e)) * phi.en();

    // Output results
    printf("%-2i %7s %2i  %5.1f %2i  %5.0e %15.9f %15.3f %15.9f %15.3f", i++,
           phi.symbol().c_str(), phi.kappa(), phi.rinf(), phi.its(), phi.eps(),
           dEnms, dEnms * PhysConst::Hartree_invcm, phi.en() + dEnms,
           (phi.en() + dEnms) * PhysConst::Hartree_invcm);
    printf("\n");
  }
}

} // namespace Module
