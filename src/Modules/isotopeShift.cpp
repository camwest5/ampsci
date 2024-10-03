#include "Modules/isotopeShift.hpp"
#include "Angular/Wigner369j.hpp"
#include "DiracOperator/DiracOperator.hpp" //For E1 operator
#include "DiracOperator/Operators/RadialF.hpp"
#include "ExternalField/TDHF.hpp"
#include "IO/InputBlock.hpp"
#include "Physics/PhysConst_constants.hpp" // For GHz unit conversion
#include "Wavefunction/Wavefunction.hpp"
#include "ampsci.hpp"

#include <gsl/gsl_fit.h>

namespace Module {

void fieldShift(const IO::InputBlock &input, const Wavefunction &wf) {

  input.check(
      {{"", "Calculates field shift: F = d(E)/d(<r^2>)"},
       {"print", "Print each step? [true]"},
       {"min_pc", "Minimum percentage shift in r [1.0e-3]"},
       {"max_pc", "Maximum percentage shift in r [1.0e-1]"},
       {"num_steps", "Number of steps for derivative (for each sign)? [3]"}});
  // If we are just requesting 'help', don't run module:
  if (input.has_option("help")) {
    return;
  }

  const auto print = input.get("print", true);

  const auto min_pc = input.get("min_pc", 1.0e-3);
  const auto max_pc = input.get("max_pc", 1.0e-1);
  const auto num_steps = input.get<unsigned long>("num_steps", 3);

  Wavefunction wfB(wf.grid_sptr(), wf.nucleus(), wf.alpha() / PhysConst::alpha);

  std::cout << "Calculating field shift corrections for \n"
            << wf.atom() << ", " << wf.nucleus() << "\n"
            << "By fitting de = F<dr^2> for small delta r\n";

  wfB.copySigma(
      wf.Sigma()); // Not quite correct, as wfB's sigma should be determined 'from scratch'

  const auto core_string = wf.coreConfiguration();
  const auto val_string = DiracSpinor::state_config(wf.valence());
  const auto r0 = wf.get_rrms();

  std::vector<std::vector<std::pair<double, double>>> data(wf.valence().size());
  const auto delta_grid = Grid(r0 * min_pc / 100.0, r0 * max_pc / 100.0,
                               num_steps, GridType::logarithmic);

  if (print) {
    std::cout << "\n   r_rms (fm)    del(r)     del(r^2)     dE (GHz)   F "
                 "(GHz/fm^2)\n";
  } else {
    std::cout << "\nRunning...\n";
  }
  for (const auto pm : {-1, 1}) {
    for (const auto del : delta_grid.r()) {
      const auto rB = r0 + pm * del;
      const auto dr2 = rB * rB - r0 * r0;

      auto nuc_b = wf.nucleus();
      nuc_b.set_rrms(rB);

      wfB.update_Vnuc(Nuclear::formPotential(nuc_b, wf.grid().r()));

      wfB.solve_core("HartreeFock", 0.0, core_string, 0.0, false);
      wfB.solve_valence(val_string, false);
      wfB.hartreeFockBrueckner(false);

      for (auto i = 0ul; i < wfB.valence().size(); ++i) {
        const auto &Fv = wfB.valence()[i];
        const auto &Fv0 = *wf.getState(Fv.n(), Fv.kappa());
        const auto dE = (Fv.en() - Fv0.en()) * PhysConst::Hartree_GHz;
        const auto tF = dE / dr2;
        if (print)
          printf("%4s  %7.5f  %+7.5f  %11.4e  %11.4e  %10.3e\n",
                 Fv.shortSymbol().c_str(), rB, rB - r0, dr2, dE, tF);
        auto &data_v = data[i];
        data_v.emplace_back(dr2, dE);
      }
      if (print)
        std::cout << "\n";
    }
  }

  std::cout << "\n";

  auto sorter = [](auto p1, auto p2) { return p1.first < p2.first; };

  for (auto i = 0ul; i < wfB.valence().size(); ++i) {
    const auto &Fv = wfB.valence()[i];
    auto &data_v = data[i];
    std::sort(begin(data_v), end(data_v), sorter);

    [[maybe_unused]] double c0, c1, cov00, cov01, cov11, sumsq;

    // Fit, without c0
    gsl_fit_mul(&data_v[0].first, 2, &data_v[0].second, 2, data_v.size(), &c1,
                &cov11, &sumsq);
    std::cout << Fv.symbol() << " "
              << "F = " << c1 << " GHz/fm^2, sd = " << std::sqrt(sumsq) << "\n";
  }
}

void isotopeShift(const IO::InputBlock &input, const Wavefunction &wf) {
  using namespace qip::overloads;
  input.check(
      {{"", "Determines isotope shift"},
       {"A2", "Second isotope's mass number"},
       {"new_correlations",
        "Create new correlations for second isotope? [true]"},
       {"range", "Range of isotope mass numbers around A2 to include [0]"}});
  // If we are just requesting 'help', don't run module:
  if (input.has_option("help")) {
    return;
  }

  const auto A2 = input.get<int>("A2");
  const bool correlations = input.get<bool>("new_correlations", true);
  const int range = input.get<int>("range", 0);

  // Read original input file again, much like in main()
  auto new_input =
      IO::InputBlock("ampsci", input.path(), std::fstream(input.path()));

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
    if (correlations == true) {
      new_input.merge("Correlations{read = false; write = false;}");
    }

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

  std::cout << "\nCalculating isotope shift for excited-ground state "
               "transitions between reference isotope\n  "
            << wf.atom() << " " << wf.nucleus() << "\n and the isotopes\n";

  for (int i = 0; i < wf2s.size(); i++) {
    std::cout << "  " << wf2s[i].atom() << " " << wf2s[i].nucleus() << "\n";
  }

  //std::cout << "\nA               del(r^2)     dE (MHz) F (MHz/fm^2)    "
  //        << "FS (MHz)   NMS (MHz)  SMS* (MHz)   IS (MHz)\n";

  // Find delta<r^2>
  const auto r0 = wf.get_rrms();

  for (int i = 0; i < wf2s.size(); i++) {
    const auto r2 = wf2s.at(i).get_rrms();
    const auto drr = r2 * r2 - r0 * r0;

    // Calculate field shift via <v|dV|v>
    const DiracOperator::RadialF dV(wf.vnuc() - wf2s[i].vnuc());
    ExternalField::TDHF tdhf(&dV, wf.vHF());

    std::cout << "\nCalculating field shift parameters for " << wf2s[i].atom()
              << ":\n";
    tdhf.solve_core(0, 100, true);

    double FS_ground;
    double NMS_ground;
    double SMS_ground;

    std::cout << "\nIS parameters and energy contributions:";
    std::cout << "\nA     state  Ksms (GHz amu) F (MHz/fm^2) NMS (MHz) "
                 "SMS* (MHz)    FS (MHz)"
                 "    IS (MHz)\n";

    for (auto j = 0ul; j < wf2s.at(i).valence().size(); j++) {

      const auto &Fv2 = wf2s[i].valence()[j];
      const auto &Fv0 = *wf.getState(Fv2.n(), Fv2.kappa());

      /* Not currently using this approach - this determines dE just by taking the difference
      double dE = (Fv0.en() - Fv.en()) * PhysConst::Hartree_MHz;
      double F = dE / drr;
      */

      // Field shift
      const auto factor = dV.rme3js(Fv0.twoj(), Fv0.twoj());
      auto FSv = factor * (dV.reducedME(Fv0, Fv0) + tdhf.dV(Fv0, Fv0)) *
                 PhysConst::Hartree_MHz;

      // Normal mass shift - which energy should be used? This formula is from Dzuba (2005)
      const auto NMSv = (Fv0.en() / PhysConst::u_NMU) *
                        ((1.0 / wf2s[i].Anuc()) - (1.0 / wf.Anuc())) *
                        PhysConst::Hartree_MHz;

      // Specific mass shift - currently first order, <v|T|v> = tvv2
      double tvv0 = 0;
      double tvv2 = 0;

      // Sum over core states
      for (auto k = 0ul; k < wf.core().size(); k++) {

        // Does A0 and A2 wavefunctions at the same time for each state, but this is not efficient if running multiple isotopes (only need to do wf0 once).

        // Naming: A is generally wf0, unlabeled or A2 should be generally wf2s[i].
        auto Fa2 = wf2s[i].core()[k];
        auto Fa0 = wf.core()[k];

        // Reduced matrix element <v||C^1||a>
        double RME2 = Angular::Ck_kk(1, Fv2.kappa(), Fa2.kappa());
        double RME0 = Angular::Ck_kk(1, Fv0.kappa(), Fa0.kappa());

        // Doesn't include factor -i, not important here but note for future use
        double Pva2 = DiracOperator::p().radialIntegral(Fv2, Fa2);
        double Pva0 = DiracOperator::p().radialIntegral(Fv0, Fa0);

        tvv2 += (1.0 / Fv2.twojp1()) * std::abs(RME2 * RME2) *
                std::abs(Pva2 * Pva2);
        tvv0 += (1.0 / Fv0.twojp1()) * std::abs(RME0 * RME0) *
                std::abs(Pva0 * Pva0);
      }
      tvv2 *= -1.0;
      tvv0 *= -1.0;
      double MA2 = wf2s[i].Anuc() * PhysConst::u_NMU;
      double MA0 = wf.Anuc() * PhysConst::u_NMU;

      double SMS0 = tvv0 * (MA0 / ((MA0 + 1) * (MA0 + 1)));
      double SMS2 = tvv2 * (MA2 / ((MA2 + 1) * (MA2 + 1)));

      double SMSv = (SMS2 - SMS0) * PhysConst::Hartree_MHz;

      // Store ground state
      if (j == 0) {
        FS_ground = FSv;
        NMS_ground = NMSv;
        SMS_ground = SMSv;
      }

      // IS parameters for each state
      double F = FSv / drr;
      double K = tvv2 * PhysConst::Hartree_GHz;

      // IS parameters between excited and ground states
      double FS = FS_ground - FSv;
      double NMS = NMS_ground - NMSv;
      double SMS = SMS_ground - SMSv;
      double IS = FS + NMS + SMS;

      fmt::print(
          "{:3}  {:4}  {:13.4f} {:13.4f} {:9.4f} {:10.4f} {:11.4f} {:11.4f}\n",
          wf2s[i].Anuc(), Fv2.symbol().c_str(), K, F, NMS, SMS, FS, IS);
      /*
      printf("%3i %4s  %11.4e  %11.4f  %11.4f %11.4f %11.4f %11.4f %11.4f \n",
             wf2s.at(i).Anuc(), Fv.symbol().c_str(), drr, dE, dE / drr, FS, NMS,
             SMS, IS);
      */
    }
    std::cout
        << "*SMS is only to first order, and Ksms is incorrectly calculated!\n";

    // See Viatkina 2023 for a good list of results for Ca+. Currently matching for FS.

    // Good for Calcium, doesn't agree with Caesium??

    // use namespace qip::overloads
    // de = <V|dV|V> = Fv * (v * Fv)
    // auto dV = wf.Vnuc() - wf2.Vnuc()
    // Look for F and reproduce

    // Check that normal mass shift is small
  }
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
