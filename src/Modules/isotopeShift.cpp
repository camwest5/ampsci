
#include "Modules/isotopeShift.hpp"
#include "Angular/Wigner369j.hpp"
#include "DiracOperator/DiracOperator.hpp" //For E1 operator
#include "DiracOperator/Operators/RadialF.hpp"
#include "ExternalField/TDHF.hpp"
#include "IO/InputBlock.hpp"
#include "Physics/PhysConst_constants.hpp" // For GHz unit conversion
#include "Wavefunction/Wavefunction.hpp"
#include "ampsci.hpp"
#include <cmath>

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
  const bool correlations = input.get<bool>("new_correlations", true);
  const int range = input.get<int>("plusminus", 0);

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

  std::cout << "\nCalculating isotope shift contributions between reference "
               "isotope\n  "
            << wf.atom() << " " << wf.nucleus() << "\n and the isotopes\n";

  for (int i = 0; i < wf2s.size(); i++) {
    std::cout << "  " << wf2s[i].atom() << " " << wf2s[i].nucleus() << "\n";
  }

  //std::cout << "\nA               del(r^2)     dE (MHz) F (MHz/fm^2)    "
  //        << "FS (MHz)   NMS (MHz)  SMS* (MHz)   IS (MHz)\n";

  // Find mass shift constants for all valence states using reference isotope
  std::vector<double> Ksms(wf.valence().size());
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
    const DiracOperator::RadialF dV(wf.vnuc() - wf2s[i].vnuc());
    ExternalField::TDHF tdhf(&dV, wf.vHF());

    std::cout << "\nCalculating field shift parameters for " << wf2s[i].atom()
              << ":\n";
    tdhf.solve_core(0, 100, true);

    double FS_ground;
    double NMS_ground;
    double SMS_ground;

    std::cout << "\nIsotope shift parameters and energy contributions:";
    std::cout << "\nA     state  Knms (GHz amu) Ksms (GHz amu) F "
                 "(MHz/fm^2) NMS (MHz) SMS* (MHz)    FS (MHz) IS to "
              << wf.valence()[0].shortSymbol() << " (MHz)\n";

    for (auto j = 0ul; j < wf.valence().size(); j++) {

      const auto &Fv = wf.valence()[j];

      /* 
      // Not currently using this approach - this determines dE just by taking the difference
      // Might be worth keeping for future reference

      double dE = (Fv.en() - Fv.en()) * PhysConst::Hartree_MHz;
      double F = dE / drr;
      */

      // Field shift
      const auto factor = dV.rme3js(Fv.twoj(), Fv.twoj());
      auto FSv = factor * (dV.reducedME(Fv, Fv) + tdhf.dV(Fv, Fv)) *
                 PhysConst::Hartree_MHz;

      // Normal mass shift - which energy should be used? This formula is from Dzuba (2005) and also Viatkina (2023) (where the negative is introduced)

      // Field shift parameter
      double F = FSv / drr;

      // Mass shift contributions

      // In MHz
      const auto NMSv =
          Knms * ((1.0 / wf2s[i].Anuc()) - (1.0 / wf.Anuc())) * 1000;

      // In MHz
      const auto SMSv =
          Ksms * ((1.0 / wf2s[i].Anuc()) - (1.0 / wf.Anuc())) * 1000;

      // Ground state parameters
      if (j == 0) {
        FS_ground = FSv;
        NMS_ground = NMSv[j];
        SMS_ground = SMSv[j];
      }

      // IS parameters between excited and ground states
      const auto FS = FSv - FS_ground;
      const auto NMS = NMSv[j] - NMS_ground;
      const auto SMS = SMSv[j] - SMS_ground;
      const auto IS = FS + NMS + SMS;

      fmt::print("{:3}  {:4}  {:13.4f} {:13.4f} {:13.4f} {:9.4f} {:10.4f} "
                 "{:11.4f} {:11.4f}\n",
                 wf2s[i].Anuc(), Fv.symbol().c_str(), Knms[j], Ksms[j], F,
                 NMSv[j], SMSv[j], FSv, IS);
    }
    std::cout << "*SMS is only to first order\n\nNote: "
                 "NMS, SMS and FS are the differences between isotopes. "
                 "\nSubtract between states to find isotope shifts.\n";
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
