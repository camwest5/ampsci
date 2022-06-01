#include "IO/ChronoTimer.hpp"
#include "IO/FRW_fileReadWrite.hpp" //for 'ExtraPotential'
#include "IO/InputBlock.hpp"
#include "Maths/Grid.hpp"
#include "Maths/Interpolator.hpp" //for 'ExtraPotential'
#include "Modules/runModules.hpp"
#include "Physics/include.hpp"
#include "Wavefunction/Wavefunction.hpp"
#include "git.hpp"
#include "qip/Vector.hpp"
#include <iostream>
#include <memory>
#include <string>

void ampsci(const IO::InputBlock &input);

const std::string ampsci_help{R"(
  ampsci - Atomic Many-body Perturbation theory in the Screened Coulomb
  Interaction. Run the program with input options from the command line, e.g.:

  $ ./ampsci filename
    - Runs ampsci with input option specified in file "filename"
    - If "filename" is blank, will assume it is "ampsci.in"

  $ ./ampsci <At> <Core> <Valence>
    - For quick use: simple HF calculation. e.g.,
    $ ./ampsci Cs
      - Runs ampsci for Cs using Hartree Fock (V^N) approximation
    $ ./ampsci Cs [Xe] 6sd5d
      - Runs ampsci for Cs using Hartree Fock with Xe-like core and valence
        states up to n=6 for s,p-states and n=5 for d-states
    $ ./ampsci Cs
      - Runs ampsci for Cs using Hartree Fock (V^N) approximation

  $ ./ampsci -v
    - Prints version info (same as --version)
  $ ./ampsci -h
    - Print help info (same as --help)

  Output is printed to screen. It's recommended to forward this to a text file.
  The input options and the ampsci version details are also printed, so that
  the program output contains all required info to exactly reproduce it.
  e.g.,
  $ ./ampsci input |tee -a outout
    - Runs ampsci using input options in file "input".
    - Output will both be written to screen, and appended to
      file "output".
)"};

//==============================================================================
int main(int argc, char *argv[]) {

  // Parse input text into strings:
  const std::string input_text = (argc > 1) ? argv[1] : "ampsci.in";
  const std::string core_text = (argc > 2) ? argv[2] : "";
  const std::string valence_text = (argc > 3) ? argv[3] : "";

  // check for special commands
  if (input_text == "-v" || input_text == "--version") {
    GitInfo::print_git_info();
    return 0;
  } else if (input_text == "-h" || input_text == "--help") {
    std::cout << ampsci_help << '\n';
    return 0;
  } else if (!input_text.empty() && input_text.front() == '-') {
    std::cout << "Unrecognised option: " << input_text << '\n';
    std::cout << ampsci_help << '\n';
    return 0;
  }

  // Print git/version info to screen:
  std::cout << "\n";
  IO::print_line();
  GitInfo::print_git_info();
  std::cout << "Run time: " << IO::time_date() << '\n';

  // If we are not given a valid input text file, assume input is in form:
  // <At> <core> <valence> (e.g., "Cs [Xe] 6sp")
  // All optional, but must appear in order. ("Core" may be skipped for H)
  const auto atom = AtomData::atomicSymbol(AtomData::atomic_Z(input_text));
  const auto core =
      atom == "H" ? "" : (core_text == "" ? "[" + atom + "]" : core_text);
  // allow core to be skipped for Hydrogen
  const auto valence = (atom == "H" && argc == 3) ? core_text : valence_text;
  const std::string default_input = "Atom{Z=" + atom + ";}" +
                                    "HartreeFock { core = " + core +
                                    "; valence = " + valence + ";}";

  // nb: std::filesystem not available in g++-7 (getafix version)
  const auto fstream = std::fstream(input_text);
  auto input = fstream.good() ? IO::InputBlock("ampsci", fstream) :
                                IO::InputBlock("ampsci", default_input);

  // Run program. Add option to run multiple times
  ampsci(input);

  return 0;
}

//==============================================================================
void ampsci(const IO::InputBlock &input) {
  IO::ChronoTimer timer("\nampsci");
  std::cout << "\n";
  IO::print_line();
  input.print();

  using namespace std::string_literals;

  input.check({{"", "All top-level inputs are InputBlocks. Format for "
                    "descriptions are:\n Type. Description [default_value]"},
               {"Atom", "InputBlock. Which atom to run for"},
               {"Grid", "InputBlock. Set radial grid parameters"},
               {"HartreeFock", "InputBlock. Options for Solving atomic system"},
               {"Nucleus", "InputBlock. Set nuclear parameters"},
               {"RadPot", "InputBlock. Inlcude QED radiative potential"},
               {"Basis", "InputBlock. Basis used for MBPT"},
               {"Spectrum", "InputBlock. Like basis; used for sum-over-states"},
               {"Correlations", "InputBlock. Options for correlations"},
               {"ExtraPotential", "InputBlock. Include an extra potential"},
               {"dVpol",
                "InputBlock. Approximate correlation (polarisation) potential"},
               {"Module::*",
                "InputBlock. Run any number of modules (* -> module name)"}});

  // Atom: Get + setup atom parameters
  input.check({"Atom"},
              {{"Z", "string/int. Atomic number [default = H]"},
               {"A", "int. Atomic mass number (set A=0 to use pointlike "
                     "nucleus) [default based on Z]"},
               {"varAlpha2", "Variation of FSC alphs: d(a^2)/a_0^2 [1.0]"}});

  const auto atom_Z = AtomData::atomic_Z(input.get({"Atom"}, "Z", "H"s));
  const auto atom_A = input.get({"Atom"}, "A", AtomData::defaultA(atom_Z));
  const auto var_alpha = [&]() {
    const auto varAlpha2 = input.get({"Atom"}, "varAlpha2", 1.0);
    return (varAlpha2 > 0.0) ? std::sqrt(varAlpha2) : 1.0e-25;
  }();

  // Grid: Get + setup grid parameters
  input.check(
      {"Grid"},
      {{"r0", "double. Initial grid point, in aB [1.0e-6]"},
       {"rmax", "double. Finial grid point [120.0]"},
       {"num_points", "int. Number of grid points [2000]"},
       {"type",
        "string. Type of grid: loglinear, logarithmic, linear [loglinear]"},
       {"b", "double. only used for loglinear: grid is ~ logarithmic for r<b, "
             "linear for r>b [rmax/3]"},
       {"du", "double. du is uniform grid step size; set this instead of "
              "num_points - will override nuRunsm_points [default set by "
              "num_points]"}});

  // Radial grid. Shared resource used by all wavefunctions/orbitals etc
  const auto r0 = input.get({"Grid"}, "r0", 1.0e-6);
  const auto rmax = input.get({"Grid"}, "rmax", 120.0);
  const auto du = input.get<double>({"Grid"}, "du");
  // if num_points = 0, uses du.
  const auto num_points = du ? 0ul : input.get({"Grid"}, "num_points", 2000ul);
  const auto b = input.get({"Grid"}, "b", rmax / 3.0);
  const auto grid_type =
      (b <= r0 || b >= rmax) ?
          "logarithmic" :
          input.get<std::string>({"Grid"}, "type", "loglinear");
  const auto radial_grid = std::make_shared<const Grid>(
      GridParameters{num_points, r0, rmax, b, grid_type, du ? *du : 0});

  // Nucleus: Get + setup nuclear parameters
  input.check(
      {"Nucleus"},
      {{"rrms", "Root-mean-square charge radius, in fm "
                "[default depends on Z and A]"},
       {"c", "Half-density radius, in fm (use instead of rrms) [default "
             "depends on Z and A]"},
       {"t", "Nuclear skin thickness, in fm [2.3]"},
       {"type", "Fermi, spherical, pointlike, Gaussian [Fermi]"}});

  // usually, give rrms. Giving c will over-ride rrms
  const auto c_hdr = input.get<double>({"Nucleus"}, "c");
  auto dflt_rrms = Nuclear::find_rrms(atom_Z, atom_A);
  if (dflt_rrms <= 0.0 && atom_A != 0 && !c_hdr) {
    dflt_rrms = Nuclear::approximate_r_rms(atom_A);
    std::cout << "\nWARNING: isotope Z=" << atom_Z << ", A=" << atom_A
              << " - cannot find rrms. Using approx formula: rrms=" << dflt_rrms
              << "\n";
  }
  const auto t_skin = input.get({"Nucleus"}, "t", Nuclear::default_t);
  // c (half density radius) takes precidence if c and r_rms are given.
  const auto rrms = c_hdr ? Nuclear::rrms_formula_c_t(*c_hdr, t_skin) :
                            input.get({"Nucleus"}, "rrms", dflt_rrms);
  // Set nuc. type explicitly to 'pointlike' if A=0, or r_rms = 0.0
  const auto nuc_type =
      (atom_A == 0 || rrms == 0.0) ?
          "pointlike" :
          input.get<std::string>({"Nucleus"}, "type", "Fermi");

  // Create wavefunction object
  Wavefunction wf(radial_grid, {atom_Z, atom_A, nuc_type, rrms, t_skin},
                  var_alpha);

  std::cout << "\nRunning for " << wf.atom() << "\n"
            << wf.nuclearParams() << "\n"
            << wf.grid().gridParameters() << "\n"
            << "========================================================\n";

  // Parse input for Hartree-Fock
  input.check(
      {"HartreeFock"},
      {{"core", "Core configuration. e.g., [Xe] for Cs"},
       {"valence", "Which valence states? e.g., 7sp5d"},
       {"eps", "HF convergance goal [1.0e-13]"},
       {"method", "HartreeFock, Hartree, KohnSham, Local [HartreeFock]"},
       {"Breit",
        "Scale for Breit. 0.0 => no Breit, 1.0 => include Breit. [0.0]"},
       {"sortOutput", "Sort energy tables by energy? [false]"}});

  const auto core = input.get<std::string>({"HartreeFock"}, "core", "[]");
  const auto HF_method =
      input.get<std::string>({"HartreeFock"}, "method", "HartreeFock");
  const auto eps_HF = input.get({"HartreeFock"}, "eps", 1.0e-13);
  const auto x_Breit = input.get({"HartreeFock"}, "Breit", 0.0);
  const auto valence = input.get<std::string>({"HartreeFock"}, "valence", "");
  const auto sorted_output = input.get({"HartreeFock"}, "sortOutput", false);

  // Set up the Hartree Fock potential/method (does not solve)
  wf.set_HF(HF_method, x_Breit, core, eps_HF, true);

  // Inlcude QED radiatve potential
  input.check(
      {"RadPot"},
      {{"",
        "QED Radiative potential will be included if this block is present"},
       {"", "The following 5 are all doubles. Scale to include * potential; "
            "usually either 0.0 or 1.0, but can take any value:"},
       {"Ueh", "  Uehling (vacuum pol). [1.0]"},
       {"SE_h", "  self-energy high-freq electric. [1.0]"},
       {"SE_l", "  self-energy low-freq electric. [1.0]"},
       {"SE_m", "  self-energy magnetic. [1.0]"},
       {"WK", "  Wickman-Kroll. [0.0]"},
       {"rcut", "Maximum radius (au) to calculate Rad Pot for [5.0]"},
       {"scale_rN",
        "Scale factor for Nuclear size. 0 for pointlike, 1 for typical [1.0]"},
       {"scale_l", "List of doubles. Extra scaling factor for each l e.g., "
                   "1,0,1 => include for s and d, but not for p [1.0]"},
       {"core_qed",
        "Include rad pot into Hartree-Fock core (relaxation) [true]"}});

  const auto include_qed = input.getBlock("RadPot") != std::nullopt;
  const auto x_Ueh = input.get({"RadPot"}, "Ueh", 1.0);
  const auto x_SEe_h = input.get({"RadPot"}, "SE_h", 1.0);
  const auto x_SEe_l = input.get({"RadPot"}, "SE_l", 1.0);
  const auto x_SEm = input.get({"RadPot"}, "SE_m", 1.0);
  const auto x_wk = input.get({"RadPot"}, "WK", 0.0);
  const auto rcut = input.get({"RadPot"}, "rcut", 5.0);
  const auto scale_rN = input.get({"RadPot"}, "scale_rN", 1.0);
  const auto x_spd = input.get({"RadPot"}, "scale_l", std::vector{1.0});
  const bool core_qed = input.get({"RadPot"}, "core_qed", true);

  // If 'core_qed' is true, add rad pot _before_ we solve HF core
  if (include_qed && core_qed) {
    std::cout << "Including QED into Hartree-Fock core (and valence)\n\n";
    wf.radiativePotential({x_Ueh, x_SEe_h, x_SEe_l, x_SEm, x_wk}, rcut,
                          scale_rN, x_spd);
  }

  // Inlcude extra potential (read in from text file):
  // Note: interpolated onto grid, but NOT extrapolated
  // (zero outside region!)
  input.check(
      {"ExtraPotential"},
      {{"", "Option to add an extra potential (to Vnuc), before HF solved."},
       {"filename",
        "Read potential from file (r v(r)) - will be interpolated [blank]"},
       {"factor", "potential is scaled by this value [default=1]"},
       {"beforeHF", "include before HF (into core states). default=false"}});

  const auto include_extra = input.getBlock("ExtraPotential") != std::nullopt;
  const auto ep_fname =
      input.get<std::string>({"ExtraPotential"}, "filename", "");
  const auto ep_factor = input.get({"ExtraPotential"}, "factor", 1.0);
  const auto ep_beforeHF = input.get({"ExtraPotential"}, "beforeHF", false);
  std::vector<double> Vextra;
  if (include_extra && ep_fname != "") {
    const auto &[x, y] = IO::FRW::readFile_xy_PoV(ep_fname);
    Vextra = Interpolator::interpolate(x, y, wf.grid().r());
    qip::scale(&Vextra, ep_factor);
  }

  // Add "extra potential", before HF (core + valence)
  if (include_extra && ep_beforeHF) {
    // qip::add(&wf.vnuc(), Vextra);
    wf.add_to_Vnuc(Vextra);
  }

  // Solve Hartree equations for the core:
  wf.solve_core(true);

  // ... if not core_qed, then add QED _after_ HF core
  if (include_qed && !core_qed) {
    wf.radiativePotential({x_Ueh, x_SEe_h, x_SEe_l, x_SEm, x_wk}, rcut,
                          scale_rN, x_spd);
    std::cout << "Including QED into Valence only\n\n";
  }

  // Add "extra potential", after HF (only valence)
  if (include_extra && !ep_beforeHF) {
    wf.add_to_Vdir(Vextra);
  }

  // Adds effective polarision potential to direct
  // potential (After HF core, before HF valence)
  input.check({"dVpol"},
              {{"a_eff", "scale factor for effective pol. potential [1]"},
               {"r_cut", "cut-off parameter [=1]"}});
  const auto a_eff = input.get<double>({"dVpol"}, "a_eff");
  if (a_eff) {
    const auto r_cut = input.get({"dVpol"}, "r_cut", 1.0);
    std::cout << "Adding effective polarisation potential: a=" << *a_eff
              << ", rc=" << r_cut << "\n";
    const auto a4 = r_cut * r_cut * r_cut * r_cut;
    auto dV = [=](auto x) {
      return a_eff ? -0.5 * *a_eff / (x * x * x * x + a4) : 0.0;
    };
    std::vector<double> dv;
    dv.reserve(wf.grid().num_points());
    for (auto r : wf.grid().r()) {
      dv.push_back(dV(r));
    }
    wf.add_to_Vdir(dv);
  }

  // Solve for the valence states:
  wf.solve_valence(valence);

  // Output Hartree Fock energies:
  std::cout << "\n" << wf.identity() << "-" << wf.Anuc() << "\n";
  wf.printCore(sorted_output);
  wf.printValence(sorted_output);

  // Construct B-spline basis:
  input.check(
      {"Basis"},
      {{"number", "int. Number of splines used in expansion [0]"},
       {"order", "int. order of splines ~7-9 [7]"},
       {"r0", "double. minimum cavity radius (first internal knot) [1.0e-4]"},
       {"r0_eps", "double. Select cavity radius r0 for each l by position "
                  "where |psi(r0)/psi_max| falls below r0_eps [1.0e-3]"},
       {"rmax", "double. maximum cavity radius [Grid{rmax}]"},
       {"states", "string. states to keep (e.g., 30spdf20ghi)"},
       {"print", "bool. Print all spline energies (for testing) [false]"},
       {"positron", "bool. Include -ve energy states [false]]"},
       {"type", "string. Derevianko (DKB) or Johnson [Derevianko]"}});

  const auto basis_input = input.getBlock("Basis");
  if (basis_input) {
    wf.formBasis(*basis_input);
    if (input.get({"Basis"}, "print", false) && !wf.basis().empty()) {
      std::cout << "Basis:\n";
      wf.printBasis(wf.basis());
    }
  }

  // Correlations: read in options
  // This is a mess - will re-do correlations part
  const auto Sigma_ok = input.check(
      {"Correlations"},
      {{"Brueckner", "Form Brueckner orbitals [false]"},
       {"energyShifts", "Calculate MBPT2 shift [false]"},
       {"n_min_core", "Minimum core n to polarise [1]"},
       {"fitTo_cm", "List of binding energies (in cm^-1) to scale Sigma for. "
                    "Must be in same order as valence states"},
       {"lambda_kappa",
        "Scaling factors for Sigma. Must be in same order as valence states"},
       {"read", "Filename to read in Sigma [false=don't read]"},
       {"write", "Filename to write Sigma to [false=don't write]"},
       {"rmin", "minimum radius to calculate sigma for [1.0e-4]"},
       {"rmax", "maximum radius to calculate sigma for [30.0]"},
       {"stride", "Only calculate Sigma every <stride> points"},
       {"each_valence", "Different Sigma for each valence states? [false]"},
       {"ek",
        "Block: Explicit list of energies to solve for. e.g., ek{6s+=-0.127, "
        "7s+=-0.552;}. Blank => HF energies"},
       {"Feynman", "Use Feynman method [false]"},
       {"fk",
        "List of doubles. Screening factors for effective all-order "
        "exchange. In Feynman method, used in exchange+ladder only; Goldstone, "
        "used direct also. If blank, will calculate them from scratch. []"},
       {"eta", "List of doubles. Hole-Particle factors. In Feynman method, "
               "used in ladder only; Goldstone, used direct also. []"},
       {"screening", "bool. Include Screening [false]"},
       {"holeParticle", "Include hole-particle interaction [false]"},
       {"ladder", "string. Filename for ladder diagram file. If blank, ladder "
                  "not included. Only in Feynman. []"},
       {"lmax", "Maximum l used for Feynman method [6]"},
       {"basis_for_Green", "Use basis for Feynman Greens function [false]"},
       {"basis_for_pol", "Use basis for Feynman polarisation op [false]"},
       {"real_omega", "[worked out by default]"},
       {"imag_omega", "w0, wratio for Im(w) grid [0.01, 1.5]"},
       {"include_G", "Inlcude lower g-part into Sigma [false]"}});

  const bool do_energyShifts =
      input.get({"Correlations"}, "energyShifts", false);
  const bool do_brueckner = input.get({"Correlations"}, "Brueckner", false);
  const auto n_min_core = input.get({"Correlations"}, "n_min_core", 1);
  const auto sigma_rmin = input.get({"Correlations"}, "rmin", 1.0e-4);
  const auto sigma_rmax = input.get({"Correlations"}, "rmax", 30.0);
  const auto default_stride = [&]() {
    // By default, choose stride such that there is 150 points over [1e-4,30]
    const auto stride =
        int(wf.grid().getIndex(30.0) - wf.grid().getIndex(1.0e-4)) / 150;
    return (stride <= 2) ? 2 : stride;
  }();
  const auto sigma_stride =
      input.get({"Correlations"}, "stride", default_stride);

  // Feynman method:
  const auto sigma_Feynman = input.get({"Correlations"}, "Feynman", false);
  const auto sigma_Screening = input.get({"Correlations"}, "screening", false);
  const auto hole_particle = input.get({"Correlations"}, "holeParticle", false);
  const auto sigma_lmax = input.get({"Correlations"}, "lmax", 6);
  const auto GreenBasis = input.get({"Correlations"}, "basis_for_Green", false);
  const auto PolBasis = input.get({"Correlations"}, "basis_for_pol", false);
  const auto each_valence = input.get({"Correlations"}, "each_valence", false);
  const auto include_G = input.get({"Correlations"}, "include_G", false);
  const auto ladder_file = input.get({"Correlations"}, "ladder", ""s);
  // force sigma_omre to be always -ve
  const auto sigma_omre = -std::abs(
      input.get({"Correlations"}, "real_omega", -0.33 * wf.energy_gap()));

  // Imaginary omegagrid params (only used for Feynman)
  double w0 = 0.01;
  double wratio = 1.5;
  {
    const auto imag_om =
        input.get({"Correlations"}, "imag_omega", std::vector{w0, wratio});
    if (imag_om.size() != 2) {
      std::cout << "ERROR: imag_omega must be a list of 2: omega_0 (first "
                   "step), and omega_ratio (ratio for log w grid)\n";
    } else {
      w0 = imag_om[0];
      wratio = imag_om[1];
    }
  }

  // Solve Sigma at specific energies
  // e.g., ek{6s+=-0.127, 7s+=-0.552;}
  const auto ek_Sig = input.getBlock({"Correlations"}, "ek");

  // Read/write Sigma to file:
  auto sigma_write = input.get<std::string>({"Correlations"}, "write", "");
  // By default,  try to  read  from  write  file  (if it exists)
  const auto sigma_read = input.get({"Correlations"}, "read", sigma_write);
  // don't  write to default filename when reading from another file
  if (sigma_read != "" && sigma_write == "")
    sigma_write = "false";

  // To fit Sigma to energies:
  auto fit_energies =
      input.get({"Correlations"}, "fitTo_cm", std::vector<double>{});
  // energies given in cm^-1, convert to au:
  qip::scale(&fit_energies, 1.0 / PhysConst::Hartree_invcm);
  const auto lambda_k =
      input.get({"Correlations"}, "lambda_kappa", std::vector<double>{});

  const auto fk = input.get({"Correlations"}, "fk", std::vector<double>{});
  const auto etak = input.get({"Correlations"}, "eta", std::vector<double>{});

  // Form correlation potential:
  if ((do_energyShifts || do_brueckner) && Sigma_ok) {
    IO::ChronoTimer t("Sigma");
    wf.formSigma(n_min_core, do_brueckner, sigma_rmin, sigma_rmax, sigma_stride,
                 each_valence, include_G, lambda_k, fk, etak, sigma_read,
                 sigma_write, ladder_file, sigma_Feynman, sigma_Screening,
                 hole_particle, sigma_lmax, GreenBasis, PolBasis, sigma_omre,
                 w0, wratio, ek_Sig);
  }

  // Calculate + print second-order energy shifts
  if (!wf.valence().empty() && do_energyShifts && Sigma_ok) {
    IO::ChronoTimer t("de");
    wf.SOEnergyShift();
  }

  // Solve Brueckner orbitals (optionally, fit Sigma to exp energies)
  if (!wf.valence().empty() && do_brueckner && Sigma_ok) {
    std::cout << "\n";
    IO::ChronoTimer t("Br");
    if (!fit_energies.empty())
      wf.fitSigma_hfBrueckner(valence, fit_energies);
    else
      wf.hartreeFockBrueckner();
  }
  // Print out info for new "Brueckner" valence orbitals:
  if (!wf.valence().empty() && do_brueckner && Sigma_ok) {
    std::cout << "\nBrueckner orbitals:\n";
    wf.printValence(sorted_output);
  }

  // Construct B-spline Spectrum:
  input.check({"Spectrum"},
              {{"number", "Number of splines used in expansion"},
               {"order", "order of splines ~7-9"},
               {"r0", "minimum cavity radius"},
               {"r0_eps", "Select cavity radius r0 for each l by position "
                          "where |psi(r0)/psi_max| falls below r0_eps"},
               {"rmax", "maximum cavity radius"},
               {"states", "states to keep (e.g., 30spdf20ghi)"},
               {"print", "Print all spline energies (for testing)"},
               {"positron", "Include -ve energy states (true/false)"},
               {"type", "Derevianko (DKB) or Johnson"}});

  const auto spectrum_in = input.getBlock("Spectrum");
  if (spectrum_in) {
    if (spectrum_in)
      wf.formSpectrum(*spectrum_in);
    if (input.get({"Spectrum"}, "print", false) && !wf.spectrum().empty()) {
      std::cout << "Spectrum:\n";
      wf.printBasis(wf.spectrum());
    }
  }

  // run each of the modules with the calculated wavefunctions
  Module::runModules(input, wf);
}

//==============================================================================
