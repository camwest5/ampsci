#include "Modules/VQE.hpp"
#include "Angular/Angular.hpp"
#include "CI/CI.hpp"
#include "Coulomb/Coulomb.hpp"
#include "DiracOperator/DiracOperator.hpp"
#include "IO/InputBlock.hpp"
#include "LinAlg/Matrix.hpp"
#include "MBPT/Sigma2.hpp"
#include "Physics/AtomData.hpp"
#include "Wavefunction/Wavefunction.hpp"
#include "fmt/format.hpp"
#include "fmt/ostream.hpp"
#include "qip/Vector.hpp"
#include <array>
#include <fstream>
#include <vector>

namespace Module {

using nkIndex = DiracSpinor::Index; // 16 bit unsignd int; unique {n,kappa}->i

//==============================================================================
//==============================================================================
//==============================================================================
// This is the actual module that runs:
void VQE(const IO::InputBlock &input, const Wavefunction &wf) {

  // Check input options:
  input.check(
      {{"ci_basis",
        "Basis used for CI expansion; must be a sub-set of full ampsci basis "
        "[default: 10spdf]"},
       {"frozen_core", "..."},
       {"J+",
        "List of total angular momentum J for even-parity CI solutions (comma "
        "separated). Must be integers (two-electron only)."},
       {"J-", "As above, but for ODD solutions."},
       {"num_solutions", "Number of CI solutions to find (for each J/pi) [5]"},
       {"write_integrals",
        "Writes orbitals, CSFs, CI matrix, and 1 and 2 particle "
        "integrals to plain text file [true]"},
       {"symmetries",
        "Account for symmetries when writing FCI_dump file [false]"},
       {"sigma1", "Include one-body MBPT correlations? [false]"},
       {"sigma2", "Include two-body MBPT correlations? [false]"},
       {"cis2_basis",
        "The subset of ci_basis for which the two-body MBPT corrections are "
        "calculated. Must be a subset of ci_basis. If existing sk file has "
        "more integrals, they will be used. [default: Nspdf, where N is "
        "maximum n for core + 3]"},
       {"s1_basis",
        "Basis used for the one-body MBPT diagrams (Sigma^1). These are the "
        "most important, so in general the default (all basis states) should "
        "be used. Must be a subset of full ampsci basis. [default: full "
        "basis]"},
       {"s2_basis",
        "Basis used for internal lines of the two-body MBPT diagrams "
        "(Sigma^2). Must be a subset of s1_basis. [default: s1_basis]"},
       {"n_min_core", "Minimum n for core to be included in MBPT [1]"},
       {"max_k",
        "Maximum k (multipolarity) to include when calculating new "
        "Coulomb integrals. Higher k often contribute negligably. Note: if qk "
        "file already has higher-k terms, they will be included. Set negative "
        "(or very large) to include all k. [6]"}});
  // If we are just requesting 'help', don't run module:
  if (input.has_option("help")) {
    return;
  }

  // Decide if should write single-particle integrals to file:
  const auto write_integrals = input.get("write_integrals", true);

  //----------------------------------------------------------------------------
  // Single-particle basis:
  std::cout << "\nConstruct single-particle basis:\n";

  // Determine the sub-set of basis to use in CI:
  const auto basis_string = input.get("ci_basis", std::string{"10spdf"});
  const auto frozen_string = input.get("frozen_core", wf.coreConfiguration());

  // std::cout << wf.coreConfiguration() << "\n";
  // std::cout << wf.coreConfiguration() << "\n";
  // std::cin.get();

  // Select from wf.basis() [MBPT basis], those which match input 'basis_string'
  const std::vector<DiracSpinor> ci_sp_basis =
      CI::basis_subset(wf.basis(), basis_string, frozen_string);

  const auto frozen_core = CI::frozen_core_subset(wf.core(), frozen_string);

  // Print info re: basis to screen:
  std::cout << "\nUsing " << DiracSpinor::state_config(ci_sp_basis) << " = "
            << ci_sp_basis.size() << " relativistic single-particle orbitals\n";

  // Write orbital list
  std::map<nkm, int> orbital_map;
  if (write_integrals) {
    std::string fname = wf.atomicSymbol() + "_orbitals_" +
                        DiracSpinor::state_config(ci_sp_basis) + ".txt";
    std::ofstream of(fname);
    int index = 0;
    fmt::print(of, "# index Symbol n kappa m\n");
    for (const auto &Fn : ci_sp_basis) {
      for (int twom = -Fn.twoj(); twom <= Fn.twoj(); twom += 2) {

        fmt::print(of, "{} {} {} {} {}/2  {}\n", index + 1, Fn.shortSymbol(),
                   Fn.n(), Fn.kappa(), twom, Fn.en());
        orbital_map.insert({{Fn.n(), Fn.kappa(), twom}, index});
        ++index;
      }
    }
    std::cout << "( = " << index
              << " single-particle orbitals, including m projections)\n";
  }
  // return;

  //----------------------------------------------------------------------------

  // check if including MBPT corrections
  const auto include_Sigma1 = input.get("sigma1", false);
  const auto include_Sigma2 = input.get("sigma2", false);
  const auto max_k_Coulomb = input.get("max_k", 6);
  const auto include_MBPT = include_Sigma1 || include_Sigma2;

  // s1 and s2 MBPT basis
  const auto s1_basis_string = input.get("s1_basis");
  const auto &s1_basis = s1_basis_string ?
                             CI::basis_subset(wf.basis(), *s1_basis_string) :
                             wf.basis();
  const auto s2_basis_string = input.get("s2_basis");
  const auto &s2_basis = s2_basis_string ?
                             CI::basis_subset(wf.basis(), *s2_basis_string) :
                             s1_basis;

  // Ensure s2_basis is subset of s1_basis
  assert(s2_basis.size() <= s1_basis.size() &&
         "s2_basis must be a subset of s1_basis");

  // Split basis' into core/excited (for MBPT evaluations)
  const auto n_min_core = input.get("n_min_core", 1);
  const auto [core_s1, excited_s1] =
      MBPT::split_basis(s1_basis, wf.FermiLevel(), n_min_core);
  const auto [core_s2, excited_s2] =
      MBPT::split_basis(s2_basis, wf.FermiLevel(), n_min_core);

  // S2 corrections are included only for this subset of the CI basis:
  const auto Ncore = DiracSpinor::max_n(wf.core()) + 3;
  const auto cis2_basis_string =
      input.get("cis2_basis", std::to_string(Ncore) + "spdf");
  const auto &cis2_basis = CI::basis_subset(ci_sp_basis, cis2_basis_string);

  //----------------------------------------------------------------------------

  if (include_MBPT) {
    std::cout << "\nIncluding MBPT: "
              << (include_Sigma1 && include_Sigma2 ? "Σ_1 + Σ_2" :
                  include_Sigma1                   ? "Σ_1" :
                                                     "Σ_2")
              << "\n";
    std::cout << "Including core excitations from n ≥ " << n_min_core << "\n";
    if (max_k_Coulomb >= 0 && max_k_Coulomb < 50) {
      std::cout << "Including k ≤ " << max_k_Coulomb
                << " in Coulomb integrals (unless already calculated)\n";
    }
    if (include_Sigma1) {

      std::cout << "With basis for Σ_1: " << DiracSpinor::state_config(s1_basis)
                << "\n";
    }
    if (include_Sigma2) {
      std::cout << "With basis for Σ_2: " << DiracSpinor::state_config(s2_basis)
                << "\n";
      std::cout << "Including Σ_2 correction to Coulomb integrals up to: "
                << DiracSpinor::state_config(cis2_basis) << "\n";
    }
    std::cout << "\n";
  }

  //----------------------------------------------------------------------------

  // Lookup table; stores all qk's
  Coulomb::QkTable qk;
  {
    std::cout << "Calculate two-body Coulomb integrals: Q^k_abcd\n";

    const auto qk_filename = input.get("qk_file", wf.atomicSymbol() + ".qk");

    // Try to read from disk (may already have calculated Qk)
    qk.read(qk_filename);
    const auto existing = qk.count();
    {

      // Try to limit number of Coulomb integrals we calculate
      // use whole basis (these are used inside Sigma_2)
      // If not including MBPT, only need to caculate smaller set of integrals

      // First, calculate the integrals between ci basis states:
      {
        std::cout << "For: " << DiracSpinor::state_config(ci_sp_basis) << "\n"
                  << std::flush;
        const auto yk = Coulomb::YkTable(ci_sp_basis);
        qk.fill(ci_sp_basis, yk, max_k_Coulomb, false);
      }

      // Selection function for which Qk's to calculate.
      // For Sigma, we only need those with 1 or 2 core electrons
      // i.e., no Q_vwxy, Q_vabc, or Q_abcd
      // Note: we *do* need Q_vwxy for the CI part (but with smaller basis)
      const auto select_Q_sigma =
          [eF = wf.FermiLevel()](int, const DiracSpinor &s,
                                 const DiracSpinor &t, const DiracSpinor &u,
                                 const DiracSpinor &v) {
            // Only calculate Coulomb integrals with 1 or 2 electrons in the core
            auto num = MBPT::number_below_Fermi(s, t, u, v, eF);
            return num == 1 || num == 2;
          };

      // Then, add those required for Sigma_1 (unless we have matrix!)
      if (include_Sigma1 /*&& !wf.Sigma()*/) {
        const auto temp_basis = qip::merge(core_s1, excited_s1);
        std::cout << "and: " << DiracSpinor::state_config(temp_basis) << "\n"
                  << std::flush;
        const auto yk = Coulomb::YkTable(temp_basis);
        qk.fill_if(temp_basis, yk, select_Q_sigma, max_k_Coulomb, false);
      }

      // Then, add those required for Sigma_2 (unless we already did Sigma_1)
      if (include_Sigma2 && !(include_Sigma1 /*&& !wf.Sigma()*/)) {
        const auto temp_basis = qip::merge(core_s2, excited_s2);
        std::cout << "and: " << DiracSpinor::state_config(temp_basis) << "\n"
                  << std::flush;
        const auto yk = Coulomb::YkTable(temp_basis);
        qk.fill_if(temp_basis, yk, select_Q_sigma, max_k_Coulomb, false);
      }

      // print summary
      qk.summary();

      // If we calculated new integrals, write to disk
      const auto total = qk.count();
      assert(total >= existing);
      const auto new_integrals = total - existing;
      std::cout << "Calculated " << new_integrals << " new Coulomb integrals\n";
      if (new_integrals > 0) {
        qk.write(qk_filename);
      }
    }
    std::cout << "\n" << std::flush;
  }

  //----------------------------------------------------------------------------

  // Create lookup table for one-particle matrix elements, h1
  const auto h1 = frozen_string == wf.coreConfiguration() ?
                      CI::calculate_h1_table(ci_sp_basis, core_s1, excited_s1,
                                             qk, include_Sigma1) :
                      CI::calculate_h1_table_v2(ci_sp_basis, wf, frozen_core);

  //----------------------------------------------------------------------------
  // Calculate MBPT corrections to two-body Coulomb integrals
  // Fix filename: account for n_min_core!
  const auto Sk_filename = wf.atomicSymbol() + "_" +
                           std::to_string(n_min_core) + "_" +
                           DiracSpinor::state_config(excited_s2) +
                           (max_k_Coulomb >= 0 && max_k_Coulomb < 50 ?
                                "_" + std::to_string(max_k_Coulomb) :
                                "") +
                           ".sk";

  Coulomb::LkTable Sk;
  if (include_Sigma2) {
    std::cout << "Calculate two-body MBPT integrals: Σ^k_abcd\n";

    std::cout << "For: " << DiracSpinor::state_config(cis2_basis) << ", using "
              << DiracSpinor::state_config(excited_s2) << "\n";

    Sk = CI::calculate_Sk(Sk_filename, cis2_basis, core_s2, excited_s2, qk,
                          max_k_Coulomb, false);
    std::cout << "\n" << std::flush;
  }

  //----------------------------------------------------------------------------
  const auto J_even_list = input.get("J+", std::vector<int>{});
  const auto J_odd_list = input.get("J-", std::vector<int>{});
  const auto num_solutions = input.get("num_solutions", 5);

  //----------------------------------------------------------------------------
  if (write_integrals) {

    const auto suffix = include_Sigma1 && include_Sigma2 ? "_s12" :
                        include_Sigma1                   ? "_s1" :
                        include_Sigma2                   ? "_s2" :
                                                           "";
    std::string fci_file = wf.atomicSymbol() + "_FCIdump_" +
                           DiracSpinor::state_config(ci_sp_basis) + suffix +
                           ".txt";
    const bool symmetries_in_CIdump = input.get("symmetries", false);

    int n_elec = wf.Znuc();
    for (auto &n : frozen_core) {
      n_elec -= n.num_electrons();
    }
    std::cout << "Nelec = " << n_elec << "\n";

    // Write out in CI_dump format
    write_FCI_dump(fci_file, ci_sp_basis, orbital_map, qk, h1,
                   symmetries_in_CIdump, include_Sigma2 ? &Sk : nullptr,
                   n_elec);

    // Print CSFs and Hamiltonian (CI) matrix:
    std::cout << "\n";
    for (const auto J : J_even_list) {
      CI::PsiJPi psi{2 * J, +1, ci_sp_basis};
      std::string csf_file = wf.atomicSymbol() + "_" + std::to_string(J) + "+";
      fmt::print("{} CSFs for J={}, even parity: {}\n", psi.CSFs().size(), J,
                 csf_file);
      write_CSFs(psi.CSFs(), 2 * J, orbital_map, csf_file);

      // Construct the CI matrix:
      const auto Hci = include_Sigma2 ? CI::construct_Hci(psi, h1, qk, &Sk) :
                                        CI::construct_Hci(psi, h1, qk);
      write_H(Hci, csf_file);
      std::cout << "\n";
    }
    for (const auto J : J_odd_list) {
      CI::PsiJPi psi{2 * J, -1, ci_sp_basis};

      std::string csf_file = wf.atomicSymbol() + "_" + std::to_string(J) + "-";
      fmt::print("{} CSFs for J={}, odd parity: {}\n", psi.CSFs().size(), J,
                 csf_file);
      write_CSFs(psi.CSFs(), 2 * J, orbital_map, csf_file);

      // Construct the CI matrix:
      const auto Hci = include_Sigma2 ? CI::construct_Hci(psi, h1, qk, &Sk) :
                                        CI::construct_Hci(psi, h1, qk);
      write_H(Hci, csf_file);
      std::cout << "\n";
    }
  }

  //----------------------------------------------------------------------------
  std::cout << "\nRun actual CI:\n";

  // even parity:
  for (const auto J : J_even_list) {
    CI::run_CI(ci_sp_basis, int(std::round(2 * J)), +1, num_solutions, h1, qk,
               Sk, include_Sigma2);
  }

  // odd parity:
  for (const auto J : J_odd_list) {
    CI::run_CI(ci_sp_basis, int(std::round(2 * J)), -1, num_solutions, h1, qk,
               Sk, include_Sigma2);
  }
}

//==============================================================================
//==============================================================================

//==============================================================================
void write_H(const LinAlg::Matrix<double> &Hci, const std::string &csf_fname) {
  std::string ci_fname = csf_fname + "_H.txt";
  std::cout << "Writing CI matrix to file: " << ci_fname << "\n";
  std::ofstream ci_file(ci_fname);
  ci_file << "# in matrix/table form: \n";
  for (std::size_t iA = 0; iA < Hci.rows(); ++iA) {
    for (std::size_t iX = 0; iX < Hci.cols(); ++iX) {
      fmt::print(ci_file, "{:+.16e} ", Hci(iA, iX));
    }
    ci_file << "\n";
  }
}

//==============================================================================
void write_CSFs(const std::vector<CI::CSF2> &CSFs, int twoJ,
                const std::map<nkm, int> &orbital_map,
                const std::string &csf_fname) {

  std::ofstream csf_file(csf_fname + "_csf.txt");
  std::ofstream proj_file(csf_fname + "_proj.txt");

  csf_file << "# csf_index  a  b\n";
  proj_file << "# proj_index  a  b  CGC\n";
  int csf_count = 0;
  int proj_count = 0;
  for (const auto &csf : CSFs) {

    const auto v = csf.state(0);
    const auto w = csf.state(1);

    const auto [nv, kv] = Angular::index_to_nk(v);
    const auto [nw, kw] = Angular::index_to_nk(w);

    const auto tjv = Angular::twoj_k(kv);
    const auto tjw = Angular::twoj_k(kw);

    fmt::print(csf_file, "{} {} {}\n", csf_count + 1,
               DiracSpinor::shortSymbol(nv, kv),
               DiracSpinor::shortSymbol(nw, kw));
    ++csf_count;

    // Each individual m projection:
    for (int two_m_v = -tjv; two_m_v <= tjv; two_m_v += 2) {
      const auto two_m_w = twoJ - two_m_v;
      // if (std::abs(two_m_w) > tjw)
      //   continue;
      const auto cgc = Angular::cg_2(tjv, two_m_v, tjw, two_m_w, twoJ, twoJ);
      if (cgc == 0.0)
        continue;

      const auto iv = orbital_map.at(nkm{nv, kv, two_m_v});
      const auto iw = orbital_map.at(nkm{nw, kw, two_m_w});
      // if (iw > iv)
      //   continue;

      const auto eta = v == w ? 1.0 / std::sqrt(2.0) : 1.0;
      const auto d_proj = eta * cgc; //?

      fmt::print(proj_file, "{} {} {} {:.8f}\n", proj_count + 1, iv + 1, iw + 1,
                 d_proj);
      ++proj_count;
    }
  }
  std::cout << "Writing " << csf_count
            << " CSFs to file: " << (csf_fname + "_csf.txt") << "\n";
  std::cout << "Writing " << proj_count
            << " projections to file: " << (csf_fname + "_proj.txt") << "\n";
}

//==============================================================================

//==============================================================================
void write_FCI_dump(const std::string &fname,
                    const std::vector<DiracSpinor> &ci_sp_basis,
                    const std::map<nkm, int> &orbital_map,
                    const Coulomb::QkTable &qk,
                    const Coulomb::meTable<double> &h1, bool symmetries,
                    const Coulomb::LkTable *Sk, int n_elec) {

  // count the orbitals (including m projections)
  int n_orb = 0;
  for (const auto &a : ci_sp_basis) {
    n_orb += a.twojp1();
  }

  // print all two-particle integrals Rk to file:
  std::ofstream g_file(fname);

  // CIDump format header:
  fmt::print(g_file, "&FCI\n  NORB={},\n  NELEC={},", n_orb, n_elec);
  g_file << "\n  ORBMJ2=";
  for (const auto &a : ci_sp_basis) {
    for (int tma = -a.twoj(); tma <= a.twoj(); tma += 2) {
      g_file << tma << ",";
    }
  }
  g_file << "\n  ORBPI=";
  for (const auto &a : ci_sp_basis) {
    for (int tma = -a.twoj(); tma <= a.twoj(); tma += 2) {
      g_file << a.parity() << ",";
    }
  }
  // g_file << "\n  ORBJ2=";
  // for (const auto &a : ci_sp_basis) {
  //   for (int tma = -a.twoj(); tma <= a.twoj(); tma += 2) {
  //     g_file << a.twoj() << ",";
  //   }
  // }
  g_file << "\n&END\n";

  // two particle part
  for (const auto &a : ci_sp_basis) {
    for (const auto &b : ci_sp_basis) {
      for (const auto &c : ci_sp_basis) {
        for (const auto &d : ci_sp_basis) {

          // Selection rules, for g/s (not required)
          const auto [k0, k1] = Sk == nullptr ?
                                    Coulomb::k_minmax_Q(a, b, c, d) :
                                    MBPT::k_minmax_S(a, b, c, d);
          if (k1 < k0)
            continue;

          for (int tma = -a.twoj(); tma <= a.twoj(); tma += 2) {
            for (int tmb = -b.twoj(); tmb <= b.twoj(); tmb += 2) {
              for (int tmc = -c.twoj(); tmc <= c.twoj(); tmc += 2) {
                for (int tmd = -d.twoj(); tmd <= d.twoj(); tmd += 2) {

                  const auto ia =
                      (uint16_t)orbital_map.at(nkm{a.n(), a.kappa(), tma});
                  const auto ib =
                      (uint16_t)orbital_map.at(nkm{b.n(), b.kappa(), tmb});
                  const auto ic =
                      (uint16_t)orbital_map.at(nkm{c.n(), c.kappa(), tmc});
                  const auto id =
                      (uint16_t)orbital_map.at(nkm{d.n(), d.kappa(), tmd});

                  if (symmetries) {
                    // Equivilant integrals:
                    // Convert four indevidual idex's to single index:
                    // nb: this only works if largest of (ia,ib,ic,id)
                    // is smaller than 2^16, which is always true
                    const auto indexify = [](uint16_t w, uint16_t x, uint16_t y,
                                             uint16_t z) {
                      return ((uint64_t)w << 48) + ((uint64_t)x << 32) +
                             ((uint64_t)y << 16) + (uint64_t)z;
                    };
                    // abcd = badc = cdab = dcba for g
                    // abcd = badc               for s
                    const uint64_t i1 = indexify(ia, ib, ic, id);
                    const uint64_t i2 = indexify(ib, ia, id, ic);
                    const uint64_t i3 = indexify(ic, id, ia, ib);
                    const uint64_t i4 = indexify(id, ic, ib, ia);

                    // Only include the unique ones:
                    if (Sk == nullptr) {
                      // g symmetry
                      if (i1 != std::min({i1, i2, i3, i4}))
                        continue;
                    } else {
                      // s symmetry
                      if (i1 != std::min({i1, i2}))
                        continue;
                    }
                  }

                  const auto g =
                      qk.g(a, b, c, d, tma, tmb, tmc, tmd) +
                      (Sk ? Sk->g(a, b, c, d, tma, tmb, tmc, tmd) : 0.0);

                  if (g == 0.0)
                    continue;

                  // Note: "Chemist" notation:
                  // b and c interchanged, and start from 1
                  fmt::print(g_file, "{:.16e} {} {} {} {}\n", g, ia + 1, ic + 1,
                             ib + 1, id + 1);
                }
              }
            }
          }
        }
      }
    }
  }

  // one-particle part:
  for (const auto &a : ci_sp_basis) {
    for (const auto &b : ci_sp_basis) {
      // h1 is scalar operator, so kappa's must be equal!
      if (b.kappa() != a.kappa())
        continue;
      for (int twom = -a.twoj(); twom <= a.twoj(); twom += 2) {
        // m_a = m_b (h is scalar operator)

        const auto index_a = orbital_map.at(nkm{a.n(), a.kappa(), twom});
        const auto index_b = orbital_map.at(nkm{b.n(), b.kappa(), twom});

        // symmetric; only store 'smallest' index set:
        if (symmetries) {
          if (index_b < index_a)
            continue;
        }

        const auto value = h1.getv(a, b);
        if (value == 0.0)
          continue;
        fmt::print(g_file, "{:.16e} {} {} {} {}\n", value, index_a + 1,
                   index_b + 1, 0, 0);
      }
    }
  }
  // nuclear repulsion
  g_file << "0.0 0 0 0 0\n";
}

} // namespace Module
