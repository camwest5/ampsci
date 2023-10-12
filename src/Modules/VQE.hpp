#pragma once
#include "CI/CSF.hpp"
#include "Coulomb/QkTable.hpp"
#include "Coulomb/meTable.hpp"
#include "DiracOperator/DiracOperator.hpp"
#include "LinAlg/Matrix.hpp"
#include "MBPT/Sigma2.hpp"
#include "Wavefunction/DiracSpinor.hpp"
#include "fmt/ostream.hpp"
#include <string>
#include <vector>

// Forward declare classes:
class Wavefunction;
namespace IO {
class InputBlock;
}

namespace Module {

// Stuct, hold n, kappa, and 2*m (for the orbital to index map)
struct nkm {
  int n;
  int kappa;
  int twom;

  //required for std::map (needs unique ordering; order doesn't matter)
  friend bool operator<(const nkm &lhs, const nkm &rhs) {
    if (lhs.n == rhs.n) {
      if (lhs.kappa == rhs.kappa) {
        return lhs.twom < rhs.twom;
      }
      return lhs.kappa < rhs.kappa;
    }
    return lhs.n < rhs.n;
  }
};

//! Just a test: example for playing with VQE
void VQE(const IO::InputBlock &input, const Wavefunction &wf);

//------------------------------------------------------------------------------
void write_CSFs(const std::vector<CI::CSF2> &CSFs, int twoJ,
                const std::map<nkm, int> &orbital_map,
                const std::string &csf_fname);

void write_H(const LinAlg::Matrix<double> &Hci, const std::string &csf_fname);

//------------------------------------------------------------------------------
template <class Integrals>
void write_CoulombIntegrals(const std::string &fname,
                            const std::vector<DiracSpinor> &ci_sp_basis,
                            const std::map<nkm, int> &orbital_map,
                            const Integrals &qk, bool ci_dump_format = false,
                            const Coulomb::meTable<double> &h1 = {},
                            int twoJ = 0, int parity = 0) {

  static_assert(std::is_same_v<Integrals, Coulomb::QkTable> ||
                std::is_same_v<Integrals, Coulomb::LkTable>);

  int n_orb = 0;
  for (const auto &a : ci_sp_basis) {
    n_orb += a.twojp1();
  }

  // print all two-particle integrals Rk to file:
  std::ofstream g_file(fname);
  if (ci_dump_format) {
    assert((std::is_same_v<Integrals, Coulomb::QkTable>));
    // I don't know what these all really mean...
    fmt::print(
        g_file,
        //  "&FCI\n  NORB={},\n  NELEC=2,\n  MJ2={},\n  PI={},\n  &END\n",
        "&FCI\n  NORB={},\n  NELEC=2,\n  MJ2={},\n  PI={},\n  ORBMJ2=", n_orb,
        twoJ, parity);
    for (const auto &a : ci_sp_basis) {
      for (int tma = -a.twoj(); tma <= a.twoj(); tma += 2) {
        g_file << tma << ",";
      }
    }
    g_file << "\n&END\n";
  } else {
    if (std::is_same_v<Integrals, Coulomb::QkTable>) {
      g_file
          << "# a  b  c  d  g_abcd    ## (nb: abcd = badc = cdab = dcba; only "
             "'smallest' is written)\n";
    } else {
      g_file << "# a  b  c  d  s_abcd    ## (nb: abcd = badc; only "
                "'smallest' is written)\n";
    }
  }

  // two particle part
  for (const auto &a : ci_sp_basis) {
    for (const auto &b : ci_sp_basis) {
      for (const auto &c : ci_sp_basis) {
        for (const auto &d : ci_sp_basis) {

          // Selection rules, for g/s
          const auto [k0, k1] = std::is_same_v<Integrals, Coulomb::QkTable> ?
                                    Coulomb::k_minmax_Q(a, b, c, d) :
                                    MBPT::k_minmax_S(a, b, c, d);
          // essentially J triangle rule (and parity)
          if (k1 < k0)
            continue;

          for (int tma = -a.twoj(); tma <= a.twoj(); tma += 2) {
            for (int tmb = -b.twoj(); tmb <= b.twoj(); tmb += 2) {
              for (int tmc = -c.twoj(); tmc <= c.twoj(); tmc += 2) {
                for (int tmd = -d.twoj(); tmd <= d.twoj(); tmd += 2) {

                  // m = j_z selection rules:
                  if (tmc - tma != tmb - tmd)
                    continue;

                  const auto ia =
                      (uint16_t)orbital_map.at(nkm{a.n(), a.kappa(), tma});
                  const auto ib =
                      (uint16_t)orbital_map.at(nkm{b.n(), b.kappa(), tmb});
                  const auto ic =
                      (uint16_t)orbital_map.at(nkm{c.n(), c.kappa(), tmc});
                  const auto id =
                      (uint16_t)orbital_map.at(nkm{d.n(), d.kappa(), tmd});

                  // Is this correct?
                  if (ci_dump_format) {
                    // ma+mb=Jz=J
                    if (tma + tmb != twoJ)
                      continue;
                    // parity
                    if (a.parity() * b.parity() != parity)
                      continue;
                    if (c.parity() * d.parity() != parity)
                      continue;
                    // Pauli
                    if (ia == ib || ic == id)
                      continue;
                  }

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
                  uint64_t i1 = indexify(ia, ib, ic, id);
                  uint64_t i2 = indexify(ib, ia, id, ic);
                  uint64_t i3 = indexify(ic, id, ia, ib);
                  uint64_t i4 = indexify(id, ic, ib, ia);

                  // Only include the unique ones:
                  if (std::is_same_v<Integrals, Coulomb::QkTable>) {
                    // g symmetry
                    if (i1 != std::min({i1, i2, i3, i4}))
                      continue;
                  } else {
                    // s symmetry
                    if (i1 != std::min({i1, i2}))
                      continue;
                  }

                  const auto g = qk.g(a, b, c, d, tma, tmb, tmc, tmd);
                  if (g == 0.0)
                    continue;

                  if (ci_dump_format) {
                    // note: b and c interchanged, and start from 1
                    fmt::print(g_file, "{:.8e} {} {} {} {}\n", g, ia + 1,
                               ic + 1, ib + 1, id + 1);
                  } else {
                    fmt::print(g_file, "{} {} {} {} {:.8e}\n", ia + 1, ib + 1,
                               ic + 1, id + 1, g);
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  // one-particle part:
  if (ci_dump_format) {
    for (const auto &a : ci_sp_basis) {
      for (const auto &b : ci_sp_basis) {
        // h1 is scalar operator, so kappa's must be equal!
        if (b.kappa() != a.kappa())
          continue;
        for (int twom = -a.twoj(); twom <= a.twoj(); twom += 2) {
          // m_a = m_b (scalar operator)

          const auto index_a = orbital_map.at(nkm{a.n(), a.kappa(), twom});
          const auto index_b = orbital_map.at(nkm{a.n(), a.kappa(), twom});
          // symmetric; only store 'smallest' index set:
          if (index_b < index_a)
            continue;

          const auto value = h1.getv(a, b);
          if (value == 0.0)
            continue;
          fmt::print(g_file, "{:.8e} {} {} {} {}\n", value, index_a + 1,
                     index_b + 1, 0, 0);
        }
      }
    }
    g_file << "0.0 0 0 0 0\n";
  }
}

} // namespace Module
