#pragma once
#include "CI/CSF.hpp"
#include "Coulomb/QkTable.hpp"
#include "Coulomb/meTable.hpp"
#include "DiracOperator/DiracOperator.hpp"
#include "LinAlg/Matrix.hpp"
#include "MBPT/Sigma2.hpp"
#include "Wavefunction/DiracSpinor.hpp"
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
void write_FCI_dump(const std::string &fname,
                    const std::vector<DiracSpinor> &ci_sp_basis,
                    const std::map<nkm, int> &orbital_map,
                    const Coulomb::QkTable &qk,
                    const Coulomb::meTable<double> &h1 = {},
                    bool symmetries = false,
                    const Coulomb::LkTable *Sk = nullptr, int n_elec = 2);

} // namespace Module
