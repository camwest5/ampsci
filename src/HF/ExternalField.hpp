#pragma once
#include "Angular/Angular_tables.hpp"
#include <vector>
class DiracSpinor;
namespace DiracOperator {
class TensorOperator;
}
class Wavefunction;
namespace MBPT {
class CorrelationPotential;
}

namespace HF {

enum class dPsiType { X, Y };

//! @brief
//! Uses time-dependent Hartree-Fock method to include core-polarisation
//! (RPA) corrections to matrix elements of some external field operator.

/*! @details
Solves set of TDHF equations
\f[ (H -\epsilon \pm \omega)\delta\psi_b = -(\delta V  + \delta\epsilon_c)\psi_b
\f] self consistantly for each electron in the core to determine dV. (See
'Method' document for detailed physics description). There is an option to limit
the maximum number of iterations; set to 1 to get the first-order correction
(nb: no damping is used for first iteration).
\par Construction
Requires a pointer to an operator (h), a set of core orbitals [taken as const
reference], a local potential (vl, typically vnuc + vdir), and the value of
alpha (fine structure constant).
\par Usage
solve_TDHFcore(omega) solves TDHF eqs for given frequency. Frequency should be
positive, but is allowed to be negative (use as a test only, with care). Can be
run again with a different frequency, typically does not need to be re-started
from scratch. Then, dV_ab(Fa,Fb) returns the correction to the matrix element:
\f[ \langle \phi_a || \delta V || \phi_b \rangle \f]
*/

class ExternalField {
public:
  ExternalField(const DiracOperator::TensorOperator *const h,
                const std::vector<DiracSpinor> &core,
                const std::vector<double> &vl, const double alpha);

private:
  // dPhi = X exp(-iwt) + Y exp(+iwt)
  // (H - e - w)X = -(h + dV - de)Phi
  // (H - e + w)Y = -(h* + dV* - de)Phi
  // X_c = sum_x X_x,
  // j(x)=j(c)-k,...,j(c)+k.  And: pi(x) = pi(c)*pi(h)
  std::vector<std::vector<DiracSpinor>> m_X = {};
  std::vector<std::vector<DiracSpinor>> m_Y = {};
  // can just write these to disk! Read them in, continue as per normal

  const DiracOperator::TensorOperator *const m_h; //??
  const std::vector<DiracSpinor> *const p_core;
  const std::vector<double> m_vl; // Add H_mag ?
  const double m_alpha;
  const int m_rank;
  const int m_pi;
  const bool m_imag;
  double m_core_eps = 1.0;

  // Angular::SixJ m_6j; // used?

public:
  //! @brief Solves TDHF equations self-consistantly for core electrons at
  //! frequency omega.
  //! @details Solves TDHF equations self-consistantly for core electrons at
  //! frequency omega. Will iterate up to a maximum of max_its. Set max_its=1 to
  //! get first-order correction [note: no dampling is used for first
  //! itteration]. If print=true, will write progress to screen
  void solve_TDHFcore(const double omega, int max_its = 100,
                      const bool print = true);

  double get_eps() const { return m_core_eps; }

  //! @brief Uses itterative matrix method; for tests only
  void solve_TDHFcore_matrix(const Wavefunction &wf, const double omega,
                             const int max_its = 25);

  //! @brief Clears the dPsi orbitals (sets to zero)
  void clear_dPsi();

  //! @brief Calculate reduced matrix element <a||dV||b> or <a||dV*||b>.
  //! Will exclude orbital 'Fexcl' from sum over core (for tests only)
  double dV_ab(const DiracSpinor &Fa, const DiracSpinor &Fb, bool conj,
               const DiracSpinor *const Fexcl = nullptr) const;

  //! @brief As above, but automatically determines if 'conjugate' version
  //! reuired (Based on sign of [en_a-en_b])
  double dV_ab(const DiracSpinor &Fa, const DiracSpinor &Fb) const;

  //! @brief Returns "reduced partial matrix element RHS": dV||Fb}.
  //! Note: Fa * dV_ab_rhs(..) equiv to dV_ab(..)
  DiracSpinor dV_ab_rhs(const int kappa_n, const DiracSpinor &Fm,
                        bool conj = false,
                        const DiracSpinor *const Fexcl = nullptr) const;

  //! @brief Returns const reference to dPsi orbitals for given core orbital Fc
  const std::vector<DiracSpinor> &get_dPsis(const DiracSpinor &Fc,
                                            dPsiType XorY) const;
  //! @brief Returns const reference to dPsi orbital of given kappa
  const DiracSpinor &get_dPsi_x(const DiracSpinor &Fc, dPsiType XorY,
                                const int kappa_x) const;

  //! Forms \delta Psi_v for valence state Fv (including core pol.) - 1 kappa
  //! @details
  //!  Solves
  //! \f[ (H + \Sigma - \epsilon - \omega)X = -(h + \delta V
  //! - \delta\epsilon)\psi \f]
  //! or
  //! \f[ (H + \Sigma - \epsilon + \omega)Y = -(h^\dagger + \delta V^\dagger
  //!   - \delta\epsilon)Psi\f]
  //! Returns \f$ \chi_\beta \f$ for given kappa_beta, where
  //! \f[ X_{j,m} = (-1)^{j_\beta-m}tjs(j,k,j;-m,0,m)\chi_j \f]
  //! XorY takes values: HF::dPsiType::X or HF::dPsiType::Y
  DiracSpinor
  solve_dPsi(const DiracSpinor &Fv, const double omega, dPsiType XorY,
             const int kappa_beta,
             const MBPT::CorrelationPotential *const Sigma = nullptr) const;
  //! Forms \delta Psi_v for valence state Fv for all kappas (see solve_dPsi)
  std::vector<DiracSpinor>
  solve_dPsis(const DiracSpinor &Fv, const double omega, dPsiType XorY,
              const MBPT::CorrelationPotential *const Sigma = nullptr) const;

  //! @brief Writes dPsi (f-component) to textfile
  void print(const std::string &ofname = "dPsi.txt") const;

private:
  // Calculate indevidual (4 electron) partial contributions to the
  // dV (reduced) matrix element (for Matrix method: not used yet)
  double dX_nm_bbe_rhs(const DiracSpinor &Fn, const DiracSpinor &Fm,
                       const DiracSpinor &Fb, const DiracSpinor &X_beta) const;
  double dY_nm_bbe_rhs(const DiracSpinor &Fn, const DiracSpinor &Fm,
                       const DiracSpinor &Fb, const DiracSpinor &Y_beta) const;

private:
  std::size_t core_index(const DiracSpinor &Fc) const;

public:
  ExternalField &operator=(const ExternalField &) = delete;
  ExternalField(const ExternalField &) = default;
  ~ExternalField() = default;
};

} // namespace HF
