#pragma once
#include "Angular/Wigner369j.hpp"
#include "DiracOperator/TensorOperator.hpp"
#include "IO/InputBlock.hpp"
#include "Potentials/BSM_Vee.hpp"
#include "Wavefunction/Wavefunction.hpp"

namespace DiracOperator {

//==============================================================================

class V_SP final : public TensorOperator {
public:
  V_SP(const std::vector<DiracSpinor> &core_in, const double mu_in)
      : TensorOperator(0, Parity::odd), m_core(core_in), m_mu(mu_in) {}
  double angularF(const int ka, const int kb) const override final {

    // Check this!
    return 1.0 / Angular::threej_2(Angular::twoj_k(ka), 0, Angular::twoj_k(kb),
                                   -1, 0, 1);
  }
  std::string name() const override { return std::string("V_SP"); }

  DiracSpinor radial_rhs(const int kappa_a,
                         const DiracSpinor &Fb) const override final {
    // Write new radial integral here
    // NB: v -> b and n -> a from my derivation

    // NB: Will be factored by some 3js etc. - convert to RME form, not full ME

    const double gghc = 1.0;
    // This is what I've just derived
    return Vee::V_SP_Fv(m_core, Fb, kappa_a, gghc, m_mu);
  }

  double radialIntegral(const DiracSpinor &Fa,
                        const DiracSpinor &Fb) const override final {
    return Fa * radial_rhs(Fa.kappa(), Fb);
  }

private:
  const std::vector<DiracSpinor> m_core;
  const double m_mu;
};

//==============================================================================

inline std::unique_ptr<DiracOperator::TensorOperator>
generate_V_SP(const IO::InputBlock &input, const Wavefunction &wf) {
  using namespace DiracOperator;
  input.check({{"no options", ""}});
  if (input.has_option("help")) {
    return nullptr;
  }
  return std::make_unique<V_SP>(wf.core(), 1.0);
}

} // namespace DiracOperator
