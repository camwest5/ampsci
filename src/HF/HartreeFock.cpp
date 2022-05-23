#include "HF/HartreeFock.hpp"
#include "Angular/Wigner369j.hpp"
#include "Coulomb/CoulombIntegrals.hpp"
#include "Coulomb/YkTable.hpp"
#include "DiracODE/DiracODE.hpp"
#include "HF/Breit.hpp"
#include "MBPT/CorrelationPotential.hpp"
#include "Maths/Grid.hpp"
#include "Physics/Parametric_potentials.hpp"
#include "Physics/RadPot.hpp"
#include "Wavefunction/DiracSpinor.hpp"
#include "Wavefunction/Wavefunction.hpp"
#include "qip/Vector.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

namespace HF {

//==============================================================================
Method parseMethod(const std::string &in_method) {
  if (in_method == "HartreeFock")
    return Method::HartreeFock;
  if (in_method == "ApproxHF")
    return Method::ApproxHF;
  if (in_method == "Hartree")
    return Method::Hartree;
  if (in_method == "KohnSham")
    return Method::KohnSham;
  if (in_method == "Local")
    return Method::Local;
  std::cout << "Warning: HF Method: " << in_method << " ?? Defaulting to HF\n";
  return Method::HartreeFock;
}

std::string parseMethod(const Method &in_method) {
  if (in_method == Method::HartreeFock)
    return "HartreeFock";
  if (in_method == Method::ApproxHF)
    return "ApproxHF";
  if (in_method == Method::Hartree)
    return "Hartree";
  if (in_method == Method::KohnSham)
    return "KohnSham";
  if (in_method == Method::Local)
    return "Local";
  return "HartreeFock";
}

//==============================================================================
//==============================================================================
HartreeFock::HartreeFock(std::shared_ptr<const Grid> grid,
                         std::vector<double> vnuc,
                         std::vector<DiracSpinor> core,
                         std::optional<QED::RadPot> vrad, double alpha,
                         Method method, double x_Breit, double in_eps)
    : m_rgrid(grid),
      m_core(std::move(core)),
      m_vnuc(std::move(vnuc)),
      m_vrad(vrad),
      m_VBr(x_Breit == 0.0 ? std::optional<HF::Breit>{} :
                             std::optional<HF::Breit>(x_Breit)),
      m_alpha(alpha),
      m_method(method),
      m_eps_HF(std::abs(in_eps) < 1.0 ? in_eps : std::pow(10, -in_eps)),
      m_vdir(m_rgrid->num_points()),
      m_Yab(m_core) {}

//==============================================================================
double HartreeFock::solve_core(bool print) {

  if (m_core.empty())
    return 0.0;

  // have optional inputs! XXX
  set_parametric_potential();

  auto eps_its = EpsIts{};

  std::string text;
  switch (m_method) {

  case Method::HartreeFock:
    // Full Hartree-Fock
    text = "Hartree-Fock\n";
    solve_initial_core(1.0e-5);
    hf_approx_core(1.0e-8);
    eps_its = hartree_fock_core();
    break;

  case Method::ApproxHF:
    // "Approximate" Hartree-Fock (with localised exchange)
    text = "Approximate (localised) Hartree-Fock\n";
    solve_initial_core(1.0e-7);
    eps_its = hf_approx_core(m_eps_HF);
    break;

  case Method::Hartree:
    // Core-Hartree method (no exchange, core has self-interaction)
    // XXX What about V^N approx?
    text = "Hartree (No exchange)\n";
    solve_initial_core(1.0e-7);
    eps_its = local_core(m_eps_HF);
    break;

  case Method::KohnSham:
    // Kohn-Sham method (including Latter correction)
    text = "Kohn-Sham\n";
    solve_initial_core(1.0e-7);
    eps_its = local_core(m_eps_HF);
    break;

  case Method::Local:
    // Local (parametric potential) method
    // For now, just default parameters. Add option! XXX + output!
    text = "Local (parametric) potential\n";
    eps_its = solve_initial_core(1.0e-16);
    // Calculate Y^k_ab integrals, since they otherwise aren't
    // (though very unlikely they will be used)
    m_Yab.calculate(m_core);
    break;
  }

  const auto [eps, its, nk] = eps_its;
  if (print) {
    std::cout << text;
    printf("%-7s:  it:%3i eps=%6.1e for %s\n", "Core", its, eps, nk.c_str());
  }

  if (eps > m_eps_HF && eps > 1.0e-6) {
    std::cout << "\n⚠️ WARNING: Core didn't converge!\n\n";
  }
  return eps;
}

//==============================================================================
void HartreeFock::solve_valence(
    std::vector<DiracSpinor> *valence, const bool print,
    const MBPT::CorrelationPotential *const Sigma) const {

  if (valence == nullptr || valence->empty())
    return;

  std::vector<EpsIts> eis(valence->size());
  std::vector<double> deltas(valence->size());

  if (Sigma && print) {
    std::cout << "Solving for Brueckner orbitals (correlation potential)\n";
    Sigma->print_scaling();
    std::cout << std::flush;
  }

#pragma omp parallel for
  for (std::size_t i = 0; i < valence->size(); i++) {
    auto &Fa = (*valence)[i];
    const auto en0 = Fa.en();
    if (m_method == Method::HartreeFock) {
      eis[i] = hf_valence(Fa, Sigma);
    } else {
      eis[i] = local_valence(Fa);
    }
    deltas.at(i) = Fa.en() - en0;
  }

  const auto worst = std::max_element(eis.cbegin(), eis.cend());
  const auto best = std::min_element(eis.cbegin(), eis.cend());
  const auto &[eps, its, state] = *worst;
  const auto &[beps, bits, bstate] = *best;

  if (Sigma && print) {
    // For Brueckner, print each valence
    for (std::size_t i = 0; i < valence->size(); i++) {
      const auto &[veps, vits, vstate] = eis.at(i);
      const auto delta = deltas.at(i);
      printf(" delta=%8.5f; eps=%5.0e [its=%3i]", delta, veps, vits);
      if (veps > m_eps_HF && veps > 1.0e-6) {
        std::cout << " ====*";
      }
      std::cout << "\n";
    }
  }

  if (print)
    printf("%-7s:  it:%3i eps=%6.1e for %s [%3i eps=%5.0e for %s]\n", "Val(Br)",
           its, eps, state.c_str(), bits, beps, bstate.c_str());
  if (eps > m_eps_HF && eps > 1.0e-6) {
    std::cout << "\n⚠️ WARNING: Valence didn't converge!\n\n";
  }
}

//==============================================================================
//==============================================================================
//==============================================================================
EpsIts HartreeFock::solve_initial_core(double eps_target) {
  double eps = 0.0;
  int its = 0;
  std::string symbol{};

  const auto *prev_Fa = &m_core.front();
  for (auto &Fa : m_core) {
    const auto &vrad_mag = get_Hrad_mag(Fa.l());
    const auto v = vlocal(Fa.l());

    double en0 = enGuessCore(Fa.n(), Fa.kappa());

    if (Fa.kappa() < 0 && Fa.kappa() != -1) {
      en0 = 0.95 * prev_Fa->en();
      if (en0 > 0)
        en0 = enGuessCore(Fa.n(), Fa.kappa());
    }

    DiracODE::boundState(Fa, en0, v, vrad_mag, m_alpha, eps_target);

    prev_Fa = &Fa;
    if (Fa.eps() > eps) {
      eps = Fa.eps();
      its = Fa.its();
      symbol = Fa.shortSymbol();
    }
  }
  return {eps, its, symbol};
}

//==============================================================================
EpsIts HartreeFock::local_core(const double eps_target_HF) {

  if (m_core.empty()) {
    return {};
  }

  const auto eta_damp = 0.5;

  // Start the itterative procedure:
  int hits = 1;
  double t_eps = 1.0;
  std::string worst{};
  for (; hits < m_max_hf_its; hits++) {
    using namespace qip::overloads;

    // Store old vdir/vex
    const auto vdir_old = m_vdir;
    // Form new v_dir, and damp:
    update_vdir();
    m_vdir = (1.0 - eta_damp) * m_vdir + eta_damp * vdir_old;

    // Solve Dirac Eq. for each state in core, using Vdir (incl KS):
    t_eps = 0;
    for (auto &Fa : m_core) {
      const double en_old = Fa.en();
      const auto dEa = Fa * ((m_vdir - vdir_old) * Fa);
      const double en_guess = (en_old < -dEa) ? en_old + dEa : en_old;

      const auto vrad_mag = get_Hrad_mag(Fa.l());
      const auto v = vlocal(Fa.l());
      DiracODE::boundState(Fa, en_guess, v, vrad_mag, m_alpha, 1.0e-7);
      const double state_eps = std::abs((Fa.en() - en_old) / en_old);
      if (state_eps > t_eps) {
        t_eps = state_eps;
        worst = Fa.shortSymbol();
      }
    }
    if (t_eps < eps_target_HF)
      break;
  }

  // Now, re-solve core orbitals with higher precission
  for (auto &Fa : m_core) {
    const auto &vrad_mag = get_Hrad_mag(Fa.l());
    const auto v = vlocal(Fa.l());
    DiracODE::boundState(Fa, Fa.en(), v, vrad_mag, m_alpha, 1.0e-15);
  }

  return {t_eps, hits, worst};
}

//==============================================================================
// Solves "approximate" (localised) HF equations for the core
EpsIts HartreeFock::hf_approx_core(const double eps_target_HF) {
  if (m_core.empty()) {
    return {};
  }

  using namespace qip::overloads;

  const auto eta_damp = 0.55;

  // Start the HF itterative procedure:
  int hits = 1;
  double eps = 1.0;
  std::string worst{};
  std::vector<std::vector<double>> vex_core(m_core.size());
  for (; hits < m_max_hf_its; hits++) {

    // Store old vdir/vex
    const auto vdir_old = m_vdir;
    const auto vex_old = vex_core;

    // Form new v_dir and v_ex:
    update_vdir();
    vex_core = form_approx_vex_core();

    // damp vdir and vex
    m_vdir = (1.0 - eta_damp) * m_vdir + eta_damp * vdir_old;
    for (std::size_t i = 0; i < m_core.size(); i++) {
      vex_core[i] = (1.0 - eta_damp) * vex_core[i] + eta_damp * vex_old[i];
    }

    // Solve Dirac Eq. for each state in core, using Vdir+Vex:
    for (std::size_t i = 0; i < m_core.size(); i++) {
      auto &Fa = m_core[i];
      const double en_old = Fa.en();
      // Guess energy correction [speeds up boundState()]
      const double dEa =
          Fa * ((m_vdir - vdir_old + vex_core[i] - vex_old[i]) * Fa);
      const double en_guess = (en_old < -dEa) ? en_old + dEa : en_old;
      // Solve Dirac equation:
      const auto Hmag = get_Hrad_mag(Fa.l());
      const auto v = vlocal(Fa.l()) + vex_core[i];
      DiracODE::boundState(Fa, en_guess, v, Hmag, m_alpha, 1.0e-7);
      const double state_eps = std::abs((Fa.en() - en_old) / en_old);
      if (state_eps > eps) {
        eps = state_eps;
        worst = Fa.shortSymbol();
      }
    }

    if (eps < eps_target_HF)
      break;
  }

  // Now, re-solve core orbitals with higher precission
  for (std::size_t i = 0; i < m_core.size(); i++) {
    auto &Fa = m_core[i];
    const auto &vrad_el = get_Hrad_el(Fa.l());
    const auto &vrad_mag = get_Hrad_mag(Fa.l());
    const auto v = qip::add(m_vnuc, m_vdir, vrad_el, vex_core[i]);
    DiracODE::boundState(Fa, Fa.en(), v, vrad_mag, m_alpha, 1.0e-14);
  }

  return {eps, hits, worst};
}

//==============================================================================
EpsIts HartreeFock::hartree_fock_core() {
  if (m_core.empty()) {
    return {};
  }

  const double eps_target = m_eps_HF;
  const auto damper = 0.35;

  std::vector<double> vl(m_rgrid->num_points()); // Vnuc + fVd
  std::vector<double> v0(m_rgrid->num_points()); // (1-f)Vd
  const auto Ncore = std::accumulate(
      m_core.cbegin(), m_core.cend(), 0,
      [](int sum, const auto &Fa) { return sum + Fa.num_electrons(); });
  const auto f_core_tmp = double(Ncore - 1) / double(Ncore);
  const auto f_core = 0.5 * (1.0 + f_core_tmp);
  const auto &vd = m_vdir;

  // Store arrays of intitial Fa and VexPsi, and VdirPsi (for En guess)
  // And allocate arrays for VexPsi, so can //-ise it loop (over orbs)!
  const auto core_zero = m_core;
  auto core_prev = m_core;
  std::vector<DiracSpinor> vexCore_zero;
  const auto vd0 = m_vdir;
  std::vector<DiracSpinor> vexF_list;
  const auto num_core_states = m_core.size();
  std::vector<double> eps_lst(num_core_states, 0.0);
  for (std::size_t i = 0; i < num_core_states; ++i) {
    auto &Fa = m_core[i];
    vexCore_zero.push_back(form_approx_vex_core_a(Fa) * Fa);
    vexF_list.push_back(DiracSpinor(Fa.n(), Fa.kappa(), Fa.grid_sptr()));
  }

  double eps = 0.0;
  std::size_t worst_index = 0;
  int it = 0;
  for (; it <= m_max_hf_its; it++) {
    const auto a_damp = damper;

    // re-calculate each Vl = vnuc + fvdir, v0 = (1-f)vdir:
    for (auto i = 0ul; i < m_rgrid->num_points(); i++) {
      vl[i] = (m_vnuc)[i] + f_core * vd[i];
      v0[i] = (1.0 - f_core) * vd[i];
    }

    // re-calculate each VexPsi:
    for (std::size_t i = 0; i < num_core_states; ++i) {
      const auto &Fa = m_core[i];
      vex_Fa_core(Fa, vexF_list[i]);
    }

    core_prev = m_core;

#pragma omp parallel for
    for (std::size_t i = 0; i < num_core_states; ++i) {
      auto &Fa = m_core[i];
      const auto &Fzero = core_zero[i];
      const auto &vexFzero = vexCore_zero[i];

      const auto Fa_prev = core_prev[i];
      const auto &VxFa = vexF_list[i];

      auto en = Fzero.en() + (Fzero * VxFa - Fa * vexFzero + Fzero * (vd * Fa) -
                              Fa * (vd0 * Fzero)) /
                                 (Fa * Fzero);
      auto v_nonlocal = v0 * Fa + VxFa;
      if (m_VBr) {
        const auto VbrFa = m_VBr->VbrFa(Fa, core_prev);
        en += (Fzero * VbrFa) / (Fa * Fzero);
        v_nonlocal += VbrFa;
      }

      const auto &Hrad_el = get_Hrad_el(Fa.l());
      const auto &Hmag = get_Hrad_mag(Fa.l());
      const auto &VlVr = qip::add(vl, Hrad_el);
      hf_orbital_green(Fa, en, VlVr, Hmag, v_nonlocal, core_prev, v0,
                       get_Breit());
      Fa = (1.0 - a_damp) * Fa + a_damp * Fa_prev;
      Fa.normalise();
      const auto d_eps = std::abs((Fa_prev.en() - Fa.en()) / Fa.en());
      eps_lst[i] = d_eps;
    }

    eps = eps_lst[0];
    for (std::size_t i = 1; i < num_core_states; ++i) {
      const auto t_eps = eps_lst[i];
      if (t_eps >= eps) {
        eps = t_eps;
        worst_index = i;
      }
    }

    const bool converged = (eps <= eps_target && it > 0);
    if (converged || it == m_max_hf_its)
      break;

    update_vdir();
  }

  return {eps, it, m_core[worst_index].shortSymbol()};
}

//==============================================================================
EpsIts HartreeFock::local_valence(DiracSpinor &Fa) const {
  using namespace qip::overloads;

  const auto eps_target = 0.01 * m_eps_HF;
  const auto eta_damp = 0.4;

  const auto Hmag = get_Hrad_mag(Fa.l());
  const auto vl = vlocal(Fa.l());

  if (Fa.en() == 0.0) {
    Fa.en() = enGuessVal(Fa.n(), Fa.kappa());
    DiracODE::boundState(Fa, Fa.en(), vl, Hmag, m_alpha, 1.0e-15);
  }

  if (excludeExchangeQ()) {
    return {0.0, 0, Fa.shortSymbol()};
  }

  auto prev_en = Fa.en();
  int it = 1;
  double eps = 1.0;
  for (; it <= m_max_hf_its; ++it) {

    const auto vlx = vl + vex_approx(Fa, m_core);
    const auto Fa_prev = Fa;
    DiracODE::boundState(Fa, Fa.en(), vlx, Hmag, m_alpha, 1.0e-15);
    eps = std::abs((prev_en - Fa.en()) / Fa.en());
    prev_en = Fa.en();

    const bool converged = (eps <= eps_target && it > 0);
    if (converged || it == m_max_hf_its)
      break;

    Fa = (1.0 - eta_damp) * Fa + eta_damp * Fa_prev;
    Fa.normalise();
  }

  return {eps, it, Fa.shortSymbol()};
}

//==============================================================================
EpsIts
HartreeFock::hf_valence(DiracSpinor &Fa,
                        const MBPT::CorrelationPotential *const Sigma) const {

  if (m_core.empty())
    return local_valence(Fa);

  const auto eps_target = 0.001 * m_eps_HF;
  const auto eta_damp = 0.4;

  const auto Hmag = get_Hrad_mag(Fa.l());
  const auto vl = vlocal(Fa.l());

  if (Fa.en() == 0.0) {
    Fa.en() = enGuessVal(Fa.n(), Fa.kappa());
    DiracODE::boundState(Fa, Fa.en(), vl, Hmag, m_alpha, 1.0e-15);
  }

  auto prev_en = Fa.en();
  int it = 1;
  double eps = 1.0;
  for (; it <= m_max_hf_its; ++it) {

    auto VxFa = vexFa(Fa);
    if (m_VBr) {
      VxFa += m_VBr->VbrFa(Fa, m_core);
    }
    if (Sigma) {
      VxFa += (*Sigma)(Fa);
    }
    const auto Fa_prev = Fa;
    DiracODE::boundState(Fa, Fa.en(), vl, Hmag, m_alpha, 1.0e-15, &VxFa,
                         &Fa_prev, zion());
    eps = std::abs((prev_en - Fa.en()) / Fa.en());
    prev_en = Fa.en();

    const bool converged = (eps <= eps_target && it > 1);
    if (converged || it == m_max_hf_its)
      break;

    Fa = (1.0 - eta_damp) * Fa + eta_damp * Fa_prev;
    Fa.normalise();
  }

  return {eps, it, Fa.shortSymbol()};
}

//==============================================================================

//==============================================================================
double HartreeFock::calculateCoreEnergy() const
// Calculates the total HF core energy:
//   E = \sum_a [ja]e_a - 0.5 \sum_(ab) (R^0_abab - \sum_k L^k_ab R^k_abba)
// where:
//   R^k_abcd = Integral [f_a*f_c + g_a*g_c] * v^k_bd
//   R^0_abab is not absymmetric
//   R^k_abba _is_ ab symmetric
{
  double Etot = 0.0;
  for (const auto &Fa : m_core) {
    const auto tja = Fa.twoj();

    double e1 = 0.0, e2 = 0.0, e3 = 0.0;
    const double xtjap1 = (tja + 1) * Fa.occ_frac();
    e1 += xtjap1 * Fa.en();
    for (const auto &Fb : m_core) {
      const auto tjb = Fb.twoj();
      const double xtjbp1 = (tjb + 1) * Fb.occ_frac();
      const auto &v0bb = *m_Yab.get(0, Fb, Fb); // XXX may be null
      const auto R0fg2 = Fa * (v0bb * Fa);
      e2 += xtjap1 * xtjbp1 * R0fg2;
      // take advantage of symmetry for third term:
      if (Fb > Fa)
        continue;
      const auto y = (Fa == Fb) ? 1.0 : 2.0; // symmetry
      const int kmin = std::abs(tja - tjb) / 2;
      const int kmax = (tja + tjb) / 2;
      // const auto &vabk = m_Yab.get_y_ab(Fa, Fb);
      for (auto k = kmin; k <= kmax; k++) {
        const auto Labk = m_Yab.Ck().get_Lambdakab(k, Fa.kappa(), Fb.kappa());
        if (Labk == 0)
          continue;
        const auto vabk = m_Yab.get(k, Fa, Fb);
        if (vabk == nullptr)
          continue;
        const auto R0fg3 = Fa * (*vabk * Fb);
        e3 += y * xtjap1 * xtjbp1 * Labk * (R0fg3);
      }
    }
    Etot += e1 - 0.5 * (e2 - e3);
  }
  return Etot;
}

//==============================================================================
std::vector<double> HartreeFock::vlocal(int l) const {
  const auto &vrad_el = get_Hrad_el(l);
  return qip::add(m_vnuc, m_vdir, vrad_el);
}

//==============================================================================
DiracSpinor HartreeFock::VBr(const DiracSpinor &Fv) const {
  if (m_VBr)
    return m_VBr->VbrFa(Fv, m_core);
  else
    return 0.0 * Fv;
}

//==============================================================================
int HartreeFock::num_core_electrons() const {
  return std::accumulate(
      m_core.cbegin(), m_core.cend(), 0,
      [](int n, const auto &Fa) { return n + Fa.num_electrons(); });
}

//==============================================================================
void HartreeFock::update_vdir(ReScale re_scale)
// Forms the direct part of the potential.
// If re_scale==true, will scale by (N-1)/N. This then given the averaged
// Hartree potential (local, same each state, no exchange).
// re_scale=false by default
{
  m_Yab.calculate(m_core);
  using namespace qip::overloads;
  m_vdir *= 0.0;

  // Rescale so that V_nuc + Vdir = -1/r at large r
  // (Only done in initial approx., before exchange)
  const double scale =
      re_scale == ReScale::yes ? (1.0 - 1.0 / num_core_electrons()) : 1.0;

  for (const auto &Fb : m_core) {
    const double f_sf = scale * Fb.twojp1() * Fb.occ_frac();
    const auto v0bb = m_Yab.get(0, Fb, Fb);
    assert(v0bb != nullptr && "Missing Fb from m_Yab in HF::update_vdir?");
    m_vdir += f_sf * (*v0bb);
  }

  if (m_method == Method::KohnSham) {
    add_KohnSham_vdir_addition();
  }
}

//==============================================================================
void HartreeFock::add_KohnSham_vdir_addition() {

  const auto f =
      -(2.0 / 3.0) * std::pow(81.0 / (32.0 * M_PI * M_PI), 1.0 / 3.0);

  std::vector<double> rho(m_rgrid->num_points());
  for (const auto &Fc : m_core) {
    rho = qip::add(rho, Fc.rho());
  }

  for (std::size_t i = 0; i < m_rgrid->num_points(); ++i) {
    const auto r = m_rgrid->r(i);
    m_vdir.at(i) += (f / r) * std::pow(r * rho[i], 1.0 / 3.0);
  }

  // KS is a V^N (not V^N-1) approximation
  const auto z_ion = zion() + 1;
  // Latter correction:
  // Enforce V(r) = Vnuc(r)+Vel(r) =~ -1/r at large r
  for (std::size_t i = m_rgrid->num_points() - 1; i != 0; --i) {
    // nb: miss i=0, but fine. Only applies large r[i]
    const auto vn = (m_vnuc)[i];
    const auto r = m_rgrid->r(i);
    if (r * std::abs(vn + m_vdir.at(i)) > z_ion)
      break;
    m_vdir.at(i) = -z_ion / r - vn;
  }
}

//==============================================================================
void HartreeFock::set_parametric_potential(Parametric::Type potential,
                                           double H_g, double d_t) {

  const auto z = int(std::round(-1.0 * m_vnuc.back() * m_rgrid->r().back()));

  // If H,d (or g,t) not given, set default parameters
  if (H_g == 0.0 || d_t == 0.0) {
    // If using local method, choose params that give good valence states
    // compared to experiment. If not using Local, then this is just starting
    // approximation, so use parameters which give good approx to HF core.
    if (m_method == Method::Local) {
      switch (potential) {
      case Parametric::Type::Green:
        Parametric::defaultGreen(z, H_g, d_t);
        break;
      case Parametric::Type::Tietz:
        Parametric::defaultTietz(z, H_g, d_t);
        break;
      }
    } else {
      // Note: If not using local, potential must be Green!
      assert(potential == Parametric::Type::Green);
      Parametric::defaultGreenCore(z, H_g, d_t);
    }
  }

  m_vdir = (potential == Parametric::Type::Green) ?
               Parametric::GreenPotential(z, m_rgrid->r(), H_g, d_t) :
               Parametric::TietzPotential(z, m_rgrid->r(), H_g, d_t);
}

//==============================================================================
void HartreeFock::form_approx_vex_core(
    std::vector<std::vector<double>> &vex) const
// Forms the 2D "approximate" exchange potential for each core state, a.
// NOTE: Must call form_vabk_core first!
// Doesn't calculate, assumes m_arr_v_abk_r array exists + is up-to-date
{
  vex.resize(m_core.size());
#pragma omp parallel for
  for (std::size_t a = 0; a < m_core.size(); a++) {
    form_approx_vex_core_a(m_core.at(a), vex[a]);
  }
}

std::vector<std::vector<double>> HartreeFock::form_approx_vex_core() const {
  std::vector<std::vector<double>> vex_core(m_core.size());
  form_approx_vex_core(vex_core);
  return vex_core;
}

std::vector<double>
HartreeFock::form_approx_vex_core_a(const DiracSpinor &Fa) const {
  std::vector<double> vex_a;
  form_approx_vex_core_a(Fa, vex_a);
  return vex_a;
}

//==============================================================================
void HartreeFock::form_approx_vex_core_a(const DiracSpinor &Fa,
                                         std::vector<double> &vex_a) const
// XXX Fa MUST be in core
// Forms the 2D "approximate" exchange potential for given core state, a.
// Does the a=b case seperately, since it's a little simpler
// Approximate:
// In order to approximate solution to HF equations, I form "local" ex.
// potential
//   [v_ex*Fa](r) = \sum_b v_ex^(a,b)(r) * Fb(r)
// v_ex is non-local; cannot write: [v_ex*Fa](r) =/= v_ex(r)*Fa(r)
// Instead: define local approx: vex_a
//   vex_a = [v_ex*Fa](r) *(Fa/Fa^2)
//         = \sum_b v_ex^(a,b)(r)*Fb(r) * (Fa/Fa^2)
//         = \sum_b v_ex^(a,b)(r)*(Fb(r)*Fa) / Fa^2
// This vex_a is then a local potential (different for each state!) that can
// be used as an addition to local direct potential to solve Dirac Eq. as
// normal. In theory, this is exact. Clearly, however, there is an issue when
// Fa is small. Luckily, however, we don't care as much when Fa is
// small! Also, since v_ex is already small (compared to vdir), we can make
// good approximation. Therefore, I only calculate vex_a when a=b, or when
// |Fa| > 1.e3 Further, largest part of v_ex is when a=b. In this case, the
// factor=1 is exact!
{
  vex_a.clear();
  vex_a.resize(m_rgrid->num_points());

  // don't subtract self-potential term for Hartree {match Core-Hartree}
  if (excludeExchangeQ())
    return;

  const auto twoj_a = Fa.twoj();

  const auto max_abs = [](double a, double b) {
    return (std::abs(a) < std::abs(b));
  };
  const auto max =
      std::abs(*std::max_element(Fa.f().begin(), Fa.f().end(), max_abs));
  const auto cut_off = 0.003 * max;

  if (!excludeExchangeQ()) {
    for (const auto &Fb : m_core) {
      // b!=a
      if (Fb == Fa)
        continue;
      const auto tjb = Fb.twoj();
      const double x_tjbp1 = (tjb + 1) * Fb.occ_frac();
      const auto irmax = std::min(Fa.max_pt(), Fb.max_pt());
      const int kmin = std::abs(twoj_a - tjb) / 2;
      const int kmax = (twoj_a + tjb) / 2;
      // const auto &vabk = m_Yab.get_y_ab(Fb, Fa);

      // hold "fraction" Fa*Fb/(Fa^2):
      std::vector<double> v_Fab(m_rgrid->num_points());
      for (std::size_t i = 0; i < irmax; i++) {
        // This is the approximte part! Divides by Fa
        if (std::abs(Fa.f(i)) < cut_off)
          continue;
        const auto fac_top = Fa.f(i) * Fb.f(i) + Fa.g(i) * Fb.g(i);
        const auto fac_bot = Fa.f(i) * Fa.f(i) + Fa.g(i) * Fa.g(i);
        v_Fab[i] = -x_tjbp1 * fac_top / fac_bot;
      } // r
      for (int k = kmin; k <= kmax; k++) {
        const auto Labk = m_Yab.Ck().get_Lambdakab(k, Fa.kappa(), Fb.kappa());
        if (Labk == 0)
          continue;
        // const auto ik = std::size_t(k - kmin);
        const auto vabk = m_Yab.get(k, Fb, Fa);
        if (vabk == nullptr)
          continue;
        for (std::size_t i = 0; i < irmax; i++) {
          vex_a[i] += Labk * (*vabk)[i] * v_Fab[i];
        } // r
      }   // k
    }     // b
  }

  // now, do a=b, ONLY if a is in the core!
  // if (a_in_coreQ)
  {
    const double x_tjap1 = (twoj_a + 1); // no occ_frac here ?
    const int kmax = twoj_a;
    // const auto &vaak = m_Yab.get_y_ab(Fa, Fa);
    const auto irmax = Fa.max_pt(); // m_rgrid->num_points();
    for (int k = 0; k <= kmax; k++) {
      const auto Labk = m_Yab.Ck().get_Lambdakab(k, Fa.kappa(), Fa.kappa());
      if (Labk == 0)
        continue;
      const auto vaak = m_Yab.get(k, Fa, Fa);
      if (vaak == nullptr)
        continue;
      for (std::size_t i = 0; i < irmax; i++) {
        // nb: If I don't 'cut' here, or fails w/ f states... ?? XX
        // Of course, cutting is fine. But WHY FAIL??
        vex_a[i] += -Labk * (*vaak)[i] * x_tjap1;
      }
    } // k
  }   // if a in core
}

//------------------------------------------------------------------------------
std::vector<double> vex_approx(const DiracSpinor &Fa,
                               const std::vector<DiracSpinor> &core, int k_cut,
                               const double lambda_cut)
// Free function
{

  std::vector<double> vex(Fa.grid().num_points());
  std::vector<double> vabk;

  const auto tja = Fa.twoj();
  const auto la = Fa.l();

  const auto max_abs = [](double a, double b) {
    return (std::abs(a) < std::abs(b));
  };
  const auto max =
      std::abs(*std::max_element(Fa.f().begin(), Fa.f().end(), max_abs));
  const auto cut_off = lambda_cut * max;

  for (const auto &Fb : core) {
    const auto tjb = Fb.twoj();
    const auto lb = Fb.l();
    const double x_tjbp1 = (tjb + 1) * Fb.occ_frac(); // when in core??
    const auto irmax = std::min(Fa.max_pt(), Fb.max_pt());
    const int kmin = std::abs(tja - tjb) / 2;
    int kmax = (tja + tjb) / 2;
    if (kmax > k_cut)
      kmax = k_cut;

    // hold "fraction" Fa*Fb/(Fa^2):
    std::vector<double> v_Fab(Fa.grid().num_points());
    for (std::size_t i = 0; i < irmax; i++) {
      // This is the approximate part! Divides by Fa
      if (std::abs(Fa.f(i)) < cut_off)
        continue;
      const auto fac_top = Fa.f(i) * Fb.f(i) + Fa.g(i) * Fb.g(i);
      const auto fac_bot = Fa.f(i) * Fa.f(i) + Fa.g(i) * Fa.g(i);
      v_Fab[i] = -x_tjbp1 * fac_top / fac_bot;
    } // r

    for (int k = kmin; k <= kmax; k++) {
      const auto parity = Angular::parity(la, lb, k);
      if (parity == 0)
        continue;
      const auto tjs = Angular::threej_2(tjb, tja, 2 * k, -1, 1, 0);
      if (tjs == 0)
        continue;
      const auto tjs2 = tjs * tjs;
      Coulomb::yk_ab(Fb, Fa, k, vabk);

      for (std::size_t i = 0; i < irmax; i++) {
        if (v_Fab[i] == 0)
          continue;
        vex[i] += tjs2 * vabk[i] * v_Fab[i];
      } // r
    }   // k
  }     // b

  return vex;
}

//==============================================================================
void HartreeFock::vex_Fa_core(const DiracSpinor &Fa, DiracSpinor &VxFa) const
// calculates V_ex Fa
// Fa can be any orbital (so long as coulomb integrals exist!)
// ..... i.e., must be core orbital
{
  VxFa.max_pt() = Fa.f().size(); // silly hack. Make sure VxFa = 0 after pinf
  VxFa *= 0.0;
  VxFa.max_pt() = Fa.max_pt(); // updated below

  if (excludeExchangeQ())
    return;

  const auto twoj_a = Fa.twoj();
  for (const auto &Fb : m_core) {
    VxFa.max_pt() = std::max(VxFa.max_pt(), Fb.max_pt());
    const auto tjb = Fb.twoj();
    const double x_tjbp1 = (Fa == Fb) ? (tjb + 1) : (tjb + 1) * Fb.occ_frac();
    const int kmin = std::abs(twoj_a - tjb) / 2;
    const int kmax = (twoj_a + tjb) / 2;
    // const auto &vabk = m_Yab.get_y_ab(Fb, Fa);
    for (int k = kmin; k <= kmax; k++) {
      const auto Labk = m_Yab.Ck().get_Lambdakab(k, Fa.kappa(), Fb.kappa());
      if (Labk == 0)
        continue;
      const auto vabk = m_Yab.get(k, Fb, Fa);
      if (vabk == nullptr)
        continue;
      for (auto i = 0u; i < Fb.max_pt(); i++) {
        const auto v = -x_tjbp1 * Labk * (*vabk)[i];
        VxFa.f(i) += v * Fb.f(i);
        VxFa.g(i) += v * Fb.g(i);
      } // r
    }   // k
  }     // b
}

// -----------------------------------------------------------------------------
DiracSpinor vexFa(const DiracSpinor &Fa, const std::vector<DiracSpinor> &core,
                  int k_cut)
// Free Function
// calculates V_ex Fa (returns new Dirac Spinor)
// Fa can be any orbital (Calculates coulomb integrals here!)
{
  DiracSpinor VxFa(Fa.n(), Fa.kappa(), Fa.grid_sptr());
  VxFa.max_pt() = Fa.max_pt(); // nb: updated below!
  // note: VxFa.max_pt() can be larger than psi.max_pt()!

  std::vector<double> vabk(Fa.grid().num_points());

  const auto tja = Fa.twoj();
  const auto la = Fa.l();
  for (const auto &Fb : core) {
    VxFa.max_pt() = std::max(Fb.max_pt(), VxFa.max_pt());
    const auto tjb = Fb.twoj();
    const auto lb = Fb.l();
    const double x_tjbp1 = (Fa == Fb) ? (tjb + 1) : (tjb + 1) * Fb.occ_frac();
    const int kmin = std::abs(tja - tjb) / 2;
    const int kmax = std::min((tja + tjb) / 2, k_cut);
    for (int k = kmin; k <= kmax; k++) {
      if (Angular::parity(la, lb, k) == 0)
        continue;
      const auto tjs = Angular::threej_2(tjb, tja, 2 * k, -1, 1, 0);
      if (tjs == 0)
        continue;
      Coulomb::yk_ab(Fb, Fa, k, vabk, Fb.max_pt());
      const auto factor = -x_tjbp1 * tjs * tjs;
      for (auto i = 0u; i < Fb.max_pt(); i++) {
        VxFa.f(i) += factor * vabk[i] * Fb.f(i);
        VxFa.g(i) += factor * vabk[i] * Fb.g(i);
      } // r
    }   // k
  }     // b

  return VxFa;
}

//==============================================================================
std::vector<double> HartreeFock::get_Hrad_el(int l) const {
  return m_vrad ? m_vrad->Vel(l) : std::vector<double>{};
}
std::vector<double> HartreeFock::get_Hrad_mag(int l) const {
  return m_vrad ? m_vrad->Hmag(l) : std::vector<double>{};
}
//==============================================================================
//==============================================================================
//==============================================================================

//==============================================================================
void HartreeFock::hf_orbital_green(
    DiracSpinor &Fa, double en0, const std::vector<double> &vl,
    const std::vector<double> &H_mag, const DiracSpinor &VnlFa0,
    const std::vector<DiracSpinor> &static_core, const std::vector<double> &dv0,
    const HF::Breit *const tVBr,
    const MBPT::CorrelationPotential *const Sigma) const
// Solves HF equation for given state, using non-local Green's method for
// inhomogeneous ODE (used for hartree_fock_core()).
// Solve Dirac Equation (Eigenvalue):
//  (H0 + Vl + Vx)Fa = 0
//  (H0 + Vl)Fa = -VxFa
// Vl is local (e.g., Vnuc + fVdir), Vx is non-local (e.g., (1-f)Vdir + Vex)
// where v0 = (1-f)Vdir  [f=1 for valence states!, so v0 may be empty]
// Vx also includes Breit, and Sigma
// Small energy adjustmenets (and wfs), solve:
// (Hl - e) dF = de * F -VxFa
// e -> e+de, F->F+dF
// Core is input so can call in a thread-safe way! (with a 'old_core' copy)
// Only used in dE from dF
{
  // should work for core, valence, w/wo (Br, Sigma, H_mag)

  // pull these outside? But make sure thread safe!
  DiracSpinor Gzero(Fa.n(), Fa.kappa(), Fa.grid_sptr());
  DiracSpinor Ginf(Fa.n(), Fa.kappa(), Fa.grid_sptr());
  DiracSpinor dFa(Fa.n(), Fa.kappa(), Fa.grid_sptr());
  const auto eps_target = 1.0e-18; // m_eps_HF;
  const auto k_max = 1;            // max k for Vex into dEa

  // const auto SigmaF = (*Sigma)(Fa);

  // nb: VnlFa0 includes Breit + Sigma etc. VBr used for energy correction

  const auto alpha = m_alpha;
  DiracODE::solve_inhomog(Fa, Gzero, Ginf, en0, vl, H_mag, alpha,
                          -1.0 * VnlFa0);

  // make small adjustments to energy to normalise Fa:
  DiracODE::Adams::GreenSolution(dFa, Ginf, Gzero, alpha, Fa);
  // should dFa = dEa * dFa, but makes it worse?
  // nb: after first it, becomes correct.
  double en = en0;
  double dEa = 0.5 * (Fa * Fa - 1.0) / (Fa * dFa);
  double eps = std::abs(dEa / en0);
  int tries = 0;
  for (; tries <= m_max_hf_its; ++tries) {
    if (eps < eps_target && tries > 1)
      break;
    {
      auto VnlF_tilde = ::HF::vexFa(dFa, static_core, k_max);
      if (tVBr)
        VnlF_tilde += tVBr->VbrFa(dFa, static_core);
      if (Sigma)
        VnlF_tilde += (*Sigma)(dFa);
      if (!dv0.empty())
        VnlF_tilde += dv0 * dFa;
      // const auto SigmaF_tilde = Sigma(dFa);
      DiracODE::Adams::GreenSolution(dFa, Ginf, Gzero, alpha,
                                     dEa * Fa - VnlF_tilde);
    }
    const auto delta_Norm = Fa * Fa - 1.0;
    const auto de0 = dEa;
    dEa *= 0.5 * (delta_Norm) / (Fa * dFa);
    eps = std::abs(dEa / en);
    en += dEa;
    dFa.max_pt() = Fa.max_pt();
    Fa -= (dEa / de0) * dFa;
  }
  Fa.en() = en;
  Fa.eps() = eps;
  Fa.its() = tries;
  Fa.normalise();
}

//==============================================================================
double HartreeFock::zion() const {
  const auto z = -1.0 * m_vnuc.back() * m_rgrid->r().back();
  return z - double(num_core_electrons());
}

//==============================================================================
double HartreeFock::enGuessCore(int n, int kappa) const
// Private
// Energy guess for core states. Not perfect, good enough
// num_el_below = total electrons BELOW
// num = num electrons in THIS shell
{
  const int l = AtomData::l_k(kappa);
  int num_el_below = 0;
  int num_el_this = 0;
  for (const auto &Fc : m_core) {
    if (n == Fc.n() && l == Fc.l())
      num_el_this += Fc.num_electrons();

    if (Fc.n() < n || (n == Fc.n() && Fc.l() < l))
      num_el_below += Fc.num_electrons();
  }

  const auto z = -1.0 * m_vnuc.back() * m_rgrid->r().back();

  // effective Z (for energy guess) -- not perfect!
  double Zeff = 1.0 + (z - num_el_below - 0.5 * num_el_this);
  if (Zeff < 1.0) {
    Zeff = 1.0;
  }

  double en_a = -0.5 * std::pow(Zeff / n, 2);
  if (n > 1) {
    en_a *= 0.5;
  }
  if (Zeff < 10) {
    if (l == 0)
      en_a *= 2.5;
    if (l == 1)
      en_a *= 3.5;
  }

  return en_a;
}

//==============================================================================
double HartreeFock::enGuessVal(int n, int ka) const
// Energy guess for valence states. Not perfect, good enough
{
  const int maxn = DiracSpinor::max_n(m_core);
  const int l = AtomData::l_k(ka);
  const int dn = n - maxn;
  double neff = 1.0 + dn;
  double x = 1;
  const auto z = -1.0 * m_vnuc.back() * m_rgrid->r().back();
  double Z_eff = z - num_core_electrons();
  if (Z_eff <= 0)
    Z_eff = 0.5;
  if (maxn < 4)
    x = 0.25;
  if (l == 1)
    neff += 0.5 * x;
  if (l == 2)
    neff += 2.0 * std::pow(x, 0.5);
  if (l >= 3)
    neff += 4.0 * x;
  return -0.5 * Z_eff * Z_eff / std::pow(neff, 2);
}

} // namespace HF
