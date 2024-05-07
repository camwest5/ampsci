#include "Modules/dcp.hpp"
#include "Angular/Angular.hpp"
#include "Coulomb/Coulomb.hpp"
#include "DiracOperator/DiracOperator.hpp" //For E1 operator
#include "IO/InputBlock.hpp"
#include "Wavefunction/Wavefunction.hpp"

namespace Module {

void dcp(const IO::InputBlock &input, const Wavefunction &wf) {

  input.check({{"h1", "Operator 1"}, //
               {"h2", "Operator 2"}});

  // If we are just requesting 'help', don't run module:
  if (input.has_option("help")) {
    return;
  }

  const auto oper1 = input.get<std::string>("h1", "E1");
  const auto oper2 = input.get<std::string>("h2", "pnc");

  const auto s = DiracOperator::generate(oper1, {}, wf);
  const auto t = DiracOperator::generate(oper2, {}, wf);

  // Fill Yk table (used to fill Qk table)
  const Coulomb::YkTable yk(wf.basis());

  // take from basis for now?
  const auto pv = DiracSpinor::find(6, -1, wf.basis());
  const auto pw = DiracSpinor::find(7, -1, wf.basis());
  if (!pv || !pw)
    return;
  const auto &v = *pv;
  const auto &w = *pw;

  const auto &[holes, excited] =
      DiracSpinor::split_by_energy(wf.basis(), wf.FermiLevel());

  auto alpha = 0.0;
  for (const auto &n : wf.basis()) {

    const auto svn = s->reducedME(v, n) * s->rme3js(v.twoj(), n.twoj(), 1);
    const auto tnw = t->reducedME(n, w) * t->rme3js(n.twoj(), w.twoj(), 1);
    const auto tvn = t->reducedME(v, n) * t->rme3js(v.twoj(), n.twoj(), 1);
    const auto snw = s->reducedME(n, w) * s->rme3js(n.twoj(), w.twoj(), 1);

    // careful!
    if (svn == 0.0)
      continue;

    alpha += svn * tnw / (w.en() - n.en()) + tvn * snw / (v.en() - n.en());
  }
  std::cout << alpha << "\n\n\n";

  auto d1 = 0.0;
  auto d2 = 0.0;
  auto d3 = 0.0;
  auto d4 = 0.0;
  for (const auto &a : holes) {
    for (const auto &n : excited) {

      const auto e_wavn = w.en() + a.en() - v.en() - n.en();

      const auto f_n = 1.0 / std::sqrt(18.0 * n.twojp1());
      const auto f_a = 1.0 / std::sqrt(18.0 * a.twojp1());

      // first term:
      for (const auto &m : excited) {

        if (n.twoj() != m.twoj())
          continue;

        const auto del_am = a.en() - m.en();
        const auto San = s->reducedME(a, n);
        const auto Tma = t->reducedME(m, a);
        const auto Tan = t->reducedME(a, n);
        const auto Sma = s->reducedME(m, a);

        const auto sign =
            Angular::neg1pow_2(n.twoj() + a.twoj() + v.twoj() + n.twoj());

        const auto Wvnwm = yk.W(0, v, n, w, m);

        d1 += sign * f_n * Wvnwm * San * Tma / del_am / e_wavn;
        d2 += sign * f_n * Wvnwm * Tan * Sma / del_am / e_wavn;
      }

      // second term:
      for (const auto &b : holes) {

        if (a.twoj() != b.twoj())
          continue;

        const auto del_bn = b.en() - n.en();
        const auto San = s->reducedME(a, n);
        const auto Tnb = t->reducedME(n, b);
        const auto Tan = t->reducedME(a, n);
        const auto Snb = s->reducedME(n, b);

        const auto sign =
            Angular::neg1pow_2(n.twoj() + a.twoj() + v.twoj() + a.twoj());

        const auto Wvbwa = yk.W(0, v, b, w, a);

        d3 += -sign * f_a * Wvbwa * San * Tnb / del_bn / e_wavn;
        d4 += -sign * f_a * Wvbwa * Tan * Snb / del_bn / e_wavn;
      }
    }
    std::cout << a << " " << d1 + d2 << " " << d3 + d4 << " " << d1 / d3
              << "\n";
  }

  std::cout << d1 + d2 + d3 + d4 << "\n";
}

} // namespace Module
