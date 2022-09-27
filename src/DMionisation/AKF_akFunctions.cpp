#include "DMionisation/AKF_akFunctions.hpp"
#include "Angular/Wigner369j.hpp"
#include "DiracOperator/DiracOperator.hpp"
#include "IO/FRW_fileReadWrite.hpp"
#include "LinAlg/Matrix.hpp"
#include "Maths/Grid.hpp"
#include "Maths/NumCalc_quadIntegrate.hpp"
#include "Maths/SphericalBessel.hpp"
#include "Physics/AtomData.hpp"
#include "Physics/PhysConst_constants.hpp"
#include "Wavefunction/ContinuumOrbitals.hpp"
#include "Wavefunction/Wavefunction.hpp"
#include <fstream>
#include <iostream>

namespace AKF {

//==============================================================================
std::vector<double>
calculateK_nk(const Wavefunction &wf, const DiracSpinor &Fa, int max_L,
              double dE,
              // const std::vector<std::vector<std::vector<double>>> &jLqr_f,
              const DiracOperator::jL &jl, bool subtract_1, bool force_rescale,
              bool subtract_self, bool force_orthog, bool zeff_cont)
// Calculates the atomic factor for a given core state (is) and energy.
// Note: dE = I + ec is depositied energy, not cntm energy
// Zeff is '-1' by default. If Zeff > 0, will solve w/ Zeff model
// Zeff no longer works at main() level.
{
  std::vector<double> AK_nk_q;
  const auto qsteps = jl.q_grid().num_points();
  AK_nk_q.resize(qsteps);

  // Convert energy deposition to contimuum state energy:
  double ec = dE + Fa.en();
  if (ec <= 0.0)
    return AK_nk_q;

  const int l = Fa.l();
  const int lc_max = l + max_L;
  const int lc_min = std::max(l - max_L, 0);
  const double x_ocf = Fa.occ_frac(); // occupancy fraction. Usually 1

  ContinuumOrbitals cntm(wf); // create cntm object [survives locally only]
  if (zeff_cont) {
    // Same Zeff as used by DarkARC (eqn B35 of arxiv:1912.08204):
    // Zeff = sqrt{I_{njl} eV / 13.6 eV} * n
    // au: Zeff = sqrt{2 * I_{njl}} * n
    const double Zeff = std::sqrt(-2.0 * Fa.en()) * Fa.n();
    cntm.solveContinuumZeff(ec, lc_min, lc_max, Zeff, &Fa, force_orthog);
  } else {
    cntm.solveContinuumHF(ec, lc_min, lc_max, &Fa, force_rescale, subtract_self,
                          force_orthog);
  }

  if (subtract_1 && (jl.name() != "jL" && jl.name() != "g0jL")) {
    std::cout << "\nWARNING: subtract 1 option currently only works for vector "
                 "and scalar operator (due to factoring out i)\n";
    subtract_1 = false;
  }

  // Definition of matrix element:
  // matrix element defined such that:
  // K(E,q) = (2L+1) * |me|^2
  // me = <a||jL||e>
  // Now that we have 'me' directly (rather than me^2), we can -1 easily:
  // <a| jL - 1 |e> = <a| jL |e> - <a|e>
  // note: only works for vector/scalar, since for pseudo-cases,
  // we factored out factor of i from me

  // Generate AK for each L, lc, and q
  // L and lc are summed, not stored individually
  for (std::size_t L = 0; L <= std::size_t(max_L); L++) {
    for (const auto &Fe : cntm.orbitals) {
      if (jl.is_zero(Fe, Fa, L))
        continue;
#pragma omp parallel for
      for (std::size_t iq = 0; iq < qsteps; iq++) {
        const auto q = jl.q_grid().r(iq);
        auto me = jl.rme(Fa, Fe, L, q);
        if (subtract_1 && (L == 0 && Fe.kappa() == Fa.kappa())) {
          me -= Fe * Fa;
        }
        AK_nk_q[iq] += double(2 * L + 1) * me * me * x_ocf;
      }
    }
  }
  return AK_nk_q;
}

//==============================================================================
double Heaviside(double x)
// Heaviside step function: Theta(x) = {(1 for x>0) and (0 for x<=0)}
{
  return (x > 0.0) ? 1.0 : 0.0;
}

//==============================================================================
std::vector<double> stepK_nk(const DiracSpinor &Fa, double dE,
                             const std::vector<double> &AFBE_table)
/*
Approximates atomic factor using table generated by AFBindingEnergy
*/
{
  using namespace qip::overloads;
  return Heaviside(dE + Fa.en()) * AFBE_table;

  // // q range needs to be the same for the table and AK
  // const auto qsteps = AFBE_table.size();

  // std::vector<double> AK_nk_step;
  // AK_nk_step.resize(qsteps);

  // for (std::size_t iq = 0; iq < qsteps; iq++) {
  //   AK_nk_step[iq] += Heaviside(dE + Fa.en()) * AFBE_table[iq];
  // }
  // return AK_nk_step;
}

//==============================================================================
void write_Knk_plaintext(
    const std::string &fname,
    const std::vector<std::vector<std::vector<double>>> &AK,
    const std::vector<std::string> &nklst, const Grid &qgrid, const Grid &Egrid)
// /*
// Writes the K factor to a text-file, in GNU-plot readable format
// */
{
  const auto desteps = AK.size();       // dE
  const auto num_states = AK[0].size(); // nk
  const auto qsteps = AK[0][0].size();  // q
  assert(nklst.size() == num_states);
  assert(qgrid.num_points() == qsteps);
  assert(Egrid.num_points() == desteps);

  const double qMeV = (1.e6 / (PhysConst::Hartree_eV * PhysConst::c));
  const double keV = (1.e3 / PhysConst::Hartree_eV);

  std::ofstream ofile(fname + "_nk.txt");

  ofile << "dE(keV) q(MeV) ";
  for (const auto &nk : nklst) {
    ofile << nk << " ";
  }
  ofile << "Sum\n\n";
  for (auto idE = 0ul; idE < desteps; idE++) {
    for (auto iq = 0ul; iq < qsteps; iq++) {
      const auto q = qgrid.r(iq);
      const auto dE = Egrid.r(idE);
      ofile << dE / keV << " " << q / qMeV << " ";
      double sum = 0.0;
      for (auto j = 0ul; j < num_states; j++) {
        sum += AK[idE][j][iq];
        ofile << AK[idE][j][iq] << " ";
      }
      ofile << sum << "\n";
    }
    if (qsteps > 1)
      ofile << "\n";
  }
}

//==============================================================================
void write_Ktot_plaintext(const std::string &fname,
                          const LinAlg::Matrix<double> &Keq, const Grid &Egrid,
                          const Grid &qgrid) {
  std::ofstream ofile(fname + "_tot.txt");

  const double keV = (1.e3 / PhysConst::Hartree_eV);
  const double qMeV = (1.e6 / (PhysConst::Hartree_eV * PhysConst::c));

  ofile << "# Total ionisation factor K(E,q)\n";
  ofile << "# In form: E/keV q/MeV K\n";
  ofile << "# Each new E is a new block\n";
  ofile << "# E grid in keV: " << Egrid.r0() / keV << " - "
        << Egrid.rmax() / keV << " in " << Egrid.num_points()
        << " logarithmic steps.\n";
  ofile << "# q grid in MeV: " << qgrid.r0() / qMeV << " - "
        << qgrid.rmax() / qMeV << " in " << qgrid.num_points()
        << " logarithmic steps.\n";

  assert(Keq.rows() == Egrid.num_points());
  assert(Keq.cols() == qgrid.num_points());

  for (auto ie = 0ul; ie < Egrid.num_points(); ie++) {
    for (auto iq = 0ul; iq < qgrid.num_points(); iq++) {
      const auto E_keV = Egrid.r(ie) / keV;
      const auto q_MeV = qgrid.r(iq) / qMeV;
      ofile << E_keV << " " << q_MeV << " " << Keq(ie, iq) << "\n";
    }
    ofile << '\n';
  }
}

//==============================================================================
void writeToTextFile_AFBE(
    const std::string &fname,
    const std::vector<std::vector<std::vector<double>>> &AK,
    const std::vector<std::string> &nklst, const Grid &qgrid,
    const std::vector<double> deion)
// /*
// Writes the K factor to a text-file, in GNU-plot readable format
// */
{

  const auto num_states = AK[0].size(); // nk
  const auto qsteps = AK[0][0].size();  // q
  assert(qgrid.num_points() == qsteps);
  assert(deion.size() == num_states);

  const double qMeV = (1.0e6 / (PhysConst::Hartree_eV * PhysConst::c));
  const double keV = (1.0e3 / PhysConst::Hartree_eV);

  std::ofstream ofile(fname + ".txt");
  ofile << "q(MeV) ";
  for (const auto &nk : nklst) {
    ofile << "dE(keV) " << nk << " ";
  }
  ofile << "Sum\n\n";
  for (auto iq = 0ul; iq < qsteps; iq++) {
    const auto q = qgrid.r(iq);
    ofile << q / qMeV << " ";
    double sum = 0.0;
    for (auto j = 0ul; j < num_states; j++) {
      sum += AK[0][j][iq];
      ofile << deion[j] / keV << " " << AK[0][j][iq] << " ";
    }
    ofile << sum << "\n";
  }
  if (qsteps > 1)
    ofile << "\n";
}

//==============================================================================
int akReadWrite(const std::string &fname, bool write,
                std::vector<std::vector<std::vector<double>>> &AK,
                std::vector<std::string> &nklst, double &qmin, double &qmax,
                double &dEmin, double &dEmax)
// /*
// Writes K function (+ all required size etc.) values to a binary file.
// The binary file is read by other programs (e.g., dmeXSection)
// Uses FileIO_fileReadWrite
// XXX NOTE: Re-creates grids! Could use Grid class!
// XXX This mean we MUST use exponential Grid! Fix this! XXX
// */
{
  IO::FRW::RoW row = write ? IO::FRW::write : IO::FRW::read;

  std::fstream iof;
  IO::FRW::open_binary(iof, fname + ".bin", row);

  if (iof.fail()) {
    std::cout << "Can't open " << fname << ".bin\n";
    return 1;
  }

  if (write) {
    auto nde = AK.size();      // dE
    auto ns = AK[0].size();    // nk
    auto nq = AK[0][0].size(); // q
    IO::FRW::binary_rw(iof, nde, row);
    IO::FRW::binary_rw(iof, ns, row);
    IO::FRW::binary_rw(iof, nq, row);
  } else {
    std::size_t nq, ns, nde;
    IO::FRW::binary_rw(iof, nde, row);
    IO::FRW::binary_rw(iof, ns, row);
    IO::FRW::binary_rw(iof, nq, row);
    std::cout << "nde = " << nde << ", ns = " << ns << ", nq = " << nq
              << std::endl;
    AK.resize(nde,
              std::vector<std::vector<double>>(ns, std::vector<double>(nq)));
    nklst.resize(ns);
  }
  IO::FRW::binary_rw(iof, qmin, row);
  IO::FRW::binary_rw(iof, qmax, row);
  IO::FRW::binary_rw(iof, dEmin, row);
  IO::FRW::binary_rw(iof, dEmax, row);
  for (std::size_t ie = 0; ie < AK.size(); ie++) {
    for (std::size_t in = 0; in < AK[0].size(); in++) {
      if (ie == 0)
        IO::FRW::binary_str_rw(iof, nklst[in], row);
      for (std::size_t iq = 0; iq < AK[0][0].size(); iq++) {
        IO::FRW::binary_rw(iof, AK[ie][in][iq], row);
      }
    }
  }

  return 0;
}

//==============================================================================
int akReadWrite_AFBE(const std::string &fname, bool write,
                     std::vector<std::vector<std::vector<double>>> &AK,
                     std::vector<std::string> &nklst, double &qmin,
                     double &qmax, std::vector<double> &deion)
// /*
// Writes K function (+ all required size etc.) values to a binary file.
// The binary file is read by other programs (e.g., dmeXSection)
// Uses FileIO_fileReadWrite
// XXX NOTE: Re-creates grids! Could use Grid class!
// XXX This mean we MUST use exponential Grid! Fix this! XXX
// */
{
  IO::FRW::RoW row = write ? IO::FRW::write : IO::FRW::read;

  std::fstream iof;
  IO::FRW::open_binary(iof, fname + ".bin", row);

  if (iof.fail()) {
    std::cout << "Can't open " << fname << ".bin\n";
    return 1;
  }

  if (write) {
    auto ns = AK[0].size();    // nk
    auto nq = AK[0][0].size(); // q
    IO::FRW::binary_rw(iof, ns, row);
    IO::FRW::binary_rw(iof, nq, row);
  } else {
    std::size_t nq, ns;
    IO::FRW::binary_rw(iof, ns, row);
    IO::FRW::binary_rw(iof, nq, row);
    AK.resize(1, std::vector<std::vector<double>>(ns, std::vector<double>(nq)));
    nklst.resize(ns);
  }
  IO::FRW::binary_rw(iof, qmin, row);
  IO::FRW::binary_rw(iof, qmax, row);
  for (std::size_t in = 0; in < AK[0].size(); in++) {
    IO::FRW::binary_str_rw(iof, nklst[in], row);
    IO::FRW::binary_rw(iof, deion[in], row);
    for (std::size_t iq = 0; iq < AK[0][0].size(); iq++) {
      IO::FRW::binary_rw(iof, AK[0][in][iq], row);
    }
  }

  return 0;
}

} // namespace AKF
