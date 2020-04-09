#include "Modules/Module_runModules.hpp"
#include "DMionisation/Module_atomicKernal.hpp"
#include "IO/UserInput.hpp"
#include "Modules/Module_fitParametric.hpp"
#include "Modules/Module_matrixElements.hpp"
#include "Modules/Module_pnc.hpp"
#include "Modules/Module_tests.hpp"
#include "Wavefunction/DiracSpinor.hpp"
#include "Wavefunction/Wavefunction.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

namespace Module {

static const std::vector<
    std::pair<std::string, void (*)(const IO::UserInputBlock &input,
                                    const Wavefunction &wf)>>
    module_list = {{"Tests", &Module_tests},
                   {"WriteOrbitals", &writeOrbitals},
                   {"AtomicKernal", &atomicKernal},
                   {"FitParametric", &fitParametric},
                   {"BohrWeisskopf", &calculateBohrWeisskopf},
                   {"pnc", &calculatePNC},
                   {"polarisability", &polarisability},
                   {"lifetimes", &calculateLifetimes}};

//******************************************************************************
void runModules(const IO::UserInput &input, const Wavefunction &wf) {
  auto modules = input.module_list();
  for (const auto &module : modules) {
    runModule(module, wf);
  }
}

//******************************************************************************
void runModule(const IO::UserInputBlock &module_input,
               const Wavefunction &wf) //
{
  const auto &in_name = module_input.name();

  // Run "MatrixElements modules" if requested
  if (in_name.substr(0, 14) == "MatrixElements") {
    return matrixElements(module_input, wf);
  }

  // Otherwise, loop through all available modules, run correct one
  for (const auto &[mod_name, mod_func] : module_list) {
    if (in_name == "Module::" + mod_name)
      return mod_func(module_input, wf);
  }

  std::cout << "\nFail 50 in runModule: no module named: " << in_name << "\n";
  std::cout << "Available modules:\n";
  for (const auto &module : module_list) {
    std::cout << "  " << module.first << "\n";
  }
  std::cout << "\n";
}

//******************************************************************************
void writeOrbitals(const IO::UserInputBlock &input, const Wavefunction &wf) {
  const std::string ThisModule = "Module::WriteOrbitals";
  input.checkBlock({"label"});

  std::cout << "\n Running: " << ThisModule << "\n";
  auto label = input.get<std::string>("label", "");
  std::string oname = wf.atomicSymbol() + "-orbitals";
  if (label != "")
    oname += "_" + label;

  oname += ".txt";
  std::ofstream of(oname);
  of << "r ";
  for (auto &psi : wf.core)
    of << "\"" << psi.symbol(true) << "\" ";
  for (auto &psi : wf.valence)
    of << "\"" << psi.symbol(true) << "\" ";
  of << "\n";
  of << "# f block\n";
  for (std::size_t i = 0; i < wf.rgrid.num_points; i++) {
    of << wf.rgrid.r[i] << " ";
    for (auto &psi : wf.core)
      of << psi.f[i] << " ";
    for (auto &psi : wf.valence)
      of << psi.f[i] << " ";
    of << "\n";
  }
  of << "\n# g block\n";
  for (std::size_t i = 0; i < wf.rgrid.num_points; i++) {
    of << wf.rgrid.r[i] << " ";
    for (auto &psi : wf.core)
      of << psi.g[i] << " ";
    for (auto &psi : wf.valence)
      of << psi.g[i] << " ";
    of << "\n";
  }
  of << "\n# density block\n";
  auto rho = wf.coreDensity();
  for (std::size_t i = 0; i < wf.rgrid.num_points; i++) {
    of << wf.rgrid.r[i] << " " << rho[i] << "\n";
  }
  of.close();
  std::cout << "Orbitals written to file: " << oname << "\n";
}

} // namespace Module
