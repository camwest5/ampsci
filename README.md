# ampsci

## Atomic Many-body Perturbation theory in the Screened Coulomb Interaction

[_ampsci_](https://ampsci.dev/)
is a c++ program for high-precision atomic structure calculations of single-valence systems.

It solves the correlated Dirac equation using the Hartree-Fock + correlation potential method (based on Dzuba-Flambaum-Sushkov method) to produce a set of atomic wavefunctions and energies.
The method is fully relativistic, includes electron correlations, all-orders screening and hole-particle interaction, finite-nuclear size, Breit interaction, radiative QED effects, RPA for matrix elements, and structure radiation/renormalisation.
QED is included via the Flambaum-Ginges radiative potential method.
Can solve for continuum states with high energy, and calculate ionisation cross sections with large energy/momentum transfer.

Designed to be fast, accurate, and easy to use.
The "modules" system (see [doc/modules.md](doc/modules.md)) makes it simple to add your own routines to use the atomic wavefunctions to calculate whatever properties you may be interested in.

* The code is on GitHub: [github.com/benroberts999/ampsci](https://github.com/benroberts999/ampsci)
* See [ampsci.dev/](https://ampsci.dev/) for full documentation
* A full description of the physics methods and approximations, including references,
is given in the physics documentation: [ampsci.pdf][man-url].
* **Important:** this is a _pre-release_ version of the code: not fully tested or documented, and should not be used for publishable calculations (without consultation)

[![github][github-badge]](https://github.com/benroberts999/ampsci)
[![doxygen][doxygen-badge]][docs-url]
[![manual][manual-badge]][man-url]

[![tests][tests-badge]][tests-url]
[![build][build-badge]][build-url]
[![macOS][macOS-badge]][macOS-url]
[![cov][cov-badge]][cov-url]

## Contents

* [Compilation](#compilation)
* [Basic Usage](#ampsci-basic-usage)
* [Documentation](#documentation)

--------------------------------------------------------------------------------

## Compilation <a name="compilation"></a>

* Easiest method is to compile using provided Makefile:
* Copy "doc/examples/Makefile" from doc/ directory to the working directory
  * `$ cp ./doc/examples/Makefile ./`
* All programs compiled using the Makefile (run `$ make`)
* The file _Makefile_ has some basic compilation options. It's currently set up to work on most linux systems; you may need to change a few options for others (see [doc/compilation.md](doc/compilation.md))
* Tested with g++ and clang++ on linux and mac

### Dependencies / Requirements

* c++ compiler that supports c++17 [clang version 6 or newer, gcc version 7 or newer]
* LAPACK and BLAS libraries [netlib.org/lapack/](http://www.netlib.org/lapack/)
* GSL (GNU scientific libraries) [gnu.org/software/gsl/](https://www.gnu.org/software/gsl/) [version 2.0 or newer]
* [optional] GNU Make ([gnu.org/software/make/](https://www.gnu.org/software/make/)) - used to compile code
* [optional] OpenMP ([openmp.org/](https://www.openmp.org/)) - used for parallisation
* [optional] git ([git-scm.com/](https://git-scm.com/)) for version tracking and to keep up-to-date with latest version

### Quick-start (linux)

This is for ubuntu/linux - for other systems, see [doc/compilation.md](doc/compilation.md)

* Get the code from [GitHub](https://github.com/benroberts999/ampsci), using git:
  * `$ sudo apt install git`
  * `$ git clone git@github.com:benroberts999/ampsci.git`
  * or `$ git clone https://github.com/benroberts999/ampsci.git`
* Or, direct download (without using git):
  * <https://github.com/benroberts999/ampsci/archive/refs/heads/main.zip>
* Install dependencies
  * `$ sudo apt install g++ liblapack-dev libblas-dev libgsl-dev make libomp-dev`
* Prepare the Makefile (already setup for ubuntu, you may need minor adjustments, see [doc/compilation.md](doc/compilation.md))
  * `$ cp ./doc/examples/Makefile ./`
* Compile ampsci using all default options:  
  * `$ make`
* Run the first example program
  * `$ cp ./doc/examples/ampsci.in ./`
  * `$ ./ampsci ampsci.in`

For full compilation guides including for mac/windows, see [doc/compilation.md](doc/compilation.md)

--------------------------------------------------------------------------------

## Basic Usage <a name="ampsci-basic-usage"></a>

The program is run with input options from the command line.

### Main method: input options from a text file

* `$ ./ampsci filename`
  * Runs ampsci with input option specified in file "filename"
  * See [doc/ampsci_input.md](doc/ampsci_input.md) for full description of input format,
and a detailed list of input options + descriptions.
  * run `$ampsci -h` to get breif instructions for input options
  * Several example input files are given in: _doc/examples/_, along with their expected output; use these to test if everything is working.

The Output is printed to screen. It's recommended to forward this to a text file.
The input options and the ampsci version details are also printed, so that the
program output contains all required info to exactly reproduce it. e.g.,

* `$ ./ampsci input |tee -a outout`
  * Runs ampsci using input options in file "input".
  * Output will both be written to screen, and appended to
    file "output".

### quick method (simple calculations)

For very simple (Hartree-Fock only) calculations, you can run ampsci directly from the command line:

* `$ ./ampsci <At> <Core> <Valence>`
  * `$ ./ampsci Cs`
    * Runs ampsci for Cs using Hartree Fock (V^N) approximation
  * `$ ./ampsci Cs [Xe] 6sd5d`
    * Runs ampsci for Cs using Hartree Fock with Xe-like core and valence
      states up to n=6 for s,p-states and n=5 for d-states
  * `$ ./ampsci Cs`
    * Runs ampsci for Cs using Hartree Fock (V^N) approximation

### Other command-line options

* `$ ./ampsci -v`
  * Prints version info (same as --version)
* `$ ./ampsci -h`
  * Print help info, including input options (same as --help, -?)
* `$ ./ampsci -m  <ModuleName>`
  * Prints list of available Modules (same as --modules)
  * ModuleName is optional. If given, will list avaiable options for that Module
  * e.g., `./ampsci -m polarisability` will print all available options for the polarisability module
* `$ ./ampsci -o <OperatorName>`
  * Prints list of available operators (same as --operators)
  * OperatorName is optional. If given, will list avaiable options for Operator
  * e.g., `./ampsci -o E1` will print all available options for E1 operator
* `$ ./ampsci -a <BlockName>`
  * Prints list of available top-level ampsci options (same as --ampsci)
  * BlockName is optional; if given will print options for given ampsci Block
  * e.g., `./ampsci -a Basis` will print all available 'Basis' options
* `$ ./ampsci -p <At> <Isotope>`
  * Prints periodic table with electronic+nuclear info (same as --periodicTable)
  * At and Isotope are optional. If given, will print info for given isotope
  * e.g., ./ampsci -p Cs, ./ampsci -p Cs 133, ./ampsci -p Cs all
* `$ ./ampsci -c`
  * Prints some handy physical constants (same as --constants)

--------------------------------------------------------------------------------

## Documentation <a name="documentation"></a>

* Full documentation available online: [ampsci.dev/](https://ampsci.dev/)

For most applications, the code is "self documenting" (meaning input options etc) can be extracted from the code.
Run `$ ./ampsci -h` from the command line to see this (see [Basic Usage](#ampsci-basic-usage) for other command-line options)

 1. Input options -- how to run the code
    * [doc/ampsci_input.md](doc/ampsci_input.md) -- detailed info on all input options
    * See also: _doc/examples/ampsci.in_ -- an example/template input file
    * In _doc/examples/_ there are several example input files, with the expected output; use these to test if everything is working

 2. Physics documentation: _ampsci.pdf_ -- Description of physics/methods used in the code
    * Includes many references to the works where the methods implemented here were developed.
    * Available online: [ampsci.dev/ampsci.pdf](https://ampsci.dev/ampsci.pdf)
    * Latex file provided in doc/tex/ampsci.tex
    * If you have latex installed, you can use Makefile to generate the pdf
      * Run `$ make docs` -- this will create new pdf file: 'doc/ampsci.pdf'

 3. Modules
    * The modules system allows the easy calculation of any atomic properties after the wavefunction has been calculated. See [doc/modules.md](doc/modules.md) for description
    * The code is designed so that you can easily create your own modules. See [doc/writing_modules.md](doc/writing_modules.md) for details

 4. Code documentation -- details on classes/functions in the code
    * Available online: [ampsci.dev/](https://ampsci.dev/)
    * This should only be required if you plan to edit the code or add new modules

--------------------------------------------------------------------------------

[tests-badge]: https://github.com/benroberts999/ampsci/actions/workflows/tests.yml/badge.svg
[tests-url]: https://github.com/benroberts999/ampsci/actions/workflows/tests.yml
[build-badge]: https://github.com/benroberts999/ampsci/actions/workflows/build.yml/badge.svg
[build-url]: https://github.com/benroberts999/ampsci/actions/workflows/build.yml
[macOS-badge]: https://github.com/benroberts999/ampsci/actions/workflows/macOS.yml/badge.svg
[macOS-url]: https://github.com/benroberts999/ampsci/actions/workflows/macOS.yml
[doxygen-badge]: https://img.shields.io/badge/documentation-ampsci.dev/-blue
[docs-url]: https://ampsci.dev/
[manual-badge]: https://img.shields.io/badge/documentation-physics%20(pdf)-blue
[man-url]: https://ampsci.dev/ampsci.pdf
[cov-badge]: https://codecov.io/gh/benroberts999/ampsci/branch/main/graph/badge.svg?token=3M5MH5QXLL
[cov-url]: https://codecov.io/gh/benroberts999/ampsci
[c++-badge]: https://img.shields.io/badge/c++-17-blue
[github-badge]: https://img.shields.io/badge/Code%20available:-GitHub-blueviolet?style=flat&logo=github&logoColor=white

[tests-badge-v2]: tests-badge.svg
[build-badge-v2]: build-badge.svg
[macOS-badge-v2]: macOS-badge.svg
[cov-badge-v2]: cov-badge.svg
