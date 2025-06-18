#pragma once

// Forward declare classes:
class Wavefunction;
namespace IO {
class InputBlock;
}

namespace Module {

void fieldShift(const IO::InputBlock &input, const Wavefunction &wf);
void isotopeShift(const IO::InputBlock &input, const Wavefunction &wf);

//! Calculates normal mass shift
void massShift(const IO::InputBlock &input, const Wavefunction &wf);

void fieldShift_direct(const IO::InputBlock &input, const Wavefunction &wf);

} // namespace Module