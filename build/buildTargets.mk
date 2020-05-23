# All makefile targets

################################################################################
#Allow exectuables to be placed in another directory:
ALLEXES = $(addprefix $(XD)/, \
 diracSCAS wigner dmeXSection periodicTable \
)

DEFAULTEXES = $(addprefix $(XD)/, \
 diracSCAS wigner periodicTable \
)

#Default make rule:
all: checkObj checkXdir $(DEFAULTEXES)

################################################################################

# Dependencies for each object target, broken by sub-directories:
include $(SD)/Angular/Angular.mk
include $(SD)/Coulomb/Coulomb.mk
include $(SD)/DiracODE/DiracODE.mk
include $(SD)/DiracOperator/DiracOperator.mk
include $(SD)/DMionisation/DMionisation.mk
include $(SD)/HF/HF.mk
include $(SD)/IO/IO.mk
include $(SD)/Maths/Maths.mk
include $(SD)/MBPT/MBPT.mk
include $(SD)/Modules/Modules.mk
include $(SD)/Physics/Physics.mk
include $(SD)/Wavefunction/Wavefunction.mk
# Dependencies for each 'main' object target
include $(SD)/main.mk

################################################################################
# Just to save typing: Many programs depend on these combos:

BASE = $(addprefix $(BD)/, \
 Adams_bound.o Wavefunction.o DiracSpinor.o AtomData.o Nuclear.o Grid.o \
 DiracOperator.o NuclearData.o Angular_tables.o LinAlg_MatrixVector.o BSplineBasis.o \
)

HF = $(addprefix $(BD)/, \
 HartreeFock.o YkTable.o Coulomb.o Parametric_potentials.o \
 Adams_Greens.o MixedStates.o ExternalField.o RadiativePotential.o \
 CorrelationPotential.o \
)

CNTM = $(addprefix $(BD)/, \
 Adams_continuum.o ContinuumOrbitals.o \
)

MODS = $(MODULELIST)

# $(addprefix $(BD)/, \
#  Module_runModules.o Module_atomicKernal.o AKF_akFunctions.o \
#  Module_matrixElements.o Module_fitParametric.o Module_pnc.o Module_tests.o \
# )

################################################################################
# Link + build all final programs

$(XD)/diracSCAS: $(BASE) $(HF) $(CNTM) $(BD)/diracSCAS.o \
$(BD)/UserInput.o $(MODS)
	$(LINK)

$(XD)/dmeXSection: $(BASE) $(CNTM) $(HF) $(BD)/dmeXSection.o \
$(BD)/AKF_akFunctions.o $(BD)/StandardHaloModel.o
	$(LINK)

$(XD)/wigner: $(BD)/wigner.o
	$(LINK)

$(XD)/periodicTable: $(BD)/periodicTable.o $(BD)/AtomData.o \
$(BD)/NuclearData.o
	$(LINK)

################################################################################
################################################################################

checkObj:
	@if [ ! -d $(BD) ]; then \
	  echo '\n ERROR: Directory: '$(BD)' doesnt exist - please create it!\n'; \
	  false; \
	fi

checkXdir:
	@if [ ! -d $(XD) ]; then \
		echo '\n ERROR: Directory: '$(XD)' doesnt exist - please create it!\n'; \
		false; \
	fi

.PHONY: clean do_the_chicken_dance checkObj checkXdir
clean:
	rm -f $(ALLEXES) $(BD)/*.o
do_the_chicken_dance:
	@echo 'Why would I do that?'
