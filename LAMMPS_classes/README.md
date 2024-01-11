# Main simulation engine for LAMMPS calculations

Here we have the principle classes responsible for defining an atomistic system, defining potentials and then performing either statics or dynamics calculations 

- class *System*: Parent class for defining units, atom_style, simulation cell geometry and boundary conditions and defining lattice.
- class *LoadSystem*: Inherits from *System* and reads atomistic data from a LAMMPS data or dump or restart file and in turn recreate/ restarts an existing LAMMPS calculation.
- class *LAMMPS_toolbox*: It is a multi-tasking class which performs a range of operations: filling a simulation cell atoms, writing data/dump/restart files, performing miscellaneous simulation cell operations, defining interatomic potentials and extract/maniputating atom properties in LAMMPS from Python.
- class *Simulations*: Parent class for running atomistic simulations. Class *MolecularStatics* and *MolecularDynamics* inherits from this class.
- class *WriteInputScript*:  This class is responsible to write a LAMMPS input script from the LAMMPS commands which are executed in Python. It should be noted that this LAMMPS calculations are NOT performed by first creating this script and then running it in Python using subprocess. Rather the LAMMPS commands are directly ran with Python and this LAMMPS input script generation is a complimentary feature of the toolkit. 


