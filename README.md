# A Python wrapper simulation toolkit for running LAMMPS simulations

This is a Python wrapper toolkit for molecular statics/dynamics simulations using open-source MD simulator code LAMMPS. 

Some selected features:

- This toolkit runs LAMMPS from Python (https://docs.lammps.org/Python_run.html); therefore LAMMPS has to be compiled as a shared library. This enables user to change/extract atom properties like position and type in LAMMPS directly through Python using functions like scatter_atoms/gather_atoms respectively defined in the LAMMPS Python API.
- A LAMMPS input file is also generated from the LAMMPS commands that are executed through Python. 
