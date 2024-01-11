# A Python wrapper simulation toolkit for running LAMMPS simulations

This is a Python wrapper toolkit for molecular statics/dynamics simulations using open-source MD simulator code LAMMPS. 

Some selected features:

- This toolkit runs LAMMPS from Python (https://docs.lammps.org/Python_run.html); therefore LAMMPS has to be compiled as a shared library. This enables user to change/extract atom properties like position and type in LAMMPS directly through Python using functions like scatter_atoms()/gather_atoms() respectively defined in the LAMMPS Python API.
- A LAMMPS input script is also generated from the LAMMPS commands that are executed through Python. This input script can then be ran independently to extract the same results provided there has been no direct manupulation of LAMMPS objects through Python as mentioned above. The input script also reports the principle outputs and progress of the simulations as comments, so one does not need to search for these information in the clutter of LAMMPS logfile.
- The code can run LAMMPS both in serial and in parallel (using mpi4py Python module).
- The code implements some automatic routines to energy minimize complex atomistic structures with many defects and few million atoms (like in nanocrystalline systems) to an fmax of below 1e-8 eV/A.
- The folder *Atomistic_configurations* has classes to construct different atomistic configurations with or without defects, for example, system with periodic array of dislocations and bicrystals with a planar grain boundary.
- The folder *LAMMPS_logfile_parser* has classes to read LAMMPS logfiles and extract thermo output.    

*lat_param_n_cohE_0K.py* is an example code which calculates the cohesive energy and lattice parameter of a fcc 25% Fe and 75% Ni random alloy system using EAM interatomic potential. 
