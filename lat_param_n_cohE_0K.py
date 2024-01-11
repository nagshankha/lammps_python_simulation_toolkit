import numpy as np
from create_cuboid import SystemCuboid
from ms import MolecularStatics
from buffer import Buffer
from mpi4py import MPI
import sys

"""
Calculation of cohesive energy and lattice parameter of a fcc system 
with 25% of Fe atoms and 75% of Ni atoms randomly arranged.

"""

####################################
######### INPUTS: Begin ############
####################################

n_atom_types = 2
concentration = np.array([0.25, 0.75])
initial_lattice_parameter = 3.52
side_length = 10 # in lattice units
path2potfile = 'Fe-Ni_2009_BonnyPasianotMalerba.eam.alloy'
atomtype_names = ['Fe', 'Ni']
folder=sys.argv[1]

####################################
######### INPUTS: End ##############
####################################


comm_world = MPI.COMM_WORLD
bf = Buffer(comm_world=comm_world)

bf.update_buffer_for_LAMMPS(input_file_path=folder+'lat_param_n_cohE.in', 
                            append_to_input_file=True)

if (not isinstance(side_length, int)) or (side_length%2 != 0) :
   raise ValueError('Side length must be even integer')

cuboid_dims = int(side_length/2)*np.array(3*[-1,1])*initial_lattice_parameter
sys_cuboid = SystemCuboid(bf, cuboid_dims, n_atom_types)  
sys_cuboid.initiate_LAMMPS_system(log_filename='lat_param_n_cohE.log',
                                  folder=folder)
sys_cuboid.create_cuboid()
sys_cuboid.define_crystal_structure('fcc', initial_lattice_parameter, 
                                    keyword_value_pairs={'origin':[0.25]*3})
sys_cuboid.fill_in_atoms()
n_atoms = bf.lmp.get_natoms()
sys_cuboid.assign_random_atomtypes(concentration=concentration)

bf.add_potential_eam(path2potfile, atomtype_names, style_appendage='alloy')

thermo_list = ['step', 'fmax', 'fnorm', 'pe', 'press', 'vol', 'lx', 'ly', 
               'lz', 'pxx', 'pyy', 'pzz', 'pxy', 'pxz', 'pyz']
ms = MolecularStatics(bf, thermo_list, boxrelax_dirs=['x', 'y', 'z'],
                      ms_diagnotic_dumps_location = (folder+
                      './MS_diagnotic_dumps_for_lattice_parameter_n_cohesive_energy/') )
ms.run(mode='S')

bf.run0()

bf.write_data_lmp(folder, 'random_alloy_energy_minimized_configuration.data',
                  get_ids=True, get_images=True, n_types = 2,
                  file_description=('Energy minimized configuration of certain '+
                                    'realization of FeNi3 random alloy'))

box_dims = bf.get_box_dims()
x_lat_param = (box_dims[1] - box_dims[0])/ side_length
y_lat_param = (box_dims[3] - box_dims[2])/ side_length
z_lat_param = (box_dims[5] - box_dims[4])/ side_length
   
pe1 = bf.lmp.get_thermo('pe')
cohesive_energy = pe1/n_atoms

bf.iwhitespace(2)
bf.icomment('############ *** ############')
bf.icomment('Results I: Lattice parameter and Cohesive Energy')
bf.iwhitespace()
bf.icomment('Cohesive energy is {0} eV/atom'.format(cohesive_energy))
if n_atom_types == 1:
   if np.allclose([x_lat_param, y_lat_param], z_lat_param):
      bf.icomment('Lattice parameter is {0} A'.format(z_lat_param))
   else:
      raise RuntimeError('Lattice parameters in x, y '+
                         'and z directions are {0}, {1} and {2} '.format(
                            x_lat_param, y_lat_param, z_lat_param) +
                         'Angstroms respectively. For elemental fcc systems, lattice '+
                         'parameter is all directions must be equal!')
else:
   bf.icomment('Lattice parameter in x direction is {0} A'.format(
                  x_lat_param))
   bf.icomment('Lattice parameter in y direction is {0} A'.format(
                  y_lat_param))
   bf.icomment('Lattice parameter in z direction is {0} A'.format(
                  z_lat_param))
bf.iwhitespace()
bf.icomment('############ *** ############')

bf.lmp.close()
