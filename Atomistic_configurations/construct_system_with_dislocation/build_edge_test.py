import sys, os
sys.path.extend([x[0] for x in os.walk(os.environ['HOME']+'/Some_useful_codes/') 
                 if (x[0].split('/')[-1] != '__pycache__' and 
                     x[0].split('/')[5] != 'LAMMPS_log_file_reader')])
from create_cuboid import SystemCuboid
from buffer import Buffer
#from mpi4py import MPI
import numpy as np
from periodic_array_of_dislocations import PeriodicArrayDislocations

#comm_world = MPI.COMM_WORLD
bf = Buffer(comm_world=None)

bf.update_buffer_for_LAMMPS(input_file_path='./build_edge_test.in', 
                            append_to_input_file=True)

n_atom_types = 1
lattice_parameter = 3.52      
orient = {'x': [1, -1, 0],
          'y': [1, 1, -2],
          'z': [1, 1, 1]}
lattice_spacing = np.array([1/np.sqrt(2), 0.5*np.sqrt(6), np.sqrt(3)])
multiplicity = np.array([2, 6, 3])
cuboid_dims = np.array([-150, 300, -5, 5, -30, 30])
cuboid_dims = np.array(cuboid_dims)*np.repeat(lattice_spacing,2)*lattice_parameter

sys_cuboid = SystemCuboid(bf, cuboid_dims, n_atom_types)  

sys_cuboid.initiate_LAMMPS_system(log_filename='build_edge_test.log',
                                  folder='./', LAMMPS_boundary_cond = ['p', 'p', 's'])

sys_cuboid.create_cuboid()
sys_cuboid.define_crystal_structure('fcc', lattice_parameter, 
                                      keyword_value_pairs={
                                         'origin': 0.5/multiplicity,
                                         'orient x': orient['x'],
                                         'orient y': orient['y'],
                                         'orient z': orient['z'],
                                         'spacing': lattice_spacing})
sys_cuboid.fill_in_atoms()


bf.write_data_lmp('./build_system/', 'initial_solid.data', n_types=n_atom_types, 
                  write_w_LAMMPS=False, get_ids = True, get_images=True,
                  file_description=('fcc solid created in cubic orientation '+
                                    'with lattice parameter {0}. '.format(
                                     lattice_parameter)+'There are {0} '.format(
                                     n_atom_types)+'atom types'))
                                        
pad_obj = PeriodicArrayDislocations(sys_cuboid, lattice_parameter)

#Edge_dislocation
pad_obj.getDislInputs()
pad_obj.compute_disp_field_iso()
pad_obj.construct_dislocation()

bf.write_data_lmp('./build_system/', 'as_constructed_solid_w_disl.data', n_types = n_atom_types, 
                  write_w_LAMMPS=False, get_ids = True, get_images=True,
                  file_description=('fcc solid with one edge dislocation created '+
                                    'in cubic orientation '+
                                    'with lattice parameter {0}. '.format(
                                     lattice_parameter)+'There are {0} '.format(
                                     n_atom_types)+'atom types'))

