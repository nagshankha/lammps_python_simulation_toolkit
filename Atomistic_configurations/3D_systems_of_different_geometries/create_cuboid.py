from buffer import Buffer
from system import System
import numpy as np

class SystemCuboid(System):
   
   def __init__(self, bf:Buffer, cuboid_dims:float, n_atom_types:int):
      
      System.__init__(self, bf, n_atom_types)
      self.cuboid_dims = cuboid_dims
      
###################################################################

   def initiate_LAMMPS_system(self, log_filename='none', folder='./',
                              LAMMPS_units = 'metal', 
                              LAMMPS_atom_style = 'atomic',
                              LAMMPS_boundary_cond = ['p', 'p', 'p'],
                              wd = None):
      
      System.initiate_LAMMPS_system(self, log_filename, folder, LAMMPS_units,
                                    LAMMPS_atom_style, wd)
      self.bf.assign_boundary_condition(LAMMPS_boundary_cond)     
      
      
###################################################################      
      
   def create_cuboid(self, simul_box_dims=None):
      
      self.bf.icommand('region reg_cuboid block {0} units box'.format(
                          ' '.join(map(str, self.cuboid_dims))))
      if simul_box_dims is None:
         self.bf.icommand('create_box {0} reg_cuboid'.format(self.n_atom_types))
      else:
         self.define_cuboid_simul_box(simul_box_dims)
      
################################################################### 

   def fill_in_atoms(self, style='region', N=None, seed=None, atom_type=1):

      if style == 'box':
         self.bf.fill_in_atoms('box', atom_type)
      elif style == 'region':
         self.bf.fill_in_atoms('region', 'reg_cuboid', atom_type)
      elif style == 'random':
         self.bf.fill_in_atoms('random', [N, seed, 'reg_cuboid'], atom_type)
      else:
         raise ValueError('For creating nanoparticle using this class, the '+
                          'style needs to either "box" or "region" or "random"')
         
      self.n_atoms = self.bf.lmp.get_natoms()
      
         
################################################################### 
   
   def __setattr__(self, name, value):
      
      System.__setattr__(self, name, value)
            
      if name == 'cuboid_dims':
         if not isinstance(value, (np.ndarray, list)):
            raise ValueError('"cuboid_dims" must be a numpy array or a list')
         elif len(value) != 6:
            raise ValueError('"cuboid_dims" must have 6 entries: '+
                             '[xlo xhi ylo yhi zlo zhi]')
         elif not all([isinstance(x, (int, float)) for x in value]):
            raise ValueError('Entries in "cuboid_dims" must be integers or floats')
         elif value[1] < value[0]:
            raise ValueError('xlo must be less than xhi')
         elif value[3] < value[2]:
            raise ValueError('ylo must be less than yhi')
         elif value[5] < value[4]:
            raise ValueError('zlo must be less than zhi')
         else:
            self.__dict__[name] = value
            
      else:
         pass 
      
################################################################### 