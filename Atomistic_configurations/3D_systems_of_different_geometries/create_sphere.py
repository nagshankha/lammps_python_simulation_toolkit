from buffer import Buffer
from system import System

class SystemSpherical(System):
   
   def __init__(self, bf:Buffer, sphere_radius:float, n_atom_types:int):
      
      System.__init__(self, bf, n_atom_types)
      self.sphere_radius = sphere_radius
      
###################################################################

   def initiate_LAMMPS_system(self, log_filename='none', folder='./',
                              LAMMPS_units = 'metal', 
                              LAMMPS_atom_style = 'atomic',
                              LAMMPS_boundary_cond = ['p', 'p', 'p'],
                              wd = None):
      
      System.initiate_LAMMPS_system(log_filename, folder, LAMMPS_units,
                                    LAMMPS_atom_style, wd)
      self.bf.assign_boundary_condition(LAMMPS_boundary_cond)     
      
      
###################################################################      
      
   def create_sphere(self, simul_box_dims=None):
      
      self.bf.icommand('region reg_sphere sphere 0 0 0 {0} units box'.format(
                          self.sphere_radius))
      if simul_box_dims is None:
         self.bf.icommand('create_box {0} reg_sphere'.format(self.n_atom_types))
      else:
         self.define_cuboid_simul_box(simul_box_dims)
      
################################################################### 

   def fill_in_atoms(self, style='region', N=None, seed=None, atom_type=1):

      if style == 'region':
         self.bf.fill_in_atoms('region', 'reg_sphere', atom_type)
      elif style == 'random':
         self.bf.fill_in_atoms('random', [N, seed, 'reg_sphere'], atom_type)
      else:
         raise ValueError('For creating nanoparticle using this class, the '+
                          'style needs to either "region" or "random"')
         
      self.n_atoms = self.bf.lmp.get_natoms()
      
         
################################################################### 
   
   def __setattr__(self, name, value):
      
      System.__setattr__(self, name, value)
            
      if name == 'sphere_radius':
         if isinstance(value, float): 
            self.__dict__[name] = value
         else:
            raise ValueError('Member variable "sphere_radius" for the size of '+
                             'the nanoparticle must be a float')
            
      else:
         pass 
      
################################################################### 