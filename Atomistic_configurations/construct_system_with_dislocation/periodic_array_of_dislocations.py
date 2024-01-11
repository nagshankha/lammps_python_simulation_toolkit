import numpy as np
from system import System
import elastic_anisotropy_and_dislocations as elastic_field

class PeriodicArrayDislocations:
   
   def __init__(self, sys, lattice_parameter, 
                lattice_spacing_along_glide_dir=np.sqrt(0.5),
                lattice_spacing_along_line_dir=np.sqrt(1.5), #Default values are for fcc
                diagnostics = False, diagnostics_folder = './diagnostics/'): 
      
      self.sys = sys
      self.lattice_parameter = lattice_parameter
      self.lattice_spacing_along_glide_dir = lattice_spacing_along_glide_dir
      self.lattice_spacing_along_line_dir = lattice_spacing_along_line_dir
      
      self.__dict__['_diagnostics'] = diagnostics
      self.__dict__['_diagnostics_folder'] = diagnostics_folder
      
      if self._diagnostics:
         self.sys.bf.write_dump_lmp('all', self._diagnostics_folder, 
                                ( 'just_added_to_disl_array_class.mpiio.dump' ), 
                                ['id', 'type', 'x', 'y', 'z', 'ix', 'iy', 'iz'])
   
   def getDislInputs(self, slip_vectors = {'x': [1.], 'y': [0.]}, 
                     disl_pos = {'x': [0.], 'z': [0.]}):
      # Default values are for an edge dislocation at the origin with glide 
      # plane normal along z
      # There can be no component of slip vector along the slip plane normal
      # slip vectors must be in terms of the lattice spacing in the respective
      # directions
      # the dislocation positions must be in the terms of the global coordinate
      # system of the simulation cell or atomic positions, for example Angstrom
      
      self.slip_vectors = slip_vectors
      self.disl_pos = disl_pos
      
      if len(np.unique(list(map(len, list(slip_vectors.values())+list(disl_pos.values()))))) != 1:
         raise ValueError('"slip_vectors" and "disl_pos" must have info for '+
                          'same number of dislocations')
      elif set(slip_vectors.keys()) == set(disl_pos.keys()):
         raise ValueError('Both "slip_vectors" and "disl_pos" cannot have the '+
                          'same keys')
      else:
         self.__dict__['line_dir'] = (set(self.slip_vectors.keys())-
                                       set(self.disl_pos.keys())).pop()
         self.__dict__['slip_plane_normal'] = (set(self.disl_pos.keys())-
                                               set(self.slip_vectors.keys())).pop()
         self.__dict__['glide_dir'] = (set(self.slip_vectors.keys()) &
                                       set(self.disl_pos.keys())).pop()
         
      inds_disl_same_plane = []
      unique_plane_pos = np.unique(np.round(disl_pos[self.slip_plane_normal], 
                                            decimals=4))
      for ppos in unique_plane_pos:
         inds_disl_same_plane.append(np.nonzero(np.isclose(
            np.round(disl_pos[self.slip_plane_normal], decimals=4), ppos))[0])
         
      if ( np.sum(list(map(len, inds_disl_same_plane))) != self.n_disl  ):
         raise RuntimeError('some dislocations went missing. check!')
         
      ######## No total partial slip on any glide plane (Check this section !!!) ##############
         
      if not np.all(np.logical_or(
             np.isclose([abs(np.sum(np.array(slip_vectors[self.glide_dir])[x]) )
                          for x in inds_disl_same_plane], 1), 
             np.isclose([np.sum(np.array(slip_vectors[self.glide_dir])[x])   
                          for x in inds_disl_same_plane], 0))):
         raise ValueError('slip_vectors in glide direction for all dislocations '+
                          'on the same plane must sum up to a full lattice spacing '+
                          'in glide direction')
      else:
         self.slip_vectors[self.glide_dir] = list( np.array(self.slip_vectors[self.glide_dir])*
                                               self.lattice_spacing_along_glide_dir*
                                               self.lattice_parameter )
      
      if not np.all(np.logical_or(
             np.isclose([abs(np.sum(np.array(slip_vectors[self.line_dir])[x]) )
                          for x in inds_disl_same_plane], 1),
             np.isclose([np.sum(np.array(slip_vectors[self.line_dir])[x]) 
                          for x in inds_disl_same_plane], 0))):
         raise ValueError('slip_vectors in line direction for all dislocations '+
                          'on the same plane must sum up to a full lattice spacing '+
                          'in line direction')
      else:
         self.slip_vectors[self.line_dir] = list( np.array(self.slip_vectors[self.line_dir])*
                                               self.lattice_spacing_along_line_dir*
                                               self.lattice_parameter )
         
      ###########################################
         
      ind_line_dir = ['x', 'y', 'z'].index(self.line_dir)
      ind_glide_dir = ['x', 'y', 'z'].index(self.glide_dir)
      ind_slip_plane_normal = ['x', 'y', 'z'].index(self.slip_plane_normal)
      
      
      if self.sys.lmp.extract_box()[-2][ind_line_dir] == 0:
         raise RuntimeError('Simulation cell must be periodic along dislocation line in '+
                            self.line_dir)
         
      if self.sys.lmp.extract_box()[-2][ind_glide_dir] == 0:
         raise RuntimeError('Simulation cell must be periodic along '+ 
                            self.glide_dir)
         
      if self.sys.lmp.extract_box()[-2][ind_slip_plane_normal] == 1:
         raise RuntimeError('Simulation cell must be non-periodic along '+ 
                            self.slip_plane_normal)
         
   
   def compute_disp_field_iso(self, nu = 0.33):
      
      ind_line_dir = ['x', 'y', 'z'].index(self.line_dir)
      atom_pos = self.sys.bf.gather_peratom_attr('x')
      burgers_vectors = np.c_[[self.slip_vectors[x] if x in self.slip_vectors.keys() 
                                 else ([0]*self.n_disl) for x in ['x', 'y', 'z']]].T
      disl_pos = np.c_[[self.disl_pos[x] if x in self.disl_pos.keys() 
                                 else ([0]*self.n_disl) for x in ['x', 'y', 'z']]].T 
      disp = np.zeros((self.sys.lmp.get_natoms(), 3))
      for i, b in enumerate(burgers_vectors):
         disp = (disp + 
                 elastic_field.isotropic_soln(1.0, nu, np.eye(3)[ind_line_dir], 
                                              b, atom_pos-disl_pos[i])[0])
      self.__dict__['disp'] = disp
      
   def compute_disp_field_aniso(self, C):
      
      ind_line_dir = ['x', 'y', 'z'].index(self.line_dir)
      ind_glide_dir = ['x', 'y', 'z'].index(self.glide_dir)
      ind_slip_plane_normal = ['x', 'y', 'z'].index(self.slip_plane_normal)
      atom_pos = self.sys.bf.gather_peratom_attr('x')
      burgers_vectors = np.c_[[self.slip_vectors[x] if x in self.slip_vectors.keys() 
                                 else ([0]*self.n_disl) for x in ['x', 'y', 'z']]].T
      disl_pos = np.c_[[self.disl_pos[x] if x in self.disl_pos.keys() 
                                 else ([0]*self.n_disl) for x in ['x', 'y', 'z']]].T 
      disp = np.zeros((self.sys.n_atoms(), 3))
      for i, b in enumerate(burgers_vectors):
         disp = (disp + 
                 elastic_field.anisotropic_soln(C, np.eye(3)[ind_line_dir], 
                                              b, atom_pos-disl_pos[i], 
                                              m0 = np.eye(3)[ind_glide_dir],
                                              n0 = np.eye(3)[ind_slip_plane_normal],
                                              method='stroh')[0])
      self.__dict__['disp'] = disp
   
   def __increase_box_length_along_edge_component(self):
      
      ind_glide_dir = ['x', 'y', 'z'].index(self.glide_dir)
      self.__dict__['box_dims'] = self.sys.bf.get_box_dims().reshape(-1,2)
      self.__dict__['final_glide_dims'] = ( 
                         self.box_dims[ind_glide_dir] + 
                         (np.array([1.,-1.])*self.lattice_parameter*
                          self.lattice_spacing_along_glide_dir) )
      new_glide_dims = ( self.box_dims[ind_glide_dir] + 
                         (np.array([-1.5,1.5])*self.lattice_parameter*
                          self.lattice_spacing_along_glide_dir) )
      self.sys.bf.icommand(f'change_box all {self.glide_dir} final '+
                           f'{new_glide_dims[0]} {new_glide_dims[1]} units box')
      
      if self._diagnostics:
         self.sys.bf.write_dump_lmp('all', self._diagnostics_folder, 
                                ( 'increased_box_lengths.mpiio.dump' ), 
                                ['id', 'type', 'x', 'y', 'z', 'ix', 'iy', 'iz'])
   
   def __impose_displacement_field(self):
      
      atom_pos = self.sys.bf.gather_peratom_attr('x')
      self.sys.bf.set_peratom_attr('x', atom_pos+self.disp)
      
      if self._diagnostics:
         self.sys.bf.write_dump_lmp('all', self._diagnostics_folder, 
                                ( 'displacements_imposed.mpiio.dump' ), 
                                ['id', 'type', 'x', 'y', 'z', 'ix', 'iy', 'iz'])
   
   def __level_slip_step(self, reassign_atom_ids='yes'):
      
      if self.glide_dir == 'x':
         self.sys.bf.icommand(f'region block1 block INF {self.final_glide_dims[0]} '+
                              'INF INF INF INF side in units box')
         self.sys.bf.icommand(f'region block2 block {self.final_glide_dims[1]} INF '+
                              'INF INF INF INF side in units box')
      if self.glide_dir == 'y':
         self.sys.bf.icommand(f'region block1 block INF INF INF {self.final_glide_dims[0]} '+
                              'INF INF side in units box')
         self.sys.bf.icommand(f'region block2 block INF INF {self.final_glide_dims[1]} '+
                              'INF INF INF side in units box')
      if self.glide_dir == 'z':
         self.sys.bf.icommand('region block1 block INF INF INF INF INF '+
                              f'{self.final_glide_dims[0]} side in units box')
         self.sys.bf.icommand('region block2 block INF INF INF INF '+
                              f'{self.final_glide_dims[1]} INF side in units box')
         
      self.sys.bf.icommand('region block2delete union 2 block1 block2')
      self.sys.bf.icommand('delete_atoms region block2delete compress '+
                           f'{reassign_atom_ids}')
      
      if self._diagnostics:
         self.sys.bf.write_dump_lmp('all', self._diagnostics_folder, 
                                ( 'atoms_deleted.mpiio.dump' ), 
                                ['id', 'type', 'x', 'y', 'z', 'ix', 'iy', 'iz'])
   
   def __reset_box_dimension(self):
      
      self.sys.bf.icommand(f'change_box all {self.glide_dir} final '+
                           f'{self.final_glide_dims[0]} {self.final_glide_dims[1]} '+
                           'units box')
      
      if self._diagnostics:
         self.sys.bf.write_dump_lmp('all', self._diagnostics_folder, 
                                ( 'reset_box_dims.mpiio.dump' ), 
                                ['id', 'type', 'x', 'y', 'z', 'ix', 'iy', 'iz'])
   
   def construct_dislocation(self, reassign_atom_ids='yes'):
      # reassign_atom_ids decide whether or not to reassign the atom ids from 
      # 1 to N after deleting the atoms in the glide direction
      
      self.__increase_box_length_along_edge_component()
      self.__impose_displacement_field()
      self.__level_slip_step(reassign_atom_ids)
      self.__reset_box_dimension()
      
      
   def __setattr__(self, name, value):
      
      if name == 'sys':
         if isinstance(value, System):
            self.__dict__[name] = value
         else:
            raise ValueError('"sys" must be an instance of class System')
            
      elif name == 'slip_vectors':
         if isinstance(value, dict):
            if ( set(value.keys()).issubset({'x', 'y', 'z'}) and 
                 len(value) == 2 and 
                 np.all([isinstance(x, list) for x in value.values()]) ):
               if len(np.unique(list(map(len, value.values())))) == 1:
                  self.__dict__[name] = value
                  self.__dict__['n_disl'] = np.unique(list(map(len, value.values())))[0]
                  self.sys.bf.icomment(f'{self.n_disl} dislocations are requested '+
                                       'for construction')
               else:
                  raise ValueError('There must be at least one dislocation')
            else:
               raise ValueError('Each slip vector must have two components '+
                                'in glide and in line direction, even if one '+
                                'of them is zero'+
                                'Therefore, "slip_vectors" must be a dictionary with 2 keys '+
                                'from x, y or z and each value must be a list')
         else:
            raise ValueError('"slip_vectors" must be a dictionary')        
            
      elif name == 'disl_pos':
         if isinstance(value, dict):
            if ( set(value.keys()).issubset({'x', 'y', 'z'}) and 
                 len(value) == 2 and 
                 np.all([isinstance(x, list) for x in value.values()]) ):
               if np.all(np.array(list(map(len, value.values()))) == self.n_disl):
                  self.__dict__[name] = value
               else:
                  raise ValueError('There must be as many dislocation positions '+
                                   'as there are slip vectors')
            else:
               raise ValueError('"disl_pos" must be a dictionary with 2 keys '+
                                'from x, y or z and each value must be a list')
         else:
            raise ValueError('"disl_pos" must be a dictionary')          
            
      else:
         self.__dict__[name] = value
