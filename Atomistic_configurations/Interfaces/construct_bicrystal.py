import numpy as np
from system import System
from bicrystals_crystallography import BicrystalCrystallography
import miscellaneous_computes_crystallography as misc_crys


class ConstructBicrystal(System, BicrystalCrystallography):

   def __init__(self, bf, lat_scale, n_atom_types, **bicrys_cryst_inputs):
   
      System.__init__(self, bf, n_atom_types)
      self.lat_scale = lat_scale 
      self.n_atom_types = n_atom_types
      
      if not set(['rot_axis', 'rot_angle', 'z_dir', 'x_dir', 'lattice']).issubset(
            bicrys_cryst_inputs.keys()):
         raise RuntimeError('The mandatory inputs to the constructor of the '+
                            'class BicrystalCrystallography must be entered')
      else:
         rot_axis = bicrys_cryst_inputs['rot_axis']
         rot_angle = bicrys_cryst_inputs['rot_angle']
         z_dir = bicrys_cryst_inputs['z_dir']
         x_dir = bicrys_cryst_inputs['x_dir']
         lattice = bicrys_cryst_inputs['lattice']
         if 'rot_angle_conv' in bicrys_cryst_inputs:
            rot_angle_conv = bicrys_cryst_inputs['rot_angle_conv']
         else:
            rot_angle_conv = 'deg'

      BicrystalCrystallography.__init__(self, rot_axis, rot_angle, z_dir, x_dir, 
                                  lattice, rot_angle_conv)     
   
   #################################################################
   
   def write_info(self, filename, folder = './'):
      
      from buffer import ProxyCommunicator
      if isinstance(self.bf.comm, ProxyCommunicator):
         BicrystalCrystallography.write_info(self, filename, folder=folder)
      else:
         if self.bf.comm.rank == 0:
            BicrystalCrystallography.write_info(self, filename, folder=folder)
         self.bf.comm.Barrier()      
      
   #################################################################
   
   def construct_system_z_periodic(self, crys1_dims, crys2_dims, pbc_x_y, 
                        origin_crys1 = None, origin_crys2 = None,
                        disp_boundary=0.0, disp_boundary_lat_units='crys1',
                        folder = './', log_filename = 'bicrys', wd=None):
   
      """
      
      crys1_dims are defined w.r.t. crys1 lattice spacings
      crys2_dims are defined w.r.t. crys2 lattice spacings
      disp_boundary are given w.r.t. z-lattice spacing of Crystal 1 (default)
                  or Crystal 2, depending on the choice in disp_boundary_lat_units
            
      crys1_dims[-1] and crys2_dims[-2] are set to zero.
      And crys1_dims[-2] and crys2_dims[-1] are set to:
      crys1_dims[-2] = - ( crys1_dims[-1] - crys1_dims[-2] ) and
      crys2_dims[-1] = crys2_dims[-1] - crys2_dims[-2]
      
      The position of the two grain boundaries are controlled by disp_boundary.
      For disp_boundary == 0.0 (default), there will be a grain boundary inside 
      the simulation box with normal aligned with the z axis and the other 
      grain boundary will be along the z-periodic boundary plane of the 
      simulation box.
      For disp_boundary != 0.0, both grain boundaries will be inside the 
      simulation box
      
      disp_boundary == 0:
   
        ------************************------- periodic z simulation box boundary
                 Crystal 2
      
              ************************ z=0
                 Crystal 1
      
        ------************************------- periodic z simulation box boundary
      
       disp_boundary > 0:
      
        ------------------------------------- periodic z simulation box boundary
                 Crystal 2
              ************************
              
                 Crystal 1             z=0
      
              ************************
                 Crystal 2
        ------------------------------------- periodic z simulation box boundary
      
       disp_boundary < 0:
      
        ------------------------------------- periodic z simulation box boundary
                 Crystal 1
              ************************
              
                 Crystal 2             z=0
      
              ************************
                 Crystal 1
        ------------------------------------- periodic z simulation box boundary
      
      
      """
      
      
      
      if not crys1_dims.dtype == int:
         raise ValueError('Box dimensions of crystal 1 must be provided in integer '+
                          'multiples of lattice spacings of crystal 1')
         
      if not crys2_dims.dtype == int:
         raise ValueError('Box dimensions of crystal 2 must be provided in integer '+
                          'multiples of lattice spacings of crystal 2')
      
      # Rescaling crys1_dims and crys2_dims along z direction
      crys1_dims[-2] = - ( crys1_dims[-1] - crys1_dims[-2] )
      crys2_dims[-1] = crys2_dims[-1] - crys2_dims[-2]
      crys1_dims[-1] = 0; crys2_dims[-2] = 0
      
      crys1_dims = crys1_dims.astype(float); crys2_dims = crys2_dims.astype(float)
      # The box dimension in z direction must be integer multiple of lattice spacings
      # plus one interplanar spacing
      crys1_dims[-2] = crys1_dims[-2] - (1.0/self.multiplicity_crys1[2])
      crys2_dims[-1] = crys2_dims[-1] + (1.0/self.multiplicity_crys2[2])
      
      
      # (crys1_dims[1]-crys1_dims[0]):(crys2_dims[1]-crys2_dims[0]) must be in
      # x_spacing_ratio
      if not ( ( np.allclose(np.array([crys1_dims[0], crys2_dims[0]]), 0.0) or
                 all( misc_crys.get_miller_indices(abs(np.array([crys1_dims[0], crys2_dims[0]]))) 
                      == self.x_spacing_ratio ) ) and 
               ( np.allclose(np.array([crys1_dims[1], crys2_dims[1]]), 0.0) or
                 all( misc_crys.get_miller_indices(abs(np.array([crys1_dims[1], crys2_dims[1]]))) 
                      == self.x_spacing_ratio ) ) ):
         raise ValueError('The x box lengths entered for crystal 1 and 2 should be '+
                          'ratio {0[0]}:{0[1]}'.format(self.x_spacing_ratio))
         
      # (crys1_dims[3]-crys1_dims[2]):(crys2_dims[3]-crys2_dims[2]) must be in
      # y_spacing_ratio
      if not ( ( np.allclose(np.array([crys1_dims[2], crys2_dims[2]]), 0.0) or
                 all( misc_crys.get_miller_indices(abs(np.array([crys1_dims[2], crys2_dims[2]]))) 
                      == self.y_spacing_ratio ) ) and 
               ( np.allclose(np.array([crys1_dims[3], crys2_dims[3]]), 0.0) or
                 all( misc_crys.get_miller_indices(abs(np.array([crys1_dims[3], crys2_dims[3]]))) 
                      == self.y_spacing_ratio ) ) ):
         raise ValueError('The y box lengths entered for crystal 1 and 2 should be '+
                          'ratio {0[0]}:{0[1]}'.format(self.y_spacing_ratio))
      
      if (not isinstance(pbc_x_y, list)) or (len(pbc_x_y) != 2):
         raise ValueError('"pbc_x_y" must be a list of length 2')
      elif pbc_x_y in [['p', 'p'], ['p', 's'], ['s', 'p']]:
         LAMMPS_boundary_cond = pbc_x_y + ['p']
      else:
         raise ValueError('"pbc_x_y" must be either [p, p] or [p, s] or [s, p]')
                  
      
      self.initiate_LAMMPS_system(log_filename=log_filename, folder=folder,
                                  wd=wd)
      self.bf.assign_boundary_condition(LAMMPS_boundary_cond)
      
      if origin_crys1 is None:
         origin_crys1 = 0.5/self.multiplicity_crys1
      else:
         if not isinstance(origin_crys1, np.ndarray):
            raise ValueError('"origin_crys1" must be a numpy array')
         elif np.shape(origin_crys1) != (3,):
            raise ValueError('"origin_crys1" must have shape (3,)')
         elif not np.all(np.r_[origin_crys1>=0, origin_crys1<1]):
            raise ValueError('"origin_crys1" must have values 0<= x,y,z < 1')
         else:
            pass
      
      self.define_crystal_structure(self.lat_name, self.lat_scale, 
                                    keyword_value_pairs={
                                                'origin': origin_crys1,
                                                'orient x': self.orient_crys1[0],
                                                'orient y': self.orient_crys1[1],
                                                'orient z': self.orient_crys1[2],
                                                'spacing': self.spacing_crys1
                                                })
         
      crys1_dims = ( (crys1_dims*np.repeat(self.spacing_crys1, 2)*self.lat_scale) + 
                      (np.array([-1, 1]*3)*1e-10) )
      crys2_dims = ( (crys2_dims*np.repeat(self.spacing_crys2, 2)*self.lat_scale) + 
                      (np.array([-1, 1]*3)*1e-10) )
      if not np.allclose(crys1_dims[:-2], crys2_dims[:-2]) :
         raise RuntimeError('The x and y dimensions of the two crystals do not match')
      else:
         system_dims = np.r_[crys1_dims[:-2], crys1_dims[-2], crys2_dims[-1]]
         
      self.bf.icommand('region full_box block {0} units box'.format(' '.join(map(str, 
                                                                              system_dims))) )
      self.bf.icommand('create_box {0} full_box'.format(self.n_atom_types))
      
      self.bf.icommand('region crys1_region block INF INF INF INF INF {0} units box'.format(1e-10))
      self.bf.icommand('create_atoms 1 region crys1_region')
      self.bf.icommand('group crys1_group region crys1_region')
      self.n_atoms_crys1 = self.bf.lmp.get_natoms()
      self.bf.icomment('There are {0} atoms in crystal 1'.format(self.n_atoms_crys1))
      self.atom_indices_crys1 = np.arange(self.n_atoms_crys1)
      
      if origin_crys2 is None:
         origin_crys2 = 0.5/self.multiplicity_crys2
      else:
         if not isinstance(origin_crys2, np.ndarray):
            raise ValueError('"origin_crys2" must be a numpy array')
         elif np.shape(origin_crys2) != (3,):
            raise ValueError('"origin_crys2" must have shape (3,)')
         elif not np.all(np.r_[origin_crys2>=0, origin_crys2<1]):
            raise ValueError('"origin_crys2" must have values 0<= x,y,z < 1')
         else:
            pass
      
      self.define_crystal_structure(self.lat_name, self.lat_scale, 
                                    keyword_value_pairs={
                                                'origin': origin_crys2,
                                                'orient x': self.orient_crys2[0],
                                                'orient y': self.orient_crys2[1],
                                                'orient z': self.orient_crys2[2],
                                                'spacing': self.spacing_crys2
                                                })
      
      self.bf.icommand('region crys2_region block INF INF INF INF -{0} INF units box'.format(1e-10))
      self.bf.icommand('create_atoms 1 region crys2_region')
      self.bf.icommand('group crys2_group subtract all crys1_group')
      self.n_atoms = self.bf.lmp.get_natoms()
      self.n_atoms_crys2 = self.n_atoms - self.n_atoms_crys1
      self.bf.icomment('There are {0} atoms in crystal 2'.format(self.n_atoms_crys2))
      self.atom_indices_crys2 = np.arange(self.n_atoms_crys1, self.n_atoms)
      
      if disp_boundary_lat_units == 'crys1':
         disp_boundary = disp_boundary*self.lat_scale*self.spacing_crys1[2]
      elif disp_boundary_lat_units == 'crys2':
         disp_boundary = disp_boundary*self.lat_scale*self.spacing_crys2[2]   
      else:
         raise ValueError('"disp_boundary_lat_units" must either be "crys1" or '+
                          '"crys2"')
         
      if abs(disp_boundary) > np.max(abs(system_dims[-2:])):
         raise ValueError('Cannot move that boundary outside the cell dimensions even if periodic!')
      
      self.bf.displace_atoms([0, 0, disp_boundary], group='all')
      
      z_interplanar_spacing_crys1 = 1.0/self.multiplicity_crys1[2]
      z_offset_crys1 = (origin_crys1[2]+1e-10) % z_interplanar_spacing_crys1
      z_offset_crys1 = z_offset_crys1*self.lat_scale*self.spacing_crys1[2]
      z_interplanar_spacing_crys1 = ( z_interplanar_spacing_crys1*self.lat_scale
                                     *self.spacing_crys1[2] )
      
      z_interplanar_spacing_crys2 = 1.0/self.multiplicity_crys2[2]
      z_offset_crys2 = (origin_crys2[2]+1e-10) % z_interplanar_spacing_crys2
      z_offset_crys2 = z_offset_crys2*self.lat_scale*self.spacing_crys2[2]
      z_interplanar_spacing_crys2 = ( z_interplanar_spacing_crys2*self.lat_scale
                                     *self.spacing_crys2[2] )
      
            
      if np.sign(disp_boundary) == -1:
         self.boundary_locs = np.array([disp_boundary, system_dims[-1]+disp_boundary])
         
         if np.isclose(z_offset_crys1, 0.0):
            self.boundary_locs[0] = self.boundary_locs[0] + (0.5*z_offset_crys2)
         else:
            self.boundary_locs[0] = ( self.boundary_locs[0]
                                    + 0.5*( -z_interplanar_spacing_crys1
                                           + z_offset_crys1 + z_offset_crys2 )
                                    )
                                    
         if np.isclose(z_offset_crys2, 0.0):      
            self.boundary_locs[1] = self.boundary_locs[1] + (0.5*z_offset_crys1)
         else:
            self.boundary_locs[1] = ( self.boundary_locs[1]
                                    + 0.5*( -z_interplanar_spacing_crys2
                                           + z_offset_crys1 + z_offset_crys2 )
                                    )
      else:
         self.boundary_locs = np.array([system_dims[-2]+disp_boundary, disp_boundary])
         
         if np.isclose(z_offset_crys2, 0.0):      
            self.boundary_locs[0] = self.boundary_locs[0] + (0.5*z_offset_crys1)
         else:
            self.boundary_locs[0] = ( self.boundary_locs[0]
                                    + 0.5*( -z_interplanar_spacing_crys2
                                           + z_offset_crys1 + z_offset_crys2 )
                                    )
                                    
         if np.isclose(z_offset_crys1, 0.0):
            self.boundary_locs[1] = self.boundary_locs[1] + (0.5*z_offset_crys2)
         else:
            self.boundary_locs[1] = ( self.boundary_locs[1]
                                    + 0.5*( -z_interplanar_spacing_crys1
                                           + z_offset_crys1 + z_offset_crys2 )
                                    )
         
      self.bf.icomment('The initial z-locations of the two boundaries at the '+
                    'point of creation of the bicrystal system are '+
                    '{0[0]} and {0[1]}'.format(self.boundary_locs))
      
      
   #################################################################
   
   
   def construct_system_z_free_surface(self, crys1_dims, crys2_dims, 
                        origin_crys1 = None, origin_crys2 = None,
                        hop_stacking_crys1 = 0, hop_stacking_crys2 = 0,
                        folder = './', log_filename = 'bicrys', wd=None):
      
      """
      
      crys1_dims are defined w.r.t. crys1 lattice spacings
      crys2_dims are defined w.r.t. crys2 lattice spacings
            
      crys1_dims[-1] and crys2_dims[-2] are set to zero.
      And crys1_dims[-2] and crys2_dims[-1] are set to:
      crys1_dims[-2] = - ( crys1_dims[-1] - crys1_dims[-2] ) and
      crys2_dims[-1] = crys2_dims[-1] - crys2_dims[-2]
      
      There is just one grain boundary which will be as follows,
      
   
        ------------------------------------- free surface at z
                 Crystal 2
      
              ************************ z=0
                 Crystal 1
      
        ------------------------------------- free surface at z
      
      
      """
      
      if not np.allclose(crys1_dims[:4], crys1_dims[:4].astype(int)):
         raise ValueError('x and y box dimensions of crystal 1 must be provided '+
                          'in integer multiples of lattice spacings of crystal 1')
         
      if not np.allclose(crys2_dims[:4], crys2_dims[:4].astype(int)):
         raise ValueError('x and y box dimensions of crystal 2 must be provided '+
                          'in integer multiples of lattice spacings of crystal 2')
      
      # Rescaling crys1_dims and crys2_dims along z direction
      crys1_dims[-2] = - ( crys1_dims[-1] - crys1_dims[-2] )
      crys2_dims[-1] = crys2_dims[-1] - crys2_dims[-2]
      crys1_dims[-1] = 0; crys2_dims[-2] = 0
      
      crys1_dims = crys1_dims.astype(float); crys2_dims = crys2_dims.astype(float)
      # The box dimension in z direction must be integer multiple of lattice spacings
      # plus one interplanar spacing
      #crys1_dims[-2] = crys1_dims[-2] - (1.0/self.db.multiplicity_crys1[2])
      #crys2_dims[-1] = crys2_dims[-1] + (1.0/self.db.multiplicity_crys2[2])
      
      
      # (crys1_dims[1]-crys1_dims[0]):(crys2_dims[1]-crys2_dims[0]) must be in
      # x_spacing_ratio
      if not ( ( np.allclose(np.array([crys1_dims[0], crys2_dims[0]]), 0.0) or
                 all( misc_crys.get_miller_indices(abs(np.array([crys1_dims[0], crys2_dims[0]]))) 
                      == self.x_spacing_ratio ) ) and 
               ( np.allclose(np.array([crys1_dims[1], crys2_dims[1]]), 0.0) or
                 all( misc_crys.get_miller_indices(abs(np.array([crys1_dims[1], crys2_dims[1]]))) 
                      == self.x_spacing_ratio ) ) ):
         raise ValueError('The x box lengths entered for crystal 1 and 2 should be '+
                          'ratio {0[0]}:{0[1]}'.format(self.x_spacing_ratio))
         
      # (crys1_dims[3]-crys1_dims[2]):(crys2_dims[3]-crys2_dims[2]) must be in
      # y_spacing_ratio
      if not ( ( np.allclose(np.array([crys1_dims[2], crys2_dims[2]]), 0.0) or
                 all( misc_crys.get_miller_indices(abs(np.array([crys1_dims[2], crys2_dims[2]]))) 
                      == self.y_spacing_ratio ) ) and 
               ( np.allclose(np.array([crys1_dims[3], crys2_dims[3]]), 0.0) or
                 all( misc_crys.get_miller_indices(abs(np.array([crys1_dims[3], crys2_dims[3]]))) 
                      == self.y_spacing_ratio ) ) ):
         raise ValueError('The y box lengths entered for crystal 1 and 2 should be '+
                          'ratio {0[0]}:{0[1]}'.format(self.y_spacing_ratio))
         
      LAMMPS_boundary_cond = ['p', 'p', 's']                  
      
      self.initiate_LAMMPS_system(log_filename=log_filename, folder=folder,
                                  wd=wd)
      self.bf.assign_boundary_condition(LAMMPS_boundary_cond)
      
      if origin_crys1 is None:
         origin_crys1 = 0.5/self.multiplicity_crys1
      else:
         if not isinstance(origin_crys1, np.ndarray):
            raise ValueError('"origin_crys1" must be a numpy array')
         elif np.shape(origin_crys1) != (3,):
            raise ValueError('"origin_crys1" must have shape (3,)')
         elif not np.all(np.r_[origin_crys1>=0, origin_crys1<1]):
            raise ValueError('"origin_crys1" must have values 0<= x,y,z < 1')
         else:
            pass
      
      self.define_crystal_structure(self.lat_name, self.lat_scale, 
                                    keyword_value_pairs={
                                                'origin': origin_crys1,
                                                'orient x': self.orient_crys1[0],
                                                'orient y': self.orient_crys1[1],
                                                'orient z': self.orient_crys1[2],
                                                'spacing': self.spacing_crys1
                                                })
         
      crys1_dims = ( (crys1_dims*np.repeat(self.spacing_crys1, 2)*self.lat_scale) + 
                      (np.array([-1, 1]*3)*1e-10) )
      crys2_dims = ( (crys2_dims*np.repeat(self.spacing_crys2, 2)*self.lat_scale) + 
                      (np.array([-1, 1]*3)*1e-10) )
      if not np.allclose(crys1_dims[:-2], crys2_dims[:-2]) :
         raise RuntimeError('The x and y dimensions of the two crystals do not match')
      else:
         system_dims = np.r_[crys1_dims[:-2], crys1_dims[-2], crys2_dims[-1]]
         
      self.bf.icommand('region full_box block {0} units box'.format(' '.join(map(str, system_dims))) )
      self.bf.icommand('create_box {0} full_box'.format(self.n_atom_types))
      
      self.bf.icommand('region crys1_region block INF INF INF INF INF {0} units box'.format(1e-10))
      self.bf.icommand('create_atoms 1 region crys1_region')
      self.bf.icommand('group crys1_group region crys1_region')
      self.n_atoms_crys1 = self.bf.lmp.get_natoms()
      self.bf.icomment('There are {0} atoms in crystal 1'.format(self.n_atoms_crys1))
      self.atom_indices_crys1 = np.arange(self.n_atoms_crys1)
      
      if origin_crys2 is None:
         origin_crys2 = 0.5/self.multiplicity_crys2
      else:
         if not isinstance(origin_crys2, np.ndarray):
            raise ValueError('"origin_crys2" must be a numpy array')
         elif np.shape(origin_crys2) != (3,):
            raise ValueError('"origin_crys2" must have shape (3,)')
         elif not np.all(np.r_[origin_crys2>=0, origin_crys2<1]):
            raise ValueError('"origin_crys2" must have values 0<= x,y,z < 1')
         else:
            pass
      
      self.define_crystal_structure(self.lat_name, self.lat_scale, 
                                    keyword_value_pairs={
                                                'origin': origin_crys2,
                                                'orient x': self.orient_crys2[0],
                                                'orient y': self.orient_crys2[1],
                                                'orient z': self.orient_crys2[2],
                                                'spacing': self.spacing_crys2
                                                })
                  
      self.bf.icommand('region crys2_region block INF INF INF INF -{0} INF units box'.format(1e-10))
      self.bf.icommand('create_atoms 1 region crys2_region')
      self.bf.icommand('group crys2_group subtract all crys1_group')
      self.n_atoms = self.bf.lmp.get_natoms()
      self.n_atoms_crys2 = self.n_atoms - self.n_atoms_crys1
      self.bf.icomment('There are {0} atoms in crystal 2'.format(self.n_atoms_crys2))
      self.atom_indices_crys2 = np.arange(self.n_atoms_crys1, self.n_atoms)
      
      # Displace crys1 and crys2 parallel to the grain boundary by some integer 
      # multiple of the relative displacement of two adjacent planes along the 
      # grain boundary normal
      rel_stack_disp_crys1 = self.crys_struc.relativeDisplacementsLatticePlanes(
                                    self.z_dir_crys1)[0][0]
      rel_stack_disp_crys2 = self.crys_struc.relativeDisplacementsLatticePlanes(
                                    self.z_dir_crys2)[0][0]
      
      if not isinstance(hop_stacking_crys1, int):
         raise ValueError('"hop_stacking_crys1" must be an integer')
      else:
         hop_disp_crys1 = hop_stacking_crys1*rel_stack_disp_crys1*self.lat_scale
         
      hop_disp_crys1 = np.linalg.solve(self.orient_crys1.T, hop_disp_crys1)
      
      if not np.isclose(hop_disp_crys1[-1], 0.0):
         raise RuntimeError('"hop_disp_crys1" has a z-component')
      else:
         self.bf.displace_atoms(hop_disp_crys1, group='crys1_group')
      
      if not isinstance(hop_stacking_crys1, int):
         raise ValueError('"hop_stacking_crys2" must be an integer')
      else:
         hop_disp_crys2 = hop_stacking_crys2*rel_stack_disp_crys2*self.lat_scale
         
      hop_disp_crys2 = np.linalg.solve(self.orient_crys2.T, hop_disp_crys2)
      
      if not np.isclose(hop_disp_crys2[-1], 0.0):
         raise RuntimeError('"hop_disp_crys2" has a z-component')
      else:
         self.bf.displace_atoms(hop_disp_crys2, group='crys2_group')
      
      
      # Determining the location of the grain boundary
      z_interplanar_spacing_crys1 = 1.0/self.multiplicity_crys1[2]
      z_offset_crys1 = (origin_crys1[2]+1e-10) % z_interplanar_spacing_crys1
      z_offset_crys1 = z_offset_crys1*self.lat_scale*self.spacing_crys1[2]
      z_interplanar_spacing_crys1 = ( z_interplanar_spacing_crys1*self.lat_scale
                                     *self.spacing_crys1[2] )
      
      z_interplanar_spacing_crys2 = 1.0/self.multiplicity_crys2[2]
      z_offset_crys2 = (origin_crys2[2]+1e-10) % z_interplanar_spacing_crys2
      z_offset_crys2 = z_offset_crys2*self.lat_scale*self.spacing_crys2[2]
      z_interplanar_spacing_crys2 = ( z_interplanar_spacing_crys2*self.lat_scale
                                     *self.spacing_crys2[2] )
      
      if np.isclose(z_offset_crys1, 0.0):
         self.boundary_locs = (0.5*z_offset_crys2)
      else:
         self.boundary_locs = ( 0.5*( -z_interplanar_spacing_crys1
                                    + z_offset_crys1 + z_offset_crys2 )
                              )
         
      self.bf.icomment('The initial z-location of the boundary '+
                    'at the point of creation of the bicrystal system is '+
                    '{0}'.format(self.boundary_locs))
      
      
#################################################################

   def __setattr__(self, name, value):
      System.__setattr__(self, name, value)
      self.__dict__[name] = value
      
#################################################################




#def const_bicrys_method1(self, full_dims, crys1_dims1, crys1_dims2, crys2_dims, 
#                            pbc_x_y, outfile):
#   
#      # full_dims lengths are defined w.r.t. crys1 lattice spacings
#      # crys1_dims1 lengths are defined w.r.t. crys1 lattice spacings
#      # crys1_dims2 lengths are defined w.r.t. crys1 lattice spacings
#      # crys2_dims lengths are defined w.r.t. crys2 lattice spacings
#      
#      lmp = lammps(cmdargs=['-log', 'bicrys.log'])
#      lmp.command('units metal')
#      lmp.command('boundary {0[0]} {0[1]} p'.format(pbc_x_y))
#      lmp.command('atom_modify map array')
#      
#      lmp.command('lattice {0} {1} origin {2[0]} {2[1]} {2[2]} '.format(self.lat_name, self.lat_param, self.origin1) + 
#                  'orient x {0[0]} {0[1]} {0[2]} '.format(self.orient1[0]) +
#                  'orient y {0[0]} {0[1]} {0[2]} '.format(self.orient1[1]) +
#                  'orient z {0[0]} {0[1]} {0[2]} '.format(self.orient1[2]) +
#                  'spacing {0[0]} {0[1]} {0[2]}'.format(self.spacings1) )
#                  
#      lmp.command('region full_box block {0} units lattice'.format(' '.join(map(str, full_dims))) )
#      lmp.command('create_box 2 full_box')
#      
#      lmp.command('region crys1_region1 block {0} units lattice'.format(' '.join(map(str, crys1_dims1))))
#      lmp.command('create_atoms 1 region crys1_region1')
#      
#      lmp.command('region crys1_region2 block {0} units lattice'.format(' '.join(map(str, crys1_dims2))))
#      lmp.command('create_atoms 1 region crys1_region2')
#      
#      lmp.command('lattice {0} {1} origin {2[0]} {2[1]} {2[2]} '.format(self.lat_name, self.lat_param, self.origin2) + 
#                  'orient x {0[0]} {0[1]} {0[2]} '.format(self.orient2[0]) +
#                  'orient y {0[0]} {0[1]} {0[2]} '.format(self.orient2[1]) +
#                  'orient z {0[0]} {0[1]} {0[2]} '.format(self.orient2[2]) +
#                  'spacing {0[0]} {0[1]} {0[2]}'.format(self.spacings2) )
#                  
#      lmp.command('region crys2_region block {0} units lattice'.format(' '.join(map(str, crys2_dims))))
#      lmp.command('create_atoms 2 region crys2_region')
#      
#      
#      lmp.command('mass * 1.0')
#      
#      lmp.command('write_data {0}'.format(outfile))
      
     
      
#################################################################
