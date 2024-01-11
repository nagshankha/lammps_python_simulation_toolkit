import numpy as np
from buffer import Buffer
from system import System
import gzip
import io

class LoadSystem(System):
   
   def __init__(self, bf:Buffer, file_path:str, n_atom_types:int):
      
      System.__init__(self, bf, n_atom_types)     
      self.file_path = file_path
      
###################################################################
      
   def read_from_data_or_restart(self, file_type, 
                                 LAMMPS_boundary_cond=None):
      
      self.bf.icomment('Reading system state from {0} file'.format(file_type))
      
      if file_type == 'data':
         if LAMMPS_boundary_cond is None:
            raise RuntimeError('Boundary condition needs to be specified while '+
                               'reading data file')
         else:
            self.bf.assign_boundary_condition(LAMMPS_boundary_cond)
         self.bf.icommand('read_data {0}'.format(self.file_path))
      elif file_type == 'restart':
         self.bf.icommand('read_restart {0}'.format(self.file_path))
         if LAMMPS_boundary_cond is None:
            pass
         else:
            self.bf.reassign_boundary_condition(LAMMPS_boundary_cond)
      else:
         raise ValueError('For this method, the input file type must be '+
                          'either "data" or "restart"')
         
      self.n_atoms = self.bf.lmp.get_natoms()
      
      self.bf.icomment('{0} atoms are created'.format(self.n_atoms))
      
      self.bf.iwhitespace()
         
      
         
###################################################################
         
   def read_from_dump_direct(self, fields:list, keyword_dict:dict={}, 
                      system_dims:np.ndarray=None, 
                      LAMMPS_boundary_cond:list=['p', 'p', 'p'], 
                      Nstep:int=None, pad:float=2.0):
      
      if self.lmp.get_natoms() != 0:
         raise RuntimeError('This functions create a new atomistic system '+
                            'by reading data from a dump file and is NOT '+
                            'meant for adding atoms to an existing system')     
         
      if not isinstance(keyword_dict, dict):
         raise ValueError('"keyword_dict" must be a dictionary')
      else:
         keyword_dict.update({'add': 'keep', 'box': 'yes'})
         
      self.bf.icomment('Reading system state from dump file')
         
      if Nstep is None:
         if self.file_path.strip()[-3:] == '.gz':
            f = gzip.open(self.file_path, 'r')
         else:
            f = open(self.file_path, 'r')
         f.readline()
         Nstep = int(f.readline().strip())
         f.close()
      else:
         if not isinstance(Nstep, int):
            raise ValueError('"Nstep" must be an integer')
         elif self.n_atom_types < 0:
            raise ValueError('"Nstep" must be a non-negative integer')
      
      if system_dims is None:
         if self.file_path.strip()[-3:] == '.gz':
            f = gzip.open(self.file_path, 'r')
            with io.TextIOWrapper(io.BufferedReader(f)) as file:
               n_atoms, LAMMPS_boundary_cond, system_dims = ( 
                  scan_over_dumpfile_for_system_dimensions(
                                       file, Nstep, pad) )
            file.close()
         else:
            f = open(self.file_path, 'r')
            n_atoms, LAMMPS_boundary_cond, system_dims = ( 
               scan_over_dumpfile_for_system_dimensions(
                                       f, Nstep, pad) )
         f.close()        
      else:
         if not isinstance(system_dims, np.ndarray):
            raise ValueError('"system_dims" must be a numpy array')
         elif len(np.shape(system_dims)) != 1:
            raise ValueError('"system_dims" must be a 1D array')
         elif len(system_dims) != 6:
            raise ValueError('"system_dims" must have 6 entries: '+
                             '[xlo xhi ylo yhi zlo zhi]')
         elif not system_dims.dtype in [int, float]:
            raise ValueError('"system_dims" must be either integer or '+
                             'float array')
         elif not np.all(np.diff(system_dims)[::2] >= 0):
            raise ValueError('"system_dims" entries must be ordered as '+
                             '[xlo xhi ylo yhi zlo zhi]')    
                             
      self.bf.comm.Barrier()  
      
      self.bf.assign_boundary_condition(LAMMPS_boundary_cond)
         
      self.bf.icommand('region simul_cell block {0} units box'.format(
                                       ' '.join(map(str, system_dims))) )
      self.bf.icommand('create_box {0} simul_cell'.format(self.n_atom_types))
         
      read_dump_str = 'read_dump {0} {1} '.format(self.file_path, Nstep)
      read_dump_str += ' '.join(fields) + ' '
      read_dump_str += ' '.join(['{0} {1}'.format(key, keyword_dict[key]) 
                                 for key in keyword_dict.keys()])
   
      self.bf.icommand(read_dump_str)
      
      self.n_atoms = self.lmp.get_natoms()
      
      if self.n_atoms != n_atoms:
         self.bf.icomment('Number of atoms in the dump file = {0}'.format(n_atoms),
                       print_to_input_script=False,
                       print_to_logfile=True)
         self.bf.icomment('Number of atoms in the system after reading the dump file = {0}'.format(
                       self.n_atoms), print_to_input_script=False,
                       print_to_logfile=True)
         raise RuntimeError('Some atoms are lost while reading the dump file')
      else:
         self.bf.icomment('{0} atoms are created'.format(self.n_atoms))
         
      self.bf.iwhitespace()

################################################################### 

   def read_dump_after_creating_tmp_datafile(self, get_images = True, 
                      get_ids = True, get_types = True, atom_types = 1,
                      get_velocities = False, get_forces = False,
                      tmp_file_dir = './', tmp_filename = 'tmp.data',
                      keep_tmp_file = False, tmp_file_desc = None, **kwargs):
      
      if self.lmp.get_natoms() != 0:
         raise RuntimeError('This functions create a new atomistic system '+
                            'by reading data from a dump file and is NOT '+
                            'meant for adding atoms to an existing system')
         
      self.bf.icomment('Reading system state from dump file by first writing '+
                       'a temporary data files which is then read. This help '+
                       'in avoiding the problem of having different forces '+
                       'on atoms by directly reading with read_dump')
      
      import io_files
      
      desired_fields = ['Position']
      if get_images:
         desired_fields = desired_fields + ['Image']
      
      if get_ids:
         desired_fields = desired_fields + ['id']
         
      if get_types:
         desired_fields = desired_fields + ['type']
      
      if get_velocities:
         desired_fields = desired_fields + ['Velocity']
      
      if get_forces:
         desired_fields = desired_fields + ['Force']
         
      if 'nonstandard_field_names' in kwargs:      
         step, n_atoms, LAMMPS_boundary_cond, xlims, ylims, zlims, tilt_values, dump_data = ( 
            io_files.read_data_from_LAMMPS_dump_file(self.file_path,
                              desired_fields=desired_fields,
                              nonstandard_field_names=kwargs[
                                 'nonstandard_field_names']) )
      else:
         step, n_atoms, LAMMPS_boundary_cond, xlims, ylims, zlims, tilt_values, dump_data = ( 
            io_files.read_data_from_LAMMPS_dump_file(self.file_path,
                              desired_fields=desired_fields) )
         
      if tilt_values is not None:
         tilt_values = np.array(tilt_values)
         
      if isinstance(tmp_file_dir, str):
         tmp_file_dir = tmp_file_dir.strip()
         from toolbox import Toolbox
         Toolbox.mkdir(self.bf, tmp_file_dir)
         if tmp_file_dir[-1] != '/':
            tmp_file_dir = tmp_file_dir + '/'
      else:
         raise ValueError('"tmp_file_dir" must be a string')
         
      if isinstance(tmp_filename, str):   
         tmp_filepath = tmp_file_dir+tmp_filename
      else:
         raise ValueError('"tmp_filename" must be a string')     
         
      if self.bf.comm.rank == 0:
         io_files.write_lammps_data_file(tmp_filepath, self.n_atom_types, 
                              atom_types=dump_data['ParticleType'] if get_types else atom_types, 
                              xyz = np.c_[dump_data['Position.x'], dump_data['Position.y'],
                                          dump_data['Position.z']], 
                              box_dims = np.array(xlims+ylims+zlims), 
                              image_flags = np.c_[dump_data['Image.x'], dump_data['Image.y'],
                                          dump_data['Image.z']] if get_images else None, 
                              ids = dump_data['ParticleIdentifier'] if get_ids else None, 
                              velocities = np.c_[dump_data['Velocity.x'], dump_data['Velocity.y'],
                                          dump_data['Velocity.z']] if get_velocities else None, 
                              tilt_values = tilt_values, file_description = tmp_file_desc)
         
      self.bf.comm.Barrier()
      
      ls = LoadSystem(self.bf, tmp_filepath, self.n_atom_types)
      ls.read_from_data_or_restart('data', LAMMPS_boundary_cond=LAMMPS_boundary_cond)
      
      self.n_atoms = self.lmp.get_natoms()
      
      if self.n_atoms != n_atoms:
         self.bf.icomment('Number of atoms in the dump file = {0}'.format(n_atoms),
                       print_to_input_script=False,
                       print_to_logfile=True)
         self.bf.icomment('Number of atoms in the system after reading the dump '+
                          'file via a tempory data file = {0}'.format(
                       self.n_atoms), print_to_input_script=False,
                       print_to_logfile=True)
         raise RuntimeError('Some atoms are lost while reading the dump file')
      else:
         pass
         #self.bf.icomment('{0} atoms are created'.format(self.n_atoms))
         
      self.bf.iwhitespace()
      
      if get_forces:
         self.bf.set_peratom_attr('f', np.c_[dump_data['Force.x'], dump_data['Force.y'],
                                             dump_data['Force.z']])
         self.bf.iwhitespace()
         
      if not keep_tmp_file:
         import os
         if self.comm.rank == 0:
            os.remove(tmp_filepath)
         self.comm.Barrier()
      
      
################################################################### 

   def __setattr__(self, name, value):
      
      System.__setattr__(self, name, value)
      
      if name == 'file_path':
         if isinstance(value, str): 
            self.__dict__[name] = value
         else:
            raise ValueError('Member variable "file_path", which is the path to '+
                             'the restart/dump/data file, must be a string')
            
      else:
         pass 

###############################################################################
######################### End of class LoadSystem #############################
###############################################################################

def scan_over_dumpfile_for_system_dimensions(f, Nstep, pad):

   # f is the file object
   for line in f:
      if line.strip() == 'ITEM: TIMESTEP':
         if int(f.readline().strip()) == Nstep:
            if not (f.readline().strip() == 'ITEM: NUMBER OF ATOMS'):
               raise RuntimeError('"ITEM: NUMBER OF ATOMS" is the next item '+ 
                                  'expected after "ITEM: TIMESTEP" '+
                                  'in the dump file')
            else:
               n_atoms = int(f.readline().strip())
            
            box_str = f.readline().strip()
            if not (box_str[:-9] == 'ITEM: BOX BOUNDS'):
               raise RuntimeError('"ITEM: BOX BOUNDS" is the next item '+ 
                                  'expected after "ITEM: NUMBER OF ATOMS" '+
                                  'in the dump file')
            else:
               bc = box_str[-8:]
               xlo, xhi = map(float, f.readline().strip().split())
               ylo, yhi = map(float, f.readline().strip().split())
               zlo, zhi = map(float, f.readline().strip().split())
               system_dims = ( np.array([xlo, xhi, ylo, yhi, zlo, zhi]) +
                               (pad*np.array([-1,1]*3)) )
            break
         else:
            continue  
         
   return n_atoms, bc.split(), system_dims


###################################################################  



      
      
      
      
