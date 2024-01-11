from write_lammps_input_script import WriteInputScript
import os
import numpy as np
import io_files
from ctypes import c_double, c_int
import warnings

class LAMMPS_Toolbox(WriteInputScript):
   
   def __init__(self):
      raise RuntimeError('class "LAMMPS_Toolbox" must not be initialised, '+
                         'ONLY inherited')
      
   ############################################################################
      
   def define_crystal_structure_in_LAMMPS(self, style, scale, 
                                          keyword_value_pairs={}):
      
      if style not in ['none', 'sc', 'bcc', 'fcc', 'hcp', 'diamond', 'sq', 'sq2',
                       'hex', 'custom']:
         raise ValueError('{0} is not a valid lattice style. '.format(style)+
                          'Check the lattice command in the LAMMPS documentation')
         
      if not isinstance(scale, float):
         raise ValueError('scale must be of type float')
      elif scale <= 0.0:
         raise ValueError('scale must be positive')
      else:
         pass
      
      if not isinstance(keyword_value_pairs, dict):
         raise ValueError('keyword_value_pairs must be a dictionary. The keywords '+
                          'corresponding values must comply to the lattice '+
                          'command syntax in LAMMPS documentation')
      elif not all([isinstance(x, str) for x in keyword_value_pairs]):
         raise ValueError('keys in keyword_value_pairs must be strings')
      elif not all([isinstance(x, (list, np.ndarray)) for x in 
                                  keyword_value_pairs.values()]):
         raise ValueError('values in keyword_value_pairs must be lists')
         
      self.icommand('lattice {0} {1} {2}'.format(style, scale,
                    ' '.join([x+' '+' '.join(map(str, keyword_value_pairs[x]))
                              for x in keyword_value_pairs]) ))
      
   ############################################################################         
         
   def fill_in_atoms(self, style='box', args='', atom_type=1, keyword_value_pairs={}):
      
      if style == 'box':
         if args != '':
            raise ValueError('No arguments for create_atom style box, therefore '+
                             'args must be an empty string')
      elif style == 'region':
         if not isinstance(args, str):
            raise ValueError('Argument for create_atom style region must be a '+
                             'regionID, a string')
      elif style == 'single':
         if hasattr(args, '__iter__') and not isinstance(args, dict):
            if all([isinstance(x, float) for x in args]) and (len(args) == 3):
               args = ' '.join(map(str, args))
            else:
               raise ValueError('args must be the coordinates of the single '+
                                'particle. So it must be an iterable with 3 floats.')
         else:
            raise  ValueError('args must be an iterable with 3 floats.')
            
      elif style == 'random':
         if not isinstance(args, list):
            raise ValueError('args must be a list')
         if not (isinstance(args[0], int) and args[0] > 0):
            raise ValueError('The first entry in args is the number of particles '+
                             'to create, therefore it must be a positive integer.')
         elif not (isinstance(args[1], int) and args[1] > 0):
            raise ValueError('The second entry in args is the random # seed, '+
                             'therefore it must be a positive integer.')
         elif not isinstance(args[2], str):
            raise ValueError('The third entry in args is a regionID, '+
                             'which must be a string.')
         else:
            args = ' '.join(map(str, args))

      elif style == 'pythonAPI':
         if not isinstance(args, dict):
            raise ValueError('Arguments for create_atom custom style "pythonAPI" '+
                             'must be a dictionary.')
            
         if 'atom_pos' not in args:
            raise ValueError('args must contain atom positions with key "atom_pos"')
         elif not isinstance(args['atom_pos'], np.ndarray):
            raise ValueError('args["atom_pos"] must be a numpy array')
         elif args['atom_pos'].dtype != float:
            raise ValueError('args["atom_pos"] must be of dtype float')
         else:
            pass
         
         if len(np.shape(args['atom_pos'])) == 2:
            if np.shape(args['atom_pos'])[1] == 3:
               args['atom_pos'] = args['atom_pos'].flatten()
            else:
               raise ValueError('args["atom_pos"] must have shape (n,3) or (3n,)')
               
         elif len(np.shape(args['atom_pos'])) == 1:
            if len(args['atom_pos'])%3 != 0 :
               raise ValueError('args["atom_pos"] must have shape (n,3) or (3n,)')
               
         else:
            raise ValueError('args["atom_pos"] must have shape (n,3) or (3n,)')
            
         n_atoms = len(args['atom_pos'])//3
            
         if 'atom_id' not in args:
            args['atom_id'] = None
         else:
            if args['atom_id'] is None:
               pass
            elif not isinstance(args['atom_id'], np.ndarray):
               raise ValueError('args["atom_id"] must be a numpy array')
            elif args['atom_id'].dtype != int:
               raise ValueError('args["atom_id"] must be of dtype integer')
            elif np.any(args['atom_id'] <=0):
               raise ValueError('All atom IDs must be positive')
            elif np.shape(args['atom_id']) != (n_atoms,):
               raise ValueError('args["atom_id"] must have shape (n,)')
            else:
               args['atom_id'] = list(args['atom_id'])
               
               
         if 'atom_type' not in args:
            raise ValueError('args must contain atom types with key "atom_type". '+
                             'It can either be a list or an integer.')
         elif isinstance(args['atom_type'], int):
            args['atom_type'] = [args['atom_type']]*n_atoms
         elif not isinstance(args['atom_type'], np.ndarray):
            raise ValueError('args["atom_type"] must be a numpy array, if not integer')
         elif not np.issubdtype(args['atom_type'].dtype, np.integer):
            raise ValueError('args["atom_type"] must be of dtype integer')
         elif np.any(args['atom_type'] <=0):
            raise ValueError('All atom types must be positive')
         elif np.shape(args['atom_type']) != (n_atoms,):
            raise ValueError('args["atom_type"] must have shape (n,)')
         else:
            args['atom_type'] = list(args['atom_type'])
            
            
         if 'atom_vel' not in args:
            args['atom_vel'] = None
         else:
            if args['atom_vel'] is None:
               pass
            elif not isinstance(args['atom_vel'], np.ndarray):
               raise ValueError('args["atom_vel"] must be a numpy array')
            elif args['atom_vel'].dtype != float:
               raise ValueError('args["atom_vel"] must be of dtype float')
            elif np.shape(args['atom_vel']) == (n_atoms, 3):
               args['atom_vel'] = args['atom_vel'].flatten()
            elif np.shape(args['atom_vel']) == (3*n_atoms):
               pass
            else:
               raise ValueError('args["atom_vel"] must have shape (n,3) or (3n,)')
               
         if 'image_flags' not in args:
            args['image_flags'] = None
         else:
            if args['image_flags'] is None:
               pass
            elif not isinstance(args['image_flags'], np.ndarray):
               raise ValueError('args["image_flags"] must be a numpy array')
            elif args['image_flags'].dtype != int:
               raise ValueError('args["image_flags"] must be of dtype integer')
            elif np.shape(args['image_flags']) == (n_atoms, 3):
               args['image_flags'] = [self.lmp.encode_image_flags(*map(int,f)) 
                                      for f in args['image_flags']]
            else:
               raise ValueError('args["image_flags"] must have shape (n,3)')
               
         if 'shrinkexceed' not in args:
            args['shrinkexceed'] = False
         elif not isinstance(args['shrinkexceed'], bool):
            raise ValueError('args["shrinkexceed"] must be a boolean')
            
         self.icomment('Creating {0} atoms using the create_atoms '.format(n_atoms)+
                       'method of the lammps class python API')   
         n = self.lmp.create_atoms(n_atoms, args['atom_id'], args['atom_type'],
                                   args['atom_pos'], v=args['atom_vel'],
                                   image=args['image_flags'], 
                                   shrinkexceed=args['shrinkexceed'])
         
         if n != n_atoms:
            raise RuntimeError('Number of created atoms are not same as the number '+
                               'of atoms that was intended to create. Maybe atoms '+
                               'got lost; so please check! Use shrinkexceed = True '+
                               'if necessary.')
         else:
            self.icomment('{0} atoms successfully created.'.format(n_atoms))
            self.iwhitespace()
            return
      
      else:
         raise ValueError('{0} is not a valid create_atom style. '.format(style)+
                          'Check the lattice command in the LAMMPS documentation. '+
                          'Note that style can also be "pythonAPI" when using the '+
                          'pythonAPI command create_atoms to create atoms.')
         
      if not isinstance(keyword_value_pairs, dict):
         raise ValueError('keyword_value_pairs must be a dictionary. The keywords '+
                          'corresponding values must comply to the create_atoms '+
                          'command syntax in LAMMPS documentation')
      elif not all([isinstance(x, str) for x in keyword_value_pairs]):
         raise ValueError('keys in keyword_value_pairs must be strings')
      elif not all([isinstance(x, (list,str)) for x in keyword_value_pairs.values()]):
         raise ValueError('values in keyword_value_pairs must be either lists '+
                          'or strings')
         
      for x in keyword_value_pairs:
         if isinstance(keyword_value_pairs[x], str):
            keyword_value_pairs[x] = [keyword_value_pairs[x]]
         
      self.icommand('create_atoms {0} {1} {2} {3}'.format(atom_type, style, args,
                    ' '.join([x+' '+' '.join(map(str, keyword_value_pairs[x]))
                              for x in keyword_value_pairs]) ))  
      self.icomment('{0} atoms successfully created.'.format(
                      self.lmp.get_natoms()))
      self.iwhitespace()
      
   ############################################################################
   
   def assign_boundary_condition(self, boundary_cond=['p', 'p', 'p']):
      
      import itertools
      
      opts = ['p', 's', 'f', 'm']
      boundary_cond_options = ( opts + [x*2 for x in opts] + 
                     [x[0]+x[1] for x in itertools.permutations(opts[1:], 2)] )
      
      for b in boundary_cond:
         if b not in boundary_cond_options:
            raise ValueError('Boundary condition in each dimension must be one '+
                             'of the following options: {0} \n'.format(
                                ' '.join(boundary_cond_options))+
                             'For more information, refer to the LAMMPS documentation '+
                             'on boundary command.')
            
      self.icommand('boundary {0}'.format(' '.join(boundary_cond)))
      
      
   ############################################################################
   
   def reassign_boundary_condition(self, boundary_cond=['p', 'p', 'p']):
      
      import itertools
      
      opts = ['p', 's', 'f', 'm']
      boundary_cond_options = ( opts + [x*2 for x in opts] + 
                     [x[0]+x[1] for x in itertools.permutations(opts[1:], 2)] )
      
      for b in boundary_cond:
         if b not in boundary_cond_options:
            raise ValueError('Boundary condition in each dimension must be one '+
                             'of the following options: {0} \n'.format(
                                ' '.join(boundary_cond_options))+
                             'For more information, refer to the LAMMPS documentation '+
                             'on boundary command.')
            
      self.icommand('change_box all boundary {0}'.format(' '.join(boundary_cond)))
      
      
   ############################################################################
   
   def mkdir_with_LAMMPS(self, path):
      
      if not hasattr(self, 'lmp'):
         raise RuntimeError('"self" must have the LAMMPS object "lmp" as an '+
                            'attribute')
      elif not isinstance(path, str):
         raise ValueError('"path" must be a string')
      else:
         self.icomment('Creating new directory', print_to_logfile=True)
         s = 'shell "/bin/sh -c \'mkdir -p {0}\'"'.format(path)
         if self.write_input_script:
            if self.append_to_input_file:
               self.lmp.command('print """{0}\n""" append {1} screen no'.format(
                                                       s, self.input_file_path))
            else:            
               self.lmp.command('print """{0}\n""" file {1} screen no'.format(
                                                       s, self.input_file_path))            
         self.lmp.command(s)
         
      self.comm.Barrier()
         
   ############################################################################
   
   def write_dump_lmp(self, group_ID, folder, filename, dump_args,
                      dump_modify_args={}):
      
      if not isinstance(folder, str):
         raise ValueError('"folder" must be a string')
         
      if not isinstance(filename, str):
         raise ValueError('Dump "filename" must be a string')
      
      if folder[-1] != '/':
         raise ValueError('"folder" should end with a "/"')
      if not os.path.isdir(folder):
         self.mkdir_with_LAMMPS(folder)
         
      if 'MPIIO' in self.lmp.installed_packages:
         if len(filename.split('.mpiio')) == 2:
            filepath = folder+filename
         else:
            raise ValueError('Since the package MPIIO is installed, the '+
                             'dump "filename" must contain one and only one '+
                             '".mpiio"')
      else:
         if len(filename.split('.mpiio')) == 2:
            raise ValueError('Since the package MPIIO is NOT installed, the '+
                             'dump "filename" must NOT contain any ".mpiio"')            
         else:
            filepath = folder+filename
      
      if not isinstance(dump_modify_args, dict):
         raise ValueError('"dump_modify_args" must be a dictionary')
      else:
         dump_modify_args0 = dump_modify_args
         dump_modify_args = {'pbc':'yes', 'format': 'float %20.15g'}
         dump_modify_args.update(dump_modify_args0)
         dump_modify_args.update({'first': 'yes', 'every': '10'})
      self.icomment('Writing the current state to a dump file', 
                    print_to_logfile=True)
      self.icommand('dump dump_tmp {0} custom/mpiio 1 {1} {2}'.format(
                                 group_ID, filepath, ' '.join(dump_args)))
      self.icommand('dump_modify dump_tmp {0}'.format(' '.join(sum(map(
                             lambda x: [str(x), str(dump_modify_args[x])], 
                             dump_modify_args), []))) )
      self.run0()
      self.icommand('undump dump_tmp')
      self.iwhitespace(1)
      
   ############################################################################
      
   def write_data_lmp(self, folder, filename, write_w_LAMMPS=False,
                      **kwargs):
      
      if not isinstance(folder, str):
         raise ValueError('"folder" must be a string')
         
      if not isinstance(filename, str):
         raise ValueError('Data "filename" must be a string')
      
      if folder[-1] != '/':
         raise ValueError('"folder" should end with a "/"')
      if not os.path.isdir(folder):
         self.mkdir_with_LAMMPS(folder) 
      filepath = folder+filename
      
      if write_w_LAMMPS:
         self.icomment('Writing the current state to a data file', 
                    print_to_logfile=True)
         self.icommand('write_data {0}'.format(filepath))
      else:
         self.icomment('Writing the current state to a data file using '+
                       'lammps PythonAPI', print_to_logfile=True)
         
         n_types = kwargs['n_types']
         xyz = self.gather_peratom_attr('x')
         box_dims = self.get_box_dims()
         
         if 'atom_types' in kwargs:
            atom_types = kwargs['atom_types']
         else:
            atom_types = self.gather_peratom_attr('type')
         
         if 'get_ids' in kwargs:
            if kwargs['get_ids']:
               atom_ids = self.gather_peratom_attr('id')
            else:
               atom_ids = None  
         else:
            atom_ids = None
         
         if 'get_images' in kwargs:
            if kwargs['get_images']:
               image_flags = self.gather_peratom_attr('image')
            else:
               image_flags = None
         else:
            image_flags = None
            
         if 'get_velocities' in kwargs:
            if kwargs['get_velocities']:
               velocities = self.gather_peratom_attr('v')
            else:
               velocities = None
         else:
            velocities = None
            
         if 'get_tilt_values' in kwargs:
            if kwargs['get_tilt_values']:
               tilt_values = self.get_box_tilts()
            else:
               tilt_values = None
         else:
            tilt_values = None
            
         if 'masses' in kwargs:
            masses = kwargs['masses']
         else:
            masses = None
            
         if 'file_description' in kwargs:
            file_description = kwargs['file_description']
         else:
            file_description = None
         
         self.comm.Barrier()
         if self.comm.rank == 0:
            io_files.write_lammps_data_file(filepath, n_types, atom_types, 
                           xyz, box_dims, image_flags = image_flags, 
                           ids = atom_ids, masses = masses, 
                           velocities = velocities, tilt_values = tilt_values,
                           file_description = file_description)
         self.comm.Barrier()
      self.iwhitespace(1)
      
      
            
   ############################################################################
      
   def write_restart_lmp(self, folder, filename):
      
      if not isinstance(folder, str):
         raise ValueError('"folder" must be a string')
         
      if not isinstance(filename, str):
         raise ValueError('Restart "filename" must be a string')
      
      if folder[-1] != '/':
         raise ValueError('"folder" should end with a "/"')
      if not os.path.isdir(folder):
         self.mkdir_with_LAMMPS(folder)  
      
      if 'MPIIO' in self.lmp.installed_packages:
         if len(filename.split('.mpiio')) == 2:
            filepath = folder+filename
         else:
            raise ValueError('Since the package MPIIO is installed, the '+
                             'restart "filename" must contain one and only one '+
                             '".mpiio"')
      else:
         if len(filename.split('.mpiio')) == 2:
            raise ValueError('Since the package MPIIO is NOT installed, the '+
                             'restart "filename" must NOT contain any ".mpiio"')            
         else:
            filepath = folder+filename
            
      self.icomment('Writing the current state to a restart file',
                    print_to_logfile=True)
      self.icommand('write_restart {0}'.format(filepath))
      self.iwhitespace(1)
      
   ############################################################################
   
   def write_regular_restart_files(self, folder, filename, N, filename2=None,
                                   proc_break_keyword = {}):
      
      if not isinstance(folder, str):
         raise ValueError('"folder" must be a string')
         
      if not isinstance(filename, str):
         raise ValueError('Restart "filename" must be a string')
      
      if folder[-1] != '/':
         raise ValueError('"folder" should end with a "/"')
      if not os.path.isdir(folder):
         self.mkdir_with_LAMMPS(folder)  
      
      if 'MPIIO' in self.lmp.installed_packages:
         if len(filename.split('.mpiio')) == 2:
            filepath = folder+filename
         else:
            raise ValueError('Since the package MPIIO is installed, the '+
                             'dump "filename" must contain one and only one '+
                             '".mpiio"')
      else:
         if len(filename.split('.mpiio')) == 2:
            raise ValueError('Since the package MPIIO is NOT installed, the '+
                             'dump "filename" must NOT contain any ".mpiio"')            
         else:
            filepath = folder+filename
      
      if filename2 is not None:
         if not isinstance(filename2, str):
            raise ValueError('"filename2" to toggle between two restart files '+
                             'must be a string')
         if 'MPIIO' in self.lmp.installed_packages:
            if len(filename2.split('.mpiio')) == 2:
               filepath2 = folder+filename2
            else:
               raise ValueError('Since the package MPIIO is installed, the '+
                                'dump "filename2" must contain one and only one '+
                                '".mpiio"')
         else:
            if len(filename2.split('.mpiio')) == 2:
               raise ValueError('Since the package MPIIO is NOT installed, the '+
                                'dump "filename2" must NOT contain any ".mpiio"')            
            else:
               filepath2 = folder+filename2
         
      if isinstance(N, int) and N > 0:
         restart_str = 'restart {0} {1} '.format(N, filepath)
      elif isinstance(N, str) and N.strip().isdigit() and N!='0':
         restart_str = 'restart {0} {1} '.format(N, filepath)
      elif isinstance(N, str) and N.strip()[:2]=='v_':
         restart_str = 'restart {0} {1} '.format(N, filepath)
      else:
         raise ValueError('N for the restart command must be a positive integer '+
                          'or an equal-style variable specified as v_name')
      self.icomment(('Setup for writing restart files at intervals of '+
                     str(N)+' timesteps'), print_to_logfile=True)
      if filename2 is not None:
         self.icomment('Writing restart file will toggle between two input '+
                       'filenames, {0} and {1}'.format(filename, filename2))
         restart_str += (filepath2+' ')
         
      if 'fileper' in proc_break_keyword or 'nfile' in proc_break_keyword:
         if len(filename.split('%')) != 2:
            raise RuntimeError('When fileper or nfile keyword is used, filename '+
                               'must have one and only one % character')
         if filename2 is not None:
            if len(filename2.split('%')) != 2:
               raise RuntimeError('When fileper or nfile keyword is used, filename2 '+
                                  'must have one and only one % character')
         
      if 'fileper' in proc_break_keyword and 'nfile' in proc_break_keyword:
         raise ValueError('One cannot use both the keywords "fileper" and "nfile"')
      elif 'fileper' in proc_break_keyword:
         Np = proc_break_keyword['fileper']
         if isinstance(Np, int) and Np > 0:
            self.icomment('One restart file will be written every {0} '.format(Np)+
                          'processor which will collect information from itself '+
                          'and the next {0}'.format(Np - 1) +
                          'processors and write it to a restart file.')
            restart_str += 'fileper {0}'.format(Np)
      elif 'nfile' in proc_break_keyword:
         Nf = proc_break_keyword['nfile']
         if isinstance(Nf, int) and Nf > 0:
            self.icomment('{0} restart files will be written. Each will collect '+
                          'information from itself and the next {0}'.format(
                             (self.comm.size//Nf) - 1) +
                          'processors and write it to a restart file.')
            restart_str += 'nfile {0}'.format(Nf)
      else:
         pass
            
      self.icommand(restart_str)
      self.iwhitespace(1)    
      
   ############################################################################
   
   def write_regular_dump_files(self, dumpID, groupID, N, folder, filename, 
                                dump_args, dump_modify_args={}):
      
      if not isinstance(folder, str):
         raise ValueError('"folder" must be a string')
         
      if not isinstance(filename, str):
         raise ValueError('Dump "filename" must be a string')
      
      if folder[-1] != '/':
         raise ValueError('"folder" should end with a "/"')
      if not os.path.isdir(folder):
         self.mkdir_with_LAMMPS(folder)
      
      if 'MPIIO' in self.lmp.installed_packages:
         if len(filename.split('.mpiio')) == 2:
            filepath = folder+filename
         else:
            raise ValueError('Since the package MPIIO is installed, the '+
                             'dump "filename" must contain one and only one '+
                             '".mpiio"')
      else:
         if len(filename.split('.mpiio')) == 2:
            raise ValueError('Since the package MPIIO is NOT installed, the '+
                             'dump "filename" must NOT contain any ".mpiio"')            
         else:
            filepath = folder+filename
      
      if not isinstance(dump_modify_args, dict):
         raise ValueError('"dump_modify_args" must be a dictionary')
      else:
         dump_modify_args0 = dump_modify_args
         dump_modify_args = {'pbc':'yes', 'format': 'float %15.10g'}
         dump_modify_args.update(dump_modify_args0)
         
      if not isinstance(dumpID, str):
         raise ValueError('dumpID must be a string')
         
      if not isinstance(groupID, str):
         raise ValueError('groupID must be a string')
         
      if isinstance(N, int) and N > 0:
         dump_str = 'dump {0} {1} custom/mpiio {2} {3} '.format(
                     dumpID, groupID, N, filepath)
      else:
         raise ValueError('N for the dump command must be a positive integer.')
         
      self.icomment(('Setup for writing dump files at intervals of '+
                     str(N)+' timesteps'), print_to_logfile=True)
      self.icommand(dump_str + ' '.join(dump_args))
      self.icommand('dump_modify {0} {1}'.format(dumpID, ' '.join(sum(map(
                             lambda x: [str(x), str(dump_modify_args[x])], 
                             dump_modify_args), []))) )
      self.iwhitespace(1)
   
   ############################################################################
   
   def turn_off_writing_restart_files(self):
      
      self.icomment('Pausing the writing of all restart files')
      self.icommand('restart 0')
      
   ############################################################################
   
   def turn_off_writing_dump_files(self, dumpID):
      
      if isinstance(dumpID, str):
         self.icomment('Pausing the writing of dump files corresponding to '+
                          'dumpID = {0}'.format(dumpID))
         self.icommand('undump {0}'.format(dumpID))
      else:
         raise ValueError('dumpID must be a string')
      
   ############################################################################
   
   def displace_atoms(self, disp, group=None, region=None, types=None, dims=None, 
                      atom_indices=None):
      
      if not isinstance(disp, (np.ndarray, list)):
         raise ValueError('"disp" must be a numpy array or a list')
      elif len(disp) != 3:
         raise ValueError('"disp" must have 3 values corresponding to the three '+
                          'displacement components')
      elif not np.all([isinstance(x, (int, float)) for x in disp]):
         raise ValueError('"disp" entries must be numeric, either integer or '+
                          'float')
      else:
         pass
   
      if np.count_nonzero([group is None, region is None, dims is None, 
                           types is None, atom_indices is None]) != 4:
         raise ValueError('You have to specify either group or region or types '+
                          'or some box dimensions if which atoms will be displaced')
      elif not group is None:
         if isinstance(group, str): 
            self.icommand('displace_atoms {0} move '.format(group)+
                          '{0[0]} {0[1]} {0[2]} units box'.format(disp)) 
         else:
            raise ValueError('"group" must be the name of a group in LAMMPS '+
                             'specified as str')
      elif not region is None:
         if isinstance(region, str): 
            group_name = region+'_grp'
            self.icommand('group {0} region {1}'.format(group_name, region))
            self.icommand('displace_atoms {0} move '.format(group_name)+
                          '{0[0]} {0[1]} {0[2]} units box'.format(disp))            
         else:
            raise ValueError('"region" must be the name of a region in LAMMPS '+
                             'specified as str')
      elif not types is None:
         import string
         if isinstance(types, list):
            group_name = 'types_'+'_'.join(map(str, types))+'_grp'
            self.lmp.command('group {0} type {1}'.format(group_name, ' '.join(types)))
            self.lmp.command('displace_atoms {0} move '.format(group_name)+
                             '{0[0]} {0[1]} {0[2]} units box'.format(disp))
         else:
            raise ValueError('"types" must be a list of atomtypes that you would '+
                             'like to displace')
      elif not dims is None:
         if not isinstance(dims, np.ndarray):
            raise ValueError('"dims" must a numpy array')
         elif dims.dtype != float:
            raise ValueError('"dims" must be a float array')
         elif np.shape(dims) != (6,):
            raise ValueError('"dims" must be [xlo xhi ylo yhi zlo zhi]')
         else: 
            random_str = ''.join(np.random.choice(list(string.ascii_lowercase), 5))
            group_name = random_str+'_grp'; region_name = random_str+'_reg'
            self.lmp.command('region {0} block {1} units box'.format(region_name, 
                                                   ' '.join(map(str, dims))))
            self.lmp.command('group {0} region {1}'.format(group_name, region_name))
            self.lmp.command('displace_atoms {0} move '.format(group_name)+
                             '{0[0]} {0[1]} {0[2]} units box'.format(disp))  
      else:
         if not isinstance(atom_indices, np.ndarray):
            raise ValueError('"atom_indices" must a numpy array')
         elif atom_indices.dtype != int:
            raise ValueError('"atom_indices" must be an integer array')
         elif len(np.shape(atom_indices)) != 1:
            raise ValueError('"atom_indices" must be an 1D array')
         else:
            pos = self.gather_peratom_attr('x')
            pos[atom_indices] = pos[atom_indices] + disp
            pos_ctypes = (3*self.lmp.get_natoms()*c_double)()
            pos_ctypes[:] = pos.flatten()
            self.lmp.scatter_atoms('x', 1, 3, pos_ctypes)
            
   ###########################################################################################
      
   def define_slice(self, dims, new_region_name=None,
                    new_group_name=None, group_type='static',
                    output_atom_indices_in_slice=False):
      
      if not isinstance(dims, (list, np.ndarray)):
         raise ValueError('"dims" must a list or a numpy array')
      elif np.shape(dims) != (6,):
         raise ValueError('"dims" must be [xlo xhi ylo yhi zlo zhi]')
      else:
         pass
      
      if group_type not in ['static', 'dynamic']:
         raise ValueError('"group_type" must be either "static" or "dynamic"')
      
      if new_group_name is None:
         pass
      else:
         if not isinstance(new_group_name, str):
            raise ValueError('if "new_group_name" is not None, then it must be '+
                             'a str with the desired name of the group')
         else:
            if isinstance(new_region_name, str):
               self.icommand('region {0} block {1} units box'.format(
                                 new_region_name, ' '.join(map(str, dims))))
               if group_type == 'static':                  
                  self.icommand('group {0} region {1}'.format(new_group_name,
                                                   new_region_name))
               else:
                  self.icommand('group {0} dynamic all region {1} every 1'.format(
                                     new_group_name, new_region_name))
            else:
               raise ValueError('"new_region_name" has to be provided as a '+
                                'string when argument "new_group_name" is a string')
      
      if isinstance(output_atom_indices_in_slice, bool):      
         if not output_atom_indices_in_slice:
            return
      else:
         raise ValueError('"output_atom_indices_in_slice" must be boolean.')
      
      dims_float = np.zeros(len(dims))      
      if any([x == 'INF' for x in dims]):
         box_dims = self.get_box_dims()
         for i,d in enumerate(dims):
            if (d=='INF') and (i%2==0):
               dims_float[i] = box_dims[i] - 1
            elif (d=='INF') and (i%2!=0):
               dims_float[i] = box_dims[i] + 1
            else:
               dims_float[i] = float(dims[i])   
      elif not ( (np.array(dims).dtype == float) or 
                 (np.array(dims).dtype == int) ):
         raise RuntimeError('Dimensions in "dims" must be either integer or '+
                            'float or "INF"')
      else:
         dims_float[:] = dims
            
      pos = self.gather_peratom_attr('x')
      sel_atoms_bool = np.all(np.c_[pos[:,0]>dims_float[0],
                                    pos[:,0]<dims_float[1],
                                    pos[:,1]>dims_float[2],
                                    pos[:,1]<dims_float[3],
                                    pos[:,2]>dims_float[4],
                                    pos[:,2]<dims_float[5],
                                   ], axis=1)
          
      return sel_atoms_bool
   
   ############################################################################
   
   def extract_regions(self, regionID: str or list, compressIDs='yes'):
      
      if isinstance(regionID, str):
         N_regions = 1
      elif isinstance(regionID, list):
         if all([isinstance(x, str) for x in regionID]):
            N_regions = len(regionID)
            region_str = ' '.join(regionID)
         else:
            raise ValueError('RegionIDs must be strings')
      else:
         raise ValueError('Argument "regionID" must be a string corresponding '+
                          'to one region that needs to be extracted or a list of '+
                          'strings corresponding to the union of regions that '+
                          'need to be extracted')
         
      if compressIDs not in ['yes', 'no']:
         raise ValueError('compressIDs must be either "yes" or "no". Refer to '+
                          'the compress keyword in the LAMMPS documentation for '+
                          'delete_atoms LAMMPS command.')
      
      if N_regions == 1:
         self.icommand('group atoms_to_keep region '+regionID)
         self.icommand('group atoms_to_delete subtract all atoms_to_keep')
      else:
         self.icommand('region region_to_keep union {0} {1}'.format( N_regions,
                                           region_str ))
         self.icommand('group atoms_to_keep region region_to_keep')
         self.icommand('group atoms_to_delete subtract all atoms_to_keep')
         self.icommand('region region_to_keep delete')
      
      self.icommand('delete_atoms group atoms_to_delete compress '+compressIDs)
      self.icommand('group atoms_to_keep delete')
      self.icommand('group atoms_to_delete delete')
   
   ############################################################################
   
   def gather_peratom_attr(self, info_type, atom_indices='all'):
      
      if not isinstance(info_type, str):
         raise ValueError('"info_type" must be str')
         
      if isinstance(atom_indices, str):
         if atom_indices=='all':
            atom_indices = np.arange(self.lmp.get_natoms())
         else:
            raise ValueError('The only string input for atom_indices is "all" '+
                             'when the user requests to consider all atoms in '+
                             'system for getattr')
      else:
         if not isinstance(atom_indices, np.ndarray):
            raise ValueError('"atom_indices" must be a np.ndarray if not "all"')
         elif atom_indices.dtype != int:
            raise ValueError('If np.ndarray, "atom_indices" must be of dtype int')
         elif len(np.shape(atom_indices)) != 1:
            raise ValueError('"atom_indices" must be a 1D array')
         elif len(atom_indices) > self.lmp.get_natoms():
            raise ValueError('"atom_indices" length must be less than or equal the number of atoms')
         elif len(np.unique(atom_indices)) != len(atom_indices):
            raise ValueError('"atom_indices" must be unique')
         
      possible_info_types_int_c1 = {'id', # LAMMPS atom ids 
                                    'type' # atom types
                                   }
      possible_info_types_double_c1 = {'mass', # LAMMPS atom ids 
                                       'pe' # atom potential energies (??)
                                      }
      possible_info_types_int_c3 = {'image' # Periodic image flags
                                   }
      possible_info_types_double_c3 = {'x', # atom positions
                                       'xu', # unwrapped atom positions 
                                       'v', # atom velocities
                                       'f' # atom forces
                                      } 
      
      if not info_type in set.union(possible_info_types_double_c1, 
                                    possible_info_types_double_c3,
                                    possible_info_types_int_c1,
                                    possible_info_types_int_c3):
         raise ValueError('Unknown "info_type" queried')
      elif info_type in possible_info_types_int_c1:
         data = np.array(self.lmp.gather_atoms(info_type, 0, 1))
      elif info_type in possible_info_types_int_c3:
         data = np.array(self.lmp.gather_atoms(info_type, 0, 3)).reshape(-1,3)
      elif info_type in possible_info_types_double_c3:
         if info_type == 'xu':
            atom_pos = self.gather_peratom_attr('x')
            image_flags = self.gather_peratom_attr('image')
            box_lengths = self.get_box_lengths()
            data = atom_pos + (image_flags*box_lengths)
         else:
            data = np.array(self.lmp.gather_atoms(info_type, 1, 3)).reshape(-1,3)
      elif info_type in possible_info_types_double_c1:
         data = np.array(self.lmp.gather_atoms(info_type, 1, 1))
         
      return data[atom_indices]
   
   ############################################################################   
   
   def set_peratom_attr(self, info_type, info_val, atom_indices='all'):
      
      if not isinstance(info_type, str):
         raise ValueError('"info_type" must be str')
         
      if not isinstance(info_val, np.ndarray):
         raise ValueError('"info_val" must be a np.ndarray')
         
      if isinstance(atom_indices, str):
         if atom_indices=='all':
            atom_indices = np.arange(self.lmp.get_natoms())
         else:
            raise ValueError('The only string input for atom_indices is "all" '+
                             'when the user requests to consider all atoms in '+
                             'system for setattr')
      else:
         if not isinstance(atom_indices, np.ndarray):
            raise ValueError('"atom_indices" must be a np.ndarray if not "all"')
         elif not np.issubdtype(atom_indices.dtype, np.integer):
            raise ValueError('If np.ndarray, "atom_indices" must be of dtype int')
         elif len(np.shape(atom_indices)) != 1:
            raise ValueError('"atom_indices" must be a 1D array')
         elif len(atom_indices) > self.lmp.get_natoms():
            raise ValueError('"atom_indices" length must be less than or equal the number of atoms')
         elif len(np.unique(atom_indices)) != len(atom_indices):
            raise ValueError('"atom_indices" must be unique')
            
      if len(info_val) != len(atom_indices):
         raise ValueError('"info_val" length must equal the length of atom indices')
         
      possible_info_types_int_c1 = {'id', # LAMMPS atom ids 
                                    'type' # atom types
                                   }
      possible_info_types_double_c1 = {'mass', # LAMMPS atom masses 
                                       'pe' # atom potential energies (??)
                                      }
      possible_info_types_double_c3 = {'x', # atom positions
                                       'v', # atom velocities
                                       'f' # atom forces
                                      } 
      possible_info_types_int_c3 = {'image' # Periodic image flags
                                   }
      
      if not info_type in set.union(possible_info_types_int_c1,
                                    possible_info_types_double_c1, 
                                    possible_info_types_double_c3,
                                    possible_info_types_int_c3):
         raise ValueError('"{0}" is an unknown peratom quantity'.format(info_type))
      elif info_type in possible_info_types_int_c1:  
         if info_type == 'type':
            self.icomment('Resetting atom types using function scatter_atom of '+
                          'LAMMPS python wrapper', print_to_logfile=True)
         elif info_type == 'id':
            self.icomment('Resetting atom IDs using function scatter_atom of '+
                          'LAMMPS python wrapper', print_to_logfile=True)
         if not np.issubdtype(info_val.dtype, np.integer):
            raise ValueError('atomtypes must be an integer array')
         elif len(np.shape(info_val)) != 1:
            raise ValueError('"atomtypes" must be a 1D array')
         data = self.gather_peratom_attr(info_type)
         data[atom_indices] = info_val
         data_ctypes = (self.lmp.get_natoms()*c_int)()
         data_ctypes[:] = data
         self.lmp.scatter_atoms(info_type, 0, 1, data_ctypes)
      elif info_type in possible_info_types_double_c3:
         if info_type == 'x':
            self.icomment('Resetting atom positions using function scatter_atom of '+
                          'LAMMPS python wrapper', print_to_logfile=True)
         elif info_type == 'v':
            self.icomment('Resetting atom velocities using function scatter_atom of '+
                          'LAMMPS python wrapper', print_to_logfile=True)
         elif info_type == 'f':
            self.icomment('Resetting forces on atoms using function scatter_atom of '+
                          'LAMMPS python wrapper', print_to_logfile=True)
         if not np.issubdtype(info_val.dtype, np.floating):
            raise ValueError('"info_val" must be an float array')
         elif len(np.shape(info_val)) != 2:
            raise ValueError('"info_val" must be a 2D array')
         elif np.shape(info_val)[1] != 3:
            raise ValueError('"info_val" must have three columns')
         data = self.gather_peratom_attr(info_type)
         data[atom_indices] = info_val
         data_ctypes = (3*self.lmp.get_natoms()*c_double)()
         data_ctypes[:] = data.flatten()
         self.lmp.scatter_atoms(info_type, 1, 3, data_ctypes)
      elif info_type in possible_info_types_double_c1:
         if not np.issubdtype(info_val.dtype, np.floating):
            raise ValueError('"info_val" must be an float array')
         elif len(np.shape(info_val)) != 1:
            raise ValueError('"info_val" must be a 1D array')
         data = self.gather_peratom_attr(info_type)
         data[atom_indices] = info_val
         data_ctypes = (self.lmp.get_natoms()*c_double)()
         data_ctypes[:] = data
         self.lmp.scatter_atoms(info_type, 1, 1, data_ctypes)
      elif info_type in possible_info_types_int_c3:  
         if info_type == 'image':
            self.icomment('Resetting atom images using function scatter_atom of '+
                          'LAMMPS python wrapper', print_to_logfile=True)
         if not np.issubdtype(info_val.dtype, np.integer):
            raise ValueError('"info_val" must be an integer array')
         elif len(np.shape(info_val)) != 2:
            raise ValueError('"info_val" must be a 2D array')
         elif np.shape(info_val)[1] != 3:
            raise ValueError('"info_val" must have three columns')
         data = self.gather_peratom_attr(info_type)
         data[atom_indices] = info_val
         data_ctypes = (3*self.lmp.get_natoms()*c_int)()
         data_ctypes[:] = data.flatten()
         self.lmp.scatter_atoms(info_type, 0, 3, data_ctypes)
         
         
      self.iwhitespace(1)
      
   ############################################################################
   
   def add_potential_eam(self, path2potfile, atomtype_names, 
                         style_appendage=None, accelerator_appendage=None,
                         i_j_str = '* *'):
                         
      if 'MANYBODY' not in self.lmp.installed_packages:
         raise RuntimeError('Package MANYBODY must be installed with '+
                            'lammps for using the EAM pair style')
   
      if not isinstance(path2potfile, str):
         raise ValueError('"path2potfile" must be a string')
         
      if not isinstance(i_j_str, str):
         raise ValueError('"i_j_str" must be a string')
         
      if not isinstance(atomtype_names, list):
         raise ValueError('"atomtype_names" must be a list')
      elif not np.all([isinstance(x, str) for x in atomtype_names]):
         raise ValueError('Entries in "atomtype_names" must be all strings')
      elif not np.all([len(x.strip().split(' '))==1 for x in atomtype_names]):
         raise ValueError('Entries in "atomtype_names" must be one word')
         
      accelerator_packages = ['GPU', 'USER-INTEL', 'KOKKOS', 'USER-OMP', 'OPT']
      accelerator_appendage_options = ['gpu', 'intel', 'kk', 'omp', 'opt']
      pair_style = 'eam'
      
      ##### FUNCTION: begin #####
      
      def append_accelerator_appendage(pair_style):
         if accelerator_appendage is not None:
            if accelerator_appendage in accelerator_appendage_options:
               if ( accelerator_packages[accelerator_appendage_options.index(
                            accelerator_appendage)] in self.lmp.installed_packages ):
                  return (pair_style + '/' + accelerator_appendage)
               else:
                  raise RuntimeError('Package {0} '.format(accelerator_packages[
                       accelerator_appendage_options.index(accelerator_appendage)])+
                                     'must be installed with lammps for using '+
                                     'the accelerator variant {0}'.format(
                                     accelerator_appendage))
            else:
               raise ValueError('EAM pair style accelerator_appendage must '+
                                'be one of the following: '+
                                ', '.join(accelerator_appendage_options))
         else:
            return pair_style
                                     
      ##### FUNCTION: end #####
      
      if style_appendage is None:
         pair_style = append_accelerator_appendage(pair_style)
      else:
         if style_appendage in ['alloy', 'fs']:
            pair_style += ('/'+style_appendage)
            pair_style = append_accelerator_appendage(pair_style)
         elif style_appendage in ['cd', 'cd/old', 'he']:
            pair_style += ('/'+style_appendage)
         else:
            raise ValueError('EAM style_appendage must be one of the following: '+
                             'alloy, cd, cd/old, fs, he')
         
      self.icomment('### Defining EAM potential parameters ###')
               
      self.icommand('pair_style {0}'.format(pair_style))
      self.icommand('pair_coeff {0} {1} {2}'.format(i_j_str, path2potfile, 
                                ' '.join(atomtype_names)))  
      
      for pot_path in ['./']+os.environ['LAMMPS_POTENTIALS'].split(':'):
         if os.path.exists(pot_path+'/'+path2potfile):
            path2potfile=pot_path+'/'+path2potfile
            break
      
      f = open(path2potfile, "rt")
      if style_appendage is None:
         for i, line in enumerate(f):
            if i == 2:
               self.rcut = float(line.split()[-1])
               break            
         self.icomment('Cutoff for current eam style potential is {0} A'.format(
                                                     self.rcut))
      else:
         for i, line in enumerate(f):
            if i == 4:
               if style_appendage == 'he':
                  self.rcut = float(line.split()[-2])
               else:
                  self.rcut = float(line.split()[-1])
               break            
         self.icomment('Cutoff for current eam/{1} style potential is {0} A'.format(
                                               self.rcut, style_appendage))
         
      f.close()
      
      self.iwhitespace()  
   
   ###########################################################################
   
   def add_potential_meamc(self, path2meamlibraryfile, path2meamfile,
                           atomtype_corr_meamfile_str, atomtype_names, 
                           i_j_str = '* *'):
                           
      if 'USER-MEAMC' not in self.lmp.installed_packages:
         raise RuntimeError('Package USER-MEAMC must be installed with '+
                            'lammps for using the MEAM pair style')
      
      if not isinstance(path2meamlibraryfile, str):
         raise ValueError('"path2meamlibraryfile" must be a string')
         
      if not isinstance(path2meamfile, str):
         raise ValueError('"path2meamfile" must be a string')
         
      if not isinstance(atomtype_corr_meamfile_str, str):
         raise ValueError('"atomtype_corr_meamfile_str" must be a string')
         
      if not isinstance(i_j_str, str):
         raise ValueError('"i_j_str" must be a string')
         
      if not isinstance(atomtype_names, list):
         raise ValueError('"atomtype_names" must be a list')
      elif not np.all([isinstance(x, str) for x in atomtype_names]):
         raise ValueError('Entries in "atomtype_names" must be all strings')
      elif not np.all([len(x.strip().split(' '))==1 for x in atomtype_names]):
         raise ValueError('Entries in "atomtype_names" must be one word')
      elif len(set(atomtype_names)) != len(atomtype_names):
         warnings.warn('For MEAM/c, it is recommended not to use repeated '+
                       'elements for the different atomtypes')
         
      self.icomment('### Defining MEAM potential parameters ###')         
      self.icommand('pair_style meam/c')
      self.icommand('pair_coeff {0} {1} {2} {3} {4}'.format(
                       i_j_str, path2meamlibraryfile, atomtype_corr_meamfile_str,
                       path2meamfile, ' '.join(atomtype_names))) 
      
      for pot_path in ['./']+os.environ['LAMMPS_POTENTIALS'].split(':'):
         if os.path.exists(pot_path+'/'+path2meamfile):
            path2meamfile=pot_path+'/'+path2meamfile
            break
      
      f = open(path2meamfile, "rt")
      for line in f:
         if line.split('=')[0].strip() == 'rc':
            self.rcut = float(line.split('=')[1].strip())
            break            
      self.icomment('Cutoff for current meam style potential is {0} A'.format(
                                                  self.rcut))
      
      self.iwhitespace()
      
   ###########################################################################
      
   def get_box_dims(self):
      
      return np.array([self.lmp.get_thermo('xlo'), self.lmp.get_thermo('xhi'),
                       self.lmp.get_thermo('ylo'), self.lmp.get_thermo('yhi'),
                       self.lmp.get_thermo('zlo'), self.lmp.get_thermo('zhi')])
                  
   ###########################################################################
            
   def get_box_tilts(self):
      
      return np.array([self.lmp.get_thermo('xy'), self.lmp.get_thermo('xz'),
                       self.lmp.get_thermo('yz')])
   
   ############################################################################
   
   def get_box_lengths(self):
      
      return np.diff(self.get_box_dims())[::2]
   
   ############################################################################
   
   def run0(self):
      
      self.icommand('run 0')
      
   ############################################################################
      
      
      
      
