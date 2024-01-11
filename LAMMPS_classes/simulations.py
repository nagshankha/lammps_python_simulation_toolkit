from buffer import Buffer
from typing import Iterable
from buffer_manager import BufferManager
import numpy as np

class Simulations(BufferManager):

   def __init__(self, bf:Buffer, thermo_list, thermo_freq=50, boxrelax_dirs=None, 
                couple='none'):
      
      self.addBuffer(bf) 
         
      periodicity = bf.lmp.extract_box()[5]       
      
      if boxrelax_dirs is None:
         self.boxrelax_dirs = None
         self.couple = 'none'
      else:   
         if not isinstance(boxrelax_dirs, list):
            raise ValueError('boxrelax_dirs must be a list')     
         elif 'x' in boxrelax_dirs and periodicity[0] == 0:
            raise RuntimeError('The simulation cell is not periodic in the x '+
                               'direction. So box/relax or npt cannot be used in '+
                               'this direction')
         elif 'y' in boxrelax_dirs and periodicity[1] == 0:
            raise RuntimeError('The simulation cell is not periodic in the y '+
                               'direction. So box/relax or npt cannot be used in '+
                               'this direction')
         elif 'z' in boxrelax_dirs and periodicity[2] == 0:
            raise RuntimeError('The simulation cell is not periodic in the z '+
                               'direction. So box/relax or npt cannot be used in '+
                               'this direction')
         else:
            self.boxrelax_dirs = boxrelax_dirs
            if couple not in ['none', 'xyz', 'xy', 'yz', 'xz']:
               raise ValueError('couple must be one of the following: {0}'.format(
                                ' '.join(['none', 'xyz', 'xy', 'yz', 'xz'])))
            else:
               self.couple = couple      
      
      self.update_thermo(thermo_list, thermo_freq)
      
      self.bf.icommand('run 0')
      
      self.bf.iwhitespace()
      
   #################################################################################
   
   def update_thermo(self, thermo_list=None, thermo_freq=None ):
      
      if thermo_freq is not None:
         self._thermo_freq = thermo_freq
         self.bf.icommand('thermo {0}'.format(self._thermo_freq))
      
      if thermo_list is not None:
         self._thermo_list = thermo_list
         self.bf.icommand('thermo_style custom {0}'.format(' '.join(self._thermo_list)))
      
         self.bf.icommand('thermo_modify flush yes')
         
         
   #################################################################################
   
   def add_items_to_thermo_list(self, thermo_items, loc=None):
      
      if not isinstance(thermo_items, list):
         raise ValueError('"thermo_items" must be a list')
      elif not all([isinstance(x, str) for x in thermo_items]):
         raise ValueError('If not None, "thermo_items" must be a list of strings')
      
      if loc is None:
         self.update_thermo(thermo_list=self._thermo_list+thermo_items)
      else:
         if not isinstance(loc, list):
            raise ValueError('If not None, "loc" must be a list')
         elif not all([isinstance(x, int) for x in loc]):
            raise ValueError('If not None, "loc" must be a list of integers')
         else:
            sort_inds = np.argsort(loc)
            loc = np.array(loc)[sort_inds]
            thermo_items = np.array(thermo_items)[sort_inds]
            u, cu = np.unique(loc, return_counts=True)
            ccu = np.cumsum(cu)
            thermo_items = [list(thermo_items[ccu[i-1]:ccu[i]]) if i!=0 
                            else list(thermo_items[:ccu[i]]) 
                            for i in range(len(ccu))]
         new_thermo_list = self._thermo_list
         for i, s in enumerate(thermo_items):
            new_thermo_list = ( new_thermo_list[:(i+u[i])]+[s]+
                                new_thermo_list[(i+u[i]):] )
         new_thermo_list2 = []
         for i in new_thermo_list:
             if isinstance(i, str):
                 new_thermo_list2 += [i]
             else:
                 new_thermo_list2 += i
         self.update_thermo(thermo_list=new_thermo_list2)

         
   #################################################################################
   
   def remove_items_from_thermo_list(self, thermo_items):
      
      if not isinstance(thermo_items, list):
         raise ValueError('"thermo_items" must be a list')
      elif not all([isinstance(x, str) for x in thermo_items]):
         raise ValueError('If not None, "thermo_items" must be a list of strings')
      
      new_thermo_list = [x for x in self._thermo_list if x not in thermo_items] 
      self.update_thermo(thermo_list=new_thermo_list)

         
   #################################################################################
   
   def reset_timestep(self, new_timestep=0, atime=None):
      
      if not (isinstance(new_timestep, int) and (new_timestep>=0)):
         raise ValueError('"new_timestep" must be a positive integer')
         
      if atime is None:
         self.bf.icommand(f'reset_timestep {new_timestep}')
      else:
         if isinstance(atime, (float, int)) and (atime>=0):
            self.bf.icommand(f'reset_timestep {new_timestep} time {atime}')
         else:
            raise ValueError('"atime" must be a positive float or integer, '+
                             'if not None')
   
   #################################################################################
   
   def canonical_MC_atom_swaps_setup_fixes(self, temp, atom_types, N=1, X=1):
   
      import itertools
      from toolbox import Toolbox
      rng = Toolbox.create_unique_randomstate_across_processors(self.bf)
   
      type_combs = list(itertools.combinations(atom_types,2))
      self.bf.icomment(f'MC fixes for pairs from types {atom_types} at temperature {temp} K')
      for type_pair in type_combs:
         self.bf.icommand('fix swap'+''.join(map(str, type_pair))+f' all atom/swap {N} {X} '+
                     f'{rng.integers(1, 1e5)} {temp} semi-grand no ke no types '+' '.join(map(str, type_pair)))
   
   #################################################################################
   
   def canonical_MC_atom_swaps_undo_fixes(self, atom_types=None, fixes=None):
   
      self.bf.icomment('Switching off MC fixes')
      if fixes is not None:
         if isinstance(fixes, list):
            self.lammps_unfix(fixes)
         else:
            raise ValueError('If not None, "fixes" must be a list')
      else:
         if isinstance(atom_types, (list, np.ndarray)):
            import itertools
            type_combs = list(itertools.combinations(atom_types,2))
            for type_pair in type_combs:
               self.lammps_unfix(f'swap'+''.join(map(str, type_pair)))
         else:
            raise ValueError('If not None, "atom_types" must be a list or a numpy ndarray. '+
                             'Also atom_types cannot be None is fixes is None')
      
   
   #################################################################################
      
   def run(self):
      pass
   
   #################################################################################
   
   def lammps_compute(self, compute_style, **kwargs):
      
      if not {'computeID', 'groupID'}.issubset(kwargs):
         raise RuntimeError('Either computeID or groupID or both are not passed to the function')
         
      compute_string = 'compute {0} {1} '.format(kwargs['computeID'], kwargs['groupID'])
         
      if compute_style == 'com':
         self.bf.lmp.command(compute_string+'com')
         
   #################################################################################
   
   def lammps_fix(self, fix_style, **kwargs):
      
      if not {'fixID', 'groupID'}.issubset(kwargs):
         raise RuntimeError('Either fixID or groupID or both are not passed to the function')
         
      fix_string = 'fix {0} {1} '.format(kwargs['fixID'], kwargs['groupID'])
      
      pass
   
   #################################################################################
      
   def lammps_unfix(self, fixID):
      
      if isinstance(fixID, str):
         self.bf.icommand('unfix {0}'.format(fixID))
      elif isinstance(fixID, list):
         for fid  in fixID:
            if not isinstance(fid, str):
               raise ValueError('fix-ID must be a string')
            else:
               self.bf.icommand('unfix {0}'.format(fid))
      else:
         raise ValueError('"fixID" must be a string or list of strings')
      
   #################################################################################
   
   def fix_com(self, fixID, directions=[0,1,2], group='all'):
      
      if not ( isinstance(fixID, int) and ( fixID > 0 ) ):
         raise ValueError('"fixID" must be a positive integer')
      
      if not isinstance(directions, list):
         raise ValueError('"directions" must be a list')
      elif not ( ( set(directions).issubset({0,1,2}) ) and 
               (len(directions) == len(set(directions)) ) ):
         raise ValueError('"directions" must be a list of non-repeating '+
                          'integers out of 0,1,2 which stands for the x, y '+
                          'and z directions')
      else:
         pass
      
      if not isinstance(group, str):
         raise ValueError('"group" must be a string')
        
      
      self.bf.lmp.command('compute avgforce_for_fix_com_{0} '.format(fixID)+
                          '{0} reduce ave fx fy fz'.format(group))
      dims = ['x', 'y', 'z']; restore_f = ['0.0', '0.0', '0.0']
      for d in directions:
         self.bf.lmp.command('variable restore_f{0}_for_fix_com_{1} '.format(dims[d], fixID)+
                             'equal -c_avgforce_for_fix_com_{0}[{1}]'.format(fixID, d+1))
         restore_f[d] = 'v_restore_f{0}_for_fix_com_{1}'.format(dims[d], fixID)
         
      self.bf.lmp.command('fix add_restore_force_for_fix_com_{0} '.format(fixID)+
                          '{0} addforce {1}'.format(group, ' '.join(restore_f)))
       
      self.bf.lmp.command('fix_modify energy no')
             
   
   #################################################################################
   
   def unfix_com(self, fixID):
      
      self.bf.lmp.command('unfix add_restore_force_for_fix_com_{0}'.format(fixID))
   
   #################################################################################   
   
   def __setattr__(self, name, value):      
      
      if name == 'bf':
         if not isinstance(value, Buffer):
            raise ValueError('Member variable "bf" must be an instance of the'+ 
                             'class "Buffer"')
         elif not hasattr(value, 'write_input_script'):
            raise AttributeError('"write_input_script" must be a member '+
                                 'variable of the Buffer instance.')
         elif not all([hasattr(value, x) for x in ['comm', 'comm_world']]):
            raise AttributeError('comm and comm_world must be attributes of '+
                                 'Buffer instance "bf". This is strange since '+
                                 'these members are created during initialisation '+
                                 'of Buffer class instance. Please check the '+
                                 'Buffer class in buffer module.')
         elif not hasattr(value, 'lmp'):
            raise AttributeError('The buffer instance "bf" must have a lammps '+
                                 'object as its member variable "lmp"')
         else:
            if value.write_input_script == True:
               if all([hasattr(value, x) for x in 
                                ['input_file_path', 'append_to_input_file', 
                                 'computer_sys_desc']]): 
                  value.update_inheritance_from_LAMMPS_Toolbox()
                  self.__dict__[name] = value
               else:
                  raise AttributeError(', '.join(['input_file_path',
                                    'append_to_input_file', 'computer_sys_desc']) + 
                                    ' must be attributes of Buffer instance "bf". '+
                                    'This can be achieved by calling the method '+
                                  '"update_buffer_for_LAMMPS" of the Buffer instance')
            else:
               value.update_inheritance_from_LAMMPS_Toolbox()
               self.__dict__[name] = value
            
      elif name == 'boxrelax_dirs':
         if value is None:
            self.__dict__[name] = value
         else:
            if not isinstance(value, Iterable):
               raise ValueError('"boxrelax_dirs" must be an Iterable')
            if len(value) > 6 or len(value) == 0:
               raise ValueError('"boxrelax_dirs" must have maximum fix entries '+
                                'and at least one entry')
            elif not set(value).issubset({'x', 'y', 'z', 'xy', 'xz', 'yz'}):
               raise ValueError('"boxrelax_dir" entries must be either "x" or "y" '+
                                'or "z"')
            else:
               self.__dict__[name] = list(set(value))
                     
      elif name == 'couple':
         if not isinstance(value, str):
            raise ValueError('couple keyword for box relax must be a str')
         elif not set({value}).issubset({'none', 'xyz', 'xy', 'yz', 'xz'}):
            raise ValueError('couple keyword for box relax must be one '+
                             'of the following: none or xyz or xy or yz or xz')
         else:
            self.__dict__[name] = value
            
      elif name == '_thermo_freq':
         if not isinstance(value, int):
            raise ValueError('thermo output frequency must be an integer')
         elif not value > 0:
            raise ValueError('thermo output frequency must be positive')
         else:
            self.__dict__[name] = value
            
      elif name == '_thermo_list':
         if not isinstance(value, list):
            raise ValueError('"thermo_list" must be a list')
         elif not all([isinstance(x, str) for x in value]):
            raise ValueError('"thermo_list" entries must be strings')
         else:
            self.__dict__[name] = value
            
            
      elif name == '_timestep':
         if isinstance(value, float) and (value > 0):
            self.__dict__[name] = value
         else:
            raise ValueError('Private member variable "_timestep" '+
                             'must be a positive float')
     
      else:
         pass
            
   #################################################################################
