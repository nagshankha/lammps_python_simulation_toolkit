from buffer import Buffer, ProxyCommunicator
from buffer_manager import BufferManager
from toolbox import Toolbox
from lammps import lammps
import numpy as np


class System(BufferManager):
   
   def __init__(self, bf:Buffer, n_atom_types:int):
      
      self.addBuffer(bf)
      self.n_atom_types = n_atom_types
            
###################################################################

   def initiate_LAMMPS_system(self, log_filename='none', folder='./',
                              LAMMPS_units = 'metal', 
                              LAMMPS_atom_style = 'atomic',
                              wd = None):
      
      Toolbox.mkdir(self.bf, folder)
      
      if not isinstance(log_filename, str):
         raise ValueError('"log_filename" must be a string')
      
      if ( log_filename != 'none' and 
           log_filename.split('.')[-1] != 'log' ):
         log_filename = log_filename + '.log'
         
      if isinstance(self.bf.comm, ProxyCommunicator):
         self.lmp = lammps(cmdargs=['-log', (folder+log_filename 
                                             if log_filename != 'none'
                                             else log_filename)])
      else:
         self.lmp = lammps(cmdargs=['-log', (folder+log_filename 
                                             if log_filename != 'none'
                                             else log_filename)],
                           comm = self.bf.comm)
      
      self.add2bf(['lmp'])
      self.bf.add2bf({'LAMMPS_units': LAMMPS_units, 
                      'LAMMPS_atom_style': LAMMPS_atom_style})
      self.bf.ipwd(wd)
      self.bf.icommand('units {0}'.format(LAMMPS_units))
      self.bf.icommand('atom_style {0}'.format(LAMMPS_atom_style))
      self.bf.icommand('atom_modify map array')
      self.bf.iwhitespace()
      
###################################################################

   def define_cuboid_simul_box(self, box_dims):
      
      if not isinstance(box_dims, (np.ndarray, list)):
         raise ValueError('"box_dims" must be a numpy array or a list')
      elif len(box_dims) != 6:
         raise ValueError('"box_dims" must have 6 entries: '+
                          '[xlo xhi ylo yhi zlo zhi]')
      elif not all([isinstance(x, (int, float)) for x in box_dims]):
         raise ValueError('Entries in "box_dims" must be integers or floats')
      elif box_dims[1] < box_dims[0]:
         raise ValueError('xlo must be less than xhi')
      elif box_dims[3] < box_dims[2]:
         raise ValueError('ylo must be less than yhi')
      elif box_dims[5] < box_dims[4]:
         raise ValueError('zlo must be less than zhi')
      else:
         box_dims_str = ' '.join(map(str, box_dims))
         self.bf.icommand('region reg_simul_box block {0} units box'.format(
                             box_dims_str))
         self.bf.icommand('create_box {0} reg_simul_box'.format(self.n_atom_types))
            
############################################################################
      
   def define_crystal_structure(self, style, scale, keyword_value_pairs={}):
      
      self.bf.define_crystal_structure_in_LAMMPS(style, scale, 
                                                 keyword_value_pairs)
      
###################################################################      

   def assign_random_atomtypes(self, type_choices=None, concentration=None):
      
      if type_choices is None:
         type_choices = np.arange(1,self.n_atom_types+1)
      else:
         if isinstance(type_choices, (np.ndarray, tuple, list)):
            type_choices = np.array(type_choices)
         else:
            raise ValueError('"type_choices" must be an iterable')
            
         if len(np.shape(type_choices)) != 1:
            raise ValueError('"type_choices" must be a 1D array')
         elif type_choices.dtype != int:
            raise ValueError('"type_choices" must be an integer array')
         elif len(set(type_choices)) != len(type_choices):
            raise ValueError('"type_choices" must not have repeated entries')
         elif ( (np.min(type_choices) <= 0) or 
                (np.max(type_choices) > self.n_atom_types) ):
            raise ValueError('Elements in "type_chocies" must be between 1 and '+
                             'the maximum number of atom types')
         else:
            pass
         
      if concentration is None:
         concentration = np.ones(len(type_choices))/len(type_choices)
      else:
         if isinstance(concentration, (np.ndarray, tuple, list)):
            concentration = np.array(concentration)
         else:
            raise ValueError('"concentration" must be an iterable')
         
         if len(np.shape(concentration)) != 1:
            raise ValueError('"concentration" must be a 1D array')
         elif concentration.dtype != float:
            raise ValueError('"concentration" must be a float array')
         elif len(concentration) != len(type_choices):
            raise ValueError('"concentration" must be of same length as '+
                             "type_choices")
         elif not np.all(np.logical_and(concentration>=0, concentration<=1)):
            raise ValueError('All concentrations must be between 0 and 1 '+
                             '(included)')
         elif not np.isclose(np.sum(concentration), 1.0):
            raise ValueError('The concentrations must sum up to 1')
         
      from toolbox import Toolbox
      rng = Toolbox.create_unique_randomstate_across_processors(self.bf)
      atom_types = rng.choice(type_choices, size=self.n_atoms,
                              p=concentration )
      
      if len(type_choices) == 1:
         self.bf.icomment('Assigning all atoms to type {0}'.format(
                                                           type_choices[0]))
      else:
         self.bf.icomment('Assigning types {0} to atoms randomly with '.format(
                           ' '.join(map(str, type_choices)) )+
                          'probabilities (concentrations) {0} respectively.'.format(
                                ' '.join(map(str, concentration)) ))
      self.bf.set_peratom_attr('type', atom_types)
      
###################################################################      

   def randomly_shuffle_atomtypes(self):
      
      from toolbox import Toolbox
      rng = Toolbox.create_unique_randomstate_across_processors(self.bf)
      atom_types = self.bf.gather_peratom_attr('type')
      rng.shuffle(atom_types )
      
      self.bf.icomment('Shuffling atom types randomly')
      self.bf.set_peratom_attr('type', atom_types)
      
###########################################################################
  
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
            
      elif name == 'lmp':
         if isinstance(value, lammps):
            self.__dict__[name] = value
         else:
            raise ValueError('Member variable "lmp" must be a lammps instance.')
            
      elif name == 'n_atom_types':
         if not ( isinstance(value, int) and (value > 0) ):
            raise ValueError('Member variable "n_atom_types" must be a positive '+
                             'integer')
         else:
            self.__dict__[name] = value
            
      elif name == 'n_atoms':
         if not ( isinstance(value, int) and (value > 0) ):
            raise ValueError('Member variable "n_atoms" must be a positive '+
                             'integer')
         else:
            self.__dict__[name] = value
            
      else:
         pass 
      
################################################################### 
