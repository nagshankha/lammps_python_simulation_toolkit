from empty_class import EmptyClass
import numpy as np

class ProxyCommunicator:
   def __init__(self):
      self.rank = 0
      self.size = 1
   def Barrier(self):
      pass
   def bcast(self, data, root=0):
      return data

class Buffer(EmptyClass):

   def __init__(self, comm_world=None):
      
      if comm_world is None:
         self.comm_world = ProxyCommunicator()
      else:
         from mpi4py import MPI
         if not isinstance(comm_world, MPI.Intracomm):
            raise ValueError('"comm_world" must be an instance of '+
                             'mpi4py.MPI.Intracomm')
         else:
            self.comm_world = comm_world
      
   #################################################################
   
   def add2bf(self, input_dict):
      if not isinstance(input_dict, dict):
         raise ValueError('For updating Buffer instance directly, the "add2bf" '+
                          'must receive a dictionary of names and values')
      elif len(input_dict) == 0:
         raise ValueError('No variables to update the Buffer instance')
      elif not all([isinstance(x, str) for x in input_dict.keys()]):
         raise ValueError('Buffer member variable names must be strings')
      elif any([( len(x.strip()) == 0 or len(x) != len(x.strip()) ) 
                for x in input_dict.keys()]):
         raise ValueError('Buffer member variable names must not be empty '+
                          'or have whitespaces')
      else:
         self.__dict__.update(input_dict)
         
   #################################################################
   
   def del_from_bf(self, input_keys):
      if not isinstance(input_keys, list):
         raise ValueError('Entries to be deleted from bf must be provided in a ' +
                          'list of strings')
      elif len(input_keys) == 0:
         raise ValueError('No variable names are provided to delete from the '+
                          'Buffer instance')
      elif not all(list(map(lambda x: isinstance(x, str), input_keys))):
         raise ValueError('Entries in "input_keys" must be strings')
      elif len(set(input_keys)-set(self.__dict__.keys())) != 0:
         raise ValueError('The following members asked to delete are not in '+
                          'the current Buffer instance: {0}'.format(
                                set(input_keys)-set(self.__dict__.keys())))
      else:
         input_keys = list(set(input_keys))
         
      list(map(self.__delitem__, filter(self.__contains__, input_keys)))
      
   #################################################################      
      
   def split_communicator(self, color_dict):
      
      if self.comm_world == None:
         raise RuntimeError('Splitting of communicator is not possible since '+
                            'MPI.COMM_WORLD does not exist in the Buffer instance')
      
      if not isinstance(color_dict, dict):
         raise ValueError('"color_dict" must be a dictionary')
      elif not all([isinstance(x, int) for x in color_dict.keys()]):
         raise ValueError('keys in "color_dict" must be integers')
      elif any(np.array(list(color_dict.keys())) < 0):
         raise ValueError('keys in "color_dict" must be non-negative integers')
      else:
         pass
      
      if not all([isinstance(x, set) for x in color_dict.values()]):
         raise ValueError('values in "color_dict" must be of sets')
      elif not all([np.array(list(x)).dtype == int for x in color_dict.values()]):
         raise ValueError('values in "color_dict" must be of sets of integers')
      elif not set.union(*color_dict.values()) == set(np.arange(self.comm_world.size)):
         raise ValueError('Every process must belong to any one of the split '+
                          'communicators. Meaning there should be no need of '+
                          'color MPI_UNDEFINED for any process during the split '+
                          'operation')
      elif not all(list(map(len, color_dict.values()))):
         raise ValueError('No value in "color_dict" should be an empty set')
      elif sum(map(len, color_dict.values())) != self.comm_world.size: 
         raise ValueError('No two values in "color_dict" should have a common '+
                          'element. Meaning no process can belong to two split '+
                          'communicators after the splitting process')
      else:
         pass
      
      for key in color_dict.keys():
         if self.comm_world.rank in color_dict[key]:
            color = key
            break
         else:
            continue            
            
      self.comm = self.comm_world.Split(color, self.comm_world.rank)
      self.comm_color = color
      
   #################################################################
   
   def update_buffer_for_LAMMPS(self, input_file_path=None, 
                append_to_input_file=True,
                computer_sys_desc=None):
   
      self.update_inheritance_from_LAMMPS_Toolbox()
      
      self.initialize_LAMMPS_input_script(input_file_path, 
                   append_to_input_file, computer_sys_desc)
   
   #################################################################
   
   def update_inheritance_from_LAMMPS_Toolbox(self):
      
      from dynamic_inheritance import dynamic_inheritance
      from lammps_toolbox import LAMMPS_Toolbox
      
      if LAMMPS_Toolbox not in self.__class__.__bases__:
         dynamic_inheritance(self.__class__, (LAMMPS_Toolbox,))     
      
   #################################################################

   def __setattr__(self, name, value):
      
      if name == 'comm_world':
         self.__dict__[name] = value
         self.__dict__['comm'] = value
      else:   
         self.__dict__[name] = value
