from empty_class import EmptyClass
import pandas as pd
import numpy as np
import warnings

class MDRunData(EmptyClass):
   
   def __init__(self, loop_time_in_secs = np.nan, nprocs = -9999,
                Nsteps = -9999, natoms = -9999, description=''):
      
      self.metadata_update(loop_time_in_secs, nprocs, Nsteps, natoms,
                           description)
   
   def metadata_update(self, loop_time_in_secs = np.nan, nprocs = -9999,
              Nsteps = -9999, natoms = -9999, description=''):
      
      self.loop_time_in_secs = loop_time_in_secs               
      self.nprocs = nprocs         
      self.Nsteps = Nsteps         
      self.natoms = natoms
      self.description = description
      
   def combine_unique_metadata(self):
      
      if isinstance(self.nprocs, list):
         if len(np.unique(self.nprocs)) == 1:
            self.nprocs = self.nprocs[0]
      
      if isinstance(self.natoms, list):
         if len(np.unique(self.natoms)) == 1:
            self.natoms = self.natoms[0]    
            
      if isinstance(self.description, list):
         if len(np.unique(self.description)) == 1:
            self.description = self.description[0]
      
      
   def update_description(self, description):
      
      if not isinstance(description, str):
         raise TypeError('While updating "description", the description must be '+
                         'a string')       
      self.description = description      
         
   def rundata_init(self, thermo_keywords):
      
      if not isinstance(thermo_keywords, list):
         raise TypeError('"thermo_keywords" must be a list')
      elif not all([isinstance(x, str) for x in thermo_keywords]):
         raise ValueError('"thermo_keywords" must be a list of strings')
         
      self.thermo_output = pd.DataFrame(columns=thermo_keywords)
   
   def rundata_update(self, data_per_step):
      
      if not hasattr(self, 'thermo_output'):
         raise RuntimeError('"thermo_output" DataFrame must be initialized first '+
                            'with "thermo_keywords" using the method "rundata_init"')
      
      if not isinstance(data_per_step, list):
         raise TypeError('"data_per_step" must be a list')
      elif len(data_per_step) != len(self.thermo_output.columns):
         raise ValueError('"data_per_step" must have the same length as the '+
                          'as the number of "thermo_keywords"')
      elif not all([isinstance(x, (int, float)) for x in data_per_step]):
         raise ValueError('"data_per_step" must be a list of integers/floats')
      
      self.thermo_output = self.thermo_output.append(dict(zip(
                          self.thermo_output.columns, data_per_step)), 
                                                     ignore_index=True)
            
   def append(self, list_md_rundata):
      
      if not isinstance(list_md_rundata, list):
         if isinstance(list_md_rundata, MDRunData):
            list_md_rundata = [list_md_rundata]
         else:
            raise TypeError('Input must be a list of MDRunData instances. If not' +
                            'it has to be a MDRunData instance.')
         
      if len(list_md_rundata) == 0:
         return
      
      md_run_data = list_md_rundata.pop(0)
      
      if not isinstance(md_run_data, MDRunData):
         raise TypeError('Only instances of MDRunData can be appended')
         
      
      if not isinstance(self.loop_time_in_secs, list):
         self.loop_time_in_secs = [self.loop_time_in_secs]
      if not isinstance(md_run_data.loop_time_in_secs, list):
         md_run_data.loop_time_in_secs = [md_run_data.loop_time_in_secs]
         
      if not isinstance(self.nprocs, list):
         self.nprocs = [self.nprocs]
      if not isinstance(md_run_data.nprocs, list):
         md_run_data.nprocs = [md_run_data.nprocs] 
         
      if not isinstance(self.Nsteps, list):
         self.Nsteps = [self.Nsteps]
      if not isinstance(md_run_data.Nsteps, list):
         md_run_data.Nsteps = [md_run_data.Nsteps]
         
      if not isinstance(self.natoms, list):
         self.natoms = [self.natoms]
      if not isinstance(md_run_data.natoms, list):
         md_run_data.natoms = [md_run_data.natoms] 
         
      if not isinstance(self.description, list):
         self.description = [self.description]
      if not isinstance(md_run_data.description, list):
         md_run_data.description = [md_run_data.description] 
         
        
      self.loop_time_in_secs += md_run_data.loop_time_in_secs
      self.nprocs += md_run_data.nprocs
      self.Nsteps += md_run_data.Nsteps
      self.natoms += md_run_data.natoms
      self.description += md_run_data.description
      
      if len(list_md_rundata) == 0:
         if len(np.unique(self.loop_time_in_secs)) == 1:
            self.loop_time_in_secs = self.loop_time_in_secs[0]
         if len(np.unique(self.nprocs)) == 1:
            self.nprocs = self.nprocs[0]
         if len(np.unique(self.Nsteps)) == 1:
            self.Nsteps = self.Nsteps[0]
         if len(np.unique(self.natoms)) == 1:
            self.natoms = self.natoms[0]
         if len(np.unique(self.description)) == 1:
            self.description = self.description[0]
         
      self.thermo_output = self.thermo_output.append(md_run_data.thermo_output, 
                                                     ignore_index=True)
   
      self.append(list_md_rundata)
         
      
            
   
   def __setattr__(self, name, value):
      
      if name == 'loop_time_in_secs':
         if isinstance(value, list):
            if not all([isinstance(x, float) for x in value]):
               raise TypeError('If "loop_time_in_secs" is a list, its entries '+
                               'must be floats')
            else:
               self.__dict__[name] = value
         elif not isinstance(value, float):
            raise TypeError('"loop_time_in_secs" must be a float or a list of floats')
         else:
            self.__dict__[name] = value
         
      elif name == 'nprocs':   
         if isinstance(value, list):
            if not all([isinstance(x, int) for x in value]):
               raise TypeError('If "nprocs" is a list, its entries '+
                               'must be integers')
            else:
               self.__dict__[name] = value
         elif not isinstance(value, int):
            raise TypeError('"nprocs" must be an integer or a list of integers')
         else:
            self.__dict__[name] = value
            
      elif name == 'Nsteps':   
         if isinstance(value, list):
            if not all([isinstance(x, int) for x in value]):
               raise TypeError('If "Nsteps" is a list, its entries '+
                               'must be integers')
            else:
               self.__dict__[name] = value
         elif not isinstance(value, int):
            raise TypeError('"Nsteps" must be an integer or a list of integers')
         else:
            self.__dict__[name] = value
            
      elif name == 'natoms':   
         if isinstance(value, list):
            if not all([isinstance(x, int) for x in value]):
               raise TypeError('If "natoms" is a list, its entries '+
                               'must be integers')
            else:
               self.__dict__[name] = value
         elif not isinstance(value, int):
            raise TypeError('"natoms" must be an integer or a list of integers')
         else:
            self.__dict__[name] = value
            
      elif name == 'thermo_output':   
         if not isinstance(value, pd.DataFrame):
            raise TypeError('"thermo_output" must be a Pandas DataFrame')
         else:
            self.__dict__[name] = value
            
      elif name == 'description':
         if isinstance(value, list):
            if not all([isinstance(x, str) for x in value]):
               raise TypeError('If "description" is a list, its entries '+
                               'must be strings')
            else:
               self.__dict__[name] = value
         elif not isinstance(value, str):
            raise TypeError('"description" must be an string or a list of strings')
         else:
            self.__dict__[name] = value   
         
      else:
         raise ValueError('Not a valid member of the MDRunData class')
