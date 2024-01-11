from simulations import Simulations
import warnings
from buffer import Buffer
import numpy as np

class MolecularDynamics(Simulations):
   
   def __init__(self, bf:Buffer, thermo_list, thermo_freq=50, boxrelax_dirs=None, 
                couple='none'):
      
      Simulations.__init__(self, bf, thermo_list, thermo_freq, boxrelax_dirs, 
                           couple)
      
   ############################################################
   
   def set_deformation(self, ):
      
      pass
   
   ############################################################
   
   def modify_velocities(self, ):
      
      pass
   
   ############################################################
   
   def set_temp_n_pressure(self, nvt_fixID, groupID, temp_start, temp_stop, 
                           press_start=None, press_stop=None, 
                           keyword_value_pairs={}):
      
      if not isinstance(nvt_fixID, str):
         raise ValueError('"nvt_fixID" must be a string')
         
      if not isinstance(groupID, str):
         raise ValueError('"groupID" must be a string')
      
      if not ( isinstance(temp_start, (int, float)) and 
               isinstance(temp_stop, (int, float)) ):
         raise ValueError('Start and stop temperatures must be an integer or '+
                          'a float.')
      else:
         nvt_str = 'temp {0} {1} $(100.0*dt) '.format(temp_start, temp_stop)
         
      if (press_start is None) and (press_stop is None):
         if self.boxrelax_dirs is None:
            nvt_str = 'fix {0} {1} nvt '.format(nvt_fixID, groupID) + nvt_str            
         else:
            raise RuntimeError('No target pressure is assigned to the dimensions '+
                               'where barostat is set. press_start and press_stop '+
                               'must not be None when boxrelax_dirs is not None')
      elif (press_start is None) or (press_stop is None):
         raise ValueError('ONLY ONE of press_start and press_stop cannot be '+
                          'None. If one of them is None, the other has to be None')   
      else:
         if self.boxrelax_dirs is None:
            raise RuntimeError('No idea in which directions to apply the '+
                               'barostat. boxrelax_dirs must not be None when '+
                               'press_start and press_stop are not None')
         elif not ( isinstance(press_start, list) and isinstance(press_stop, list) ):
            raise ValueError('Both press_start and press_stop must be lists')
         elif ( ( len(press_start) != len(self.boxrelax_dirs) ) or 
                ( len(press_stop) != len(self.boxrelax_dirs) ) ):
            raise ValueError('press_start and press_stop must be of the same '+
                             'length as boxrelax_dirs')
         elif not ( all([isinstance(x, (int, float)) for x in press_start]) and
                    all([isinstance(x, (int, float)) for x in press_stop]) ):
            raise ValueError('All pressure components must either be integer or'+
                             'float in both press_start and press_stop')
         else:
            nvt_str = 'fix {0} {1} npt '.format(nvt_fixID, groupID) + nvt_str 
            nvt_str += ' '.join(['{0} {1} {2} {3}'.format(*x) for x in 
                                 zip(self.boxrelax_dirs, press_start, press_stop,
                                     ['$(1000.0*dt)']*len(self.boxrelax_dirs))]) 
            nvt_str += ' couple {0} '.format(self.couple)
 
      
      keyword_value_pairs0 = keyword_value_pairs
      keyword_value_pairs = {'nreset': 100}
      keyword_value_pairs.update(keyword_value_pairs0)
      
      for x in keyword_value_pairs:
         if x in ['tchain', 'pchain', 'mtk', 'tloop', 'ploop', 'nreset']:
            if ( isinstance(keyword_value_pairs[x], int) and
                 (keyword_value_pairs[x] >= 0) ):
               nvt_str += x+' {0} '.format(keyword_value_pairs[x])
            else:
               raise ValueError('The value for keyword {0} must '.format(x)+
                                'be a positive integer')
         elif x in ['drag', 'ptemp']:
            if ( isinstance(keyword_value_pairs[x], float) and
                 (keyword_value_pairs[x] >= 0) ):
               nvt_str += x+' {0} '.format(keyword_value_pairs[x])
            else:
               raise ValueError('The value for keyword {0} must '.format(x)+
                                'be a positive float')
         elif x == 'dilate':
            if isinstance(keyword_value_pairs[x], str):
               nvt_str += x+' {0} '.format(keyword_value_pairs[x])
            else:
               raise ValueError('The value for keyword dilate must '+
                                'be a string, which is the dilate-group-ID')
         elif x in ['scalexy', 'scaleyz', 'scalexz', 'flip']:
            if keyword_value_pairs[x] in ['yes', 'no']:
               nvt_str += x+' {0} '.format(keyword_value_pairs[x])
            else:
               raise ValueError('The value for keyword {0} must '.format(x)+
                                'be either yes or no')
         elif x == 'fixedpoint':
            if ( isinstance(keyword_value_pairs[x], list) and
                 len(keyword_value_pairs[x]) == 3 and 
                 all([isinstance(y, (int, float)) for y in 
                      keyword_value_pairs[x]]) ):
               nvt_str += x+' {0} '.format(' '.join(map(str,
                                             keyword_value_pairs[x])))
            else:
               raise ValueError('The value for keyword fixedpoint must be a'+
                                ' list of integers or floats of length 3')
         elif x == 'update':
            if x in ['dipole', 'dipole/dlm']:
               nvt_str += x+' {0} '.format(keyword_value_pairs[x])
            else:
               raise ValueError('The value for keyword update must be either '+
                                '"dipole" or "dipole/dlm"')
         else:
            possible_keywords = ['tchain', 'pchain', 'mtk', 'tloop', 'ploop', 
                                 'nreset', 'drag', 'ptemp', 'dilate', 
                                 'scalexy', 'scaleyz', 'scalexz', 'flip', 
                                 'fixedpoint', 'update']
            raise ValueError('Keywords must be chosen among the following '+
                             'options:\n'+ ' '.join(possible_keywords))
      
      if self.boxrelax_dirs is None:
         if np.isclose(temp_start, temp_stop):
            self.bf.icomment('NVT setup to equilibrate the system at {0}K'.format(
                             temp_start))
         else:
            self.bf.icomment('MD setup to heat the system from {0}K '.format(
                             temp_start) + 'to {0}K '.format(temp_stop) +
                             'under constant volume and constant number of '+
                             'atoms.')
      else:            
         if np.isclose(temp_start, temp_stop):
            if np.all(np.isclose(press_start, press_stop)):
               self.bf.icomment('NPT setup to equilibrate the system at '+
                                '{0}K '.format(temp_start) + 'under constant '+
                                'pressure of {0} MPa '.format(
                                ' '.join(map(str, 0.1*np.array(press_start))))+
                                'along directions {0} respectively.'.format(
                                   ' '.join(self.boxrelax_dirs)) )
            else:
               self.bf.icomment('MD setup to load the system from a pressure of '+
                                '{0} MPa '.format(
                                ' '.join(map(str, 0.1*np.array(press_start))))+
                                'to a pressure of {0} MPa '.format(
                                ' '.join(map(str, 0.1*np.array(press_stop))))+
                                'along directions {0} respectively.'.format(
                                   ' '.join(self.boxrelax_dirs)) +
                                'under a constant temperature of {0}K '.format(
                                   temp_start))
         else:
            if np.all(np.isclose(press_start, press_stop)):
               self.bf.icomment('MD setup to heat the system from {0}K '.format(
                                temp_start) + 'to {0}K '.format(temp_stop) +
                                'under constant pressure of {0} MPa '.format(
                                ' '.join(map(str, 0.1*np.array(press_start))))+
                                'along directions {0} respectively.'.format(
                                   ' '.join(self.boxrelax_dirs)) )
            else:
               self.bf.icomment('MD setup to heat the system from {0}K '.format(
                                temp_start) + 'to {0}K '.format(temp_stop) +
                                'while simultaneously loading it from a '+
                                'pressure of {0} MPa '.format(
                                ' '.join(map(str, 0.1*np.array(press_start))))+
                                'to a pressure of {0} MPa '.format(
                                ' '.join(map(str, 0.1*np.array(press_stop))))+
                                'along directions {0} respectively.'.format(
                                   ' '.join(self.boxrelax_dirs)) )
      
      self.bf.icommand(nvt_str)
      self.bf.iwhitespace()
      
   ############################################################
      
   def run(self, N, keyword_value_pairs={}):
      
      if not ( isinstance(N, int) and N>0 ):
         raise ValueError('N must be a positive integer')
      else:
         run_str = 'run {0} '.format(N)
      
      if not isinstance(keyword_value_pairs, dict):
         raise ValueError('"keyword_value_pairs" must be a dictionary')
      
      if ( 'pre' in keyword_value_pairs or
           'post' in keyword_value_pairs ):
         warnings.warn('It is not a good idea to change pre or post from their '+
                       'default values pre=yes and post=yes. So these keywords '+
                       'ignored!')
         
      if 'upto' in keyword_value_pairs:
         run_str += 'upto '

      if 'start' in keyword_value_pairs:
         if 'upto' in keyword_value_pairs:
            raise RuntimeError('It is a weird construct of the run command with '+
                          'upto and start/stop. Also no such example is '+
                          'mentioned in the LAMMPS documentation for run '+
                          'command. So either keep upto or start/stop keyword(s). '+
                          'Or you can alter this code if you are confident that '+
                          'this construct is valid.')
         N1 = keyword_value_pairs['start']
         if not ( isinstance(N1, int) and N1>0 ):
            raise ValueError('N1 must be a positive integer')
         else:
            if 'stop' in keyword_value_pairs:
               N2 = keyword_value_pairs['stop']
               if not ( isinstance(N2, int) and N2>0 ):
                  raise ValueError('N2 must be a positive integer')
               elif N2 <= N1:
                  raise ValueError('N2 must be less than N1')
               else:
                  run_str += 'start {0} stop {1}'.format(N1, N2)
            else:
               raise RuntimeError('"stop" keyword must be there if "start" '+
                                  'keyword is used')
               
      if 'every' in keyword_value_pairs:
         if not isinstance(keyword_value_pairs['every'], list):
            raise ValueError('values for "every" keyword must be provided in '+
                             'a list')
         
         val_list = keyword_value_pairs['every']
         if not ( isinstance(val_list[0], int) and val_list[0]>0 ):
            raise ValueError('M for "every" keyword must be a positive integer')
         elif not all([isinstance(x, str) for x in val_list[1:]]):
            raise ValueError('The LAMMPS commands for "every" keyword must be '+
                             'strings')
         else:
            run_str += ' '.join([str(val_list[0])]+val_list[1:])         
         
      
      self.bf.icommand(run_str)
      self.bf.iwhitespace()             
   
   ############################################################