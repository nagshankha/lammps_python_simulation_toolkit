from empty_class import EmptyClass
from md_run_data import MDRunData

class LogFileReader(EmptyClass):
   
   def __init__(self, logfilepath):
      
      self.logfilepath = logfilepath
    
   #####################################################   
      
   def open_file(self):
      
      self.f = open(self.logfilepath, 'rt')
      
   #####################################################
      
   def close_file(self):
      
      self.f.close()
      
   #####################################################
   
   def jump_to_start(self):
      
      self.f.seek(0,0)
 
   #####################################################
   
   def jump_to_end(self):
      
      self.f.seek(0,2)
      
   #####################################################
   
   def extract_thermo_outputs_in_MD_run(self):
      
      run_records = []
      start = False

      for i, line in enumerate(self.f):
         
         sline = line.split()
         if len(sline) == 0:
            continue
         
         if sline != ['run', '0'] and sline[0] == 'run':
            
            if not sline[1].isdigit():
               continue
            else:
               Nsteps = int(sline[1])
               
            if len(sline) == 2:
               run_upto = False
            elif len(sline) == 3:
               if sline[-1] == 'upto':
                  run_upto = True
               else:
                  continue
            else:
               continue        
            
            self.f.readline() #Skip a line
            start = True
            run_records.append(MDRunData())
            headline = True
            continue
         
         if start:
            if headline:
               headline = False
               run_records[-1].rundata_init(sline)
            elif sline[0] == 'Loop':
               start = False
               run_records[-1].metadata_update(loop_time_in_secs = float(sline[3]), 
                          nprocs = int(sline[5]), Nsteps = int(sline[8]), 
                          natoms = int(sline[11]))
               if 'Step' in run_records[-1].thermo_output.columns:
                  if run_records[-1].Nsteps != ( run_records[-1].thermo_output['Step'].iloc[-1]
                                               - run_records[-1].thermo_output['Step'].iloc[0] ):
                     raise RuntimeError('Number of run steps is not same as the '+
                                        'number of steps printed in the diagnostics')
                  if run_upto:
                     if run_records[-1].thermo_output['Step'].iloc[-1] != Nsteps:
                        raise RuntimeError('The last run step does not correspond to '+
                                           'the N in "run N upto" command')
                  else:
                     if run_records[-1].Nsteps != Nsteps:
                        raise RuntimeError('Number of run steps does not correspond to '+
                                           'the N in "run N" command')
            else:
               run_records[-1].rundata_update(conv_str2num(sline))
      
      if len(run_records) == 1:         
         self.run_records = run_records[0]
      else:
         self.run_records = run_records
               
   #####################################################            
                  
def conv_str2num(list_of_str):

   if not isinstance(list_of_str, list):
      raise ValueError('The input must be a list of strings')
   
   if not all([isinstance(x, str) for x in list_of_str]):
      raise ValueError('The input must be a list of strings')      
      
   l = []
   
   for s in list_of_str:
      flag=0
      sl = s.split('.')
      
      if len(sl) > 2 or len(sl) == 0:
         raise RuntimeError(s+' cannot be converted to integer or float')
         
      if sl[0].isdigit() or sl[0][1:].isdigit():
         if not (sl[0][0].isdigit() or (sl[0][0] in ['+', '-'])):
            raise RuntimeError(s+' cannot be converted to integer or float')
         
         if len(sl) == 1:
            l.extend([int(s)])
            continue
         else:
            flag=1
      else:
         raise RuntimeError(s+' cannot be converted to integer or float')
      
      if flag == 1:
         if sl[1].isdigit():
            l.extend([float(s)])
         elif 'e+' in sl[1]:
            sle = sl[1].split('e+')
            if len(sle) != 2:
               raise RuntimeError(s+' cannot be converted to integer or float')
            elif not all([x.isdigit() for x in sle]):
               raise RuntimeError(s+' cannot be converted to integer or float')
            else:
               l.extend([float(s)])
         elif 'e-' in sl[1]:
            sle = sl[1].split('e-')
            if len(sle) != 2:
               raise RuntimeError(s+' cannot be converted to integer or float')
            elif not all([x.isdigit() for x in sle]):
               raise RuntimeError(s+' cannot be converted to integer or float')
            else:
               l.extend([float(s)])
      else:
         raise RuntimeError(s+' cannot be converted to integer or float')
         
   return l

         
   ##################################################### 
      
         
