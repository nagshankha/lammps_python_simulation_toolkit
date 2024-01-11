from empty_class import EmptyClass

class WriteInputScript(EmptyClass):
   
   def initialize_LAMMPS_input_script(self, input_file_path:str=None, 
                append_to_input_file:bool=True,
                computer_sys_desc:str=None):
      
      if input_file_path is None:
         self.write_input_script = False
      else:
         if not isinstance(input_file_path, str):
            raise ValueError('"input_file_path" must be a string')
         else:
            self.write_input_script = True
            self.input_file_path = input_file_path
            if isinstance(append_to_input_file, bool):
               self.append_to_input_file = append_to_input_file
            else:
               raise ValueError('"append_to_input_file" must be a boolean')
            if computer_sys_desc is None:
               import os
               if 'computer_sys_desc' in os.environ:
                  self.computer_sys_desc = os.environ['computer_sys_desc']
               else:
                  raise ValueError('Environment variable "computer_sys_desc" '+
                                   'is not found, which provides the system '+
                                   'description for documentation in the input '+
                                   'script. Please define it in your '+
                                   'current terminal before running your Python '+
                                   'script or put it in your .bashrc. '+
                                   'Alternatively you can provide the system '+
                                   'description through the optional argument '+
                                   '"computer_sys_desc"')
            elif isinstance(computer_sys_desc, str):
               self.computer_sys_desc = computer_sys_desc
            else:
               raise ValueError('"computer_sys_desc" must be a string')
      
   #########################################################################
            
   def icommand(self, command_str, inline_comment = None, enclosing_quotes='"'):
      
      if enclosing_quotes not in ['\'', '"', '"""']:
         raise ValueError('enclosing_quotes must be either \', " or """.')
      
      if not isinstance(command_str, str):
         raise ValueError('"command_str" must be a string')
         
      if inline_comment is not None:
         if not isinstance(inline_comment, str):
            raise ValueError('"inline_comment" must be a string')
            
      # ... fill in for the inline comment
         
      if self.write_input_script:
         if self.append_to_input_file:
            self.lmp.command('print '+enclosing_quotes+command_str+
                             enclosing_quotes+' append {0} screen no'.format(
                                   self.input_file_path))
         else:            
            self.lmp.command('print '+enclosing_quotes+command_str+
                             enclosing_quotes+' file {0} screen no'.format(
                                   self.input_file_path))
            
      self.lmp.command(command_str)
            
   #########################################################################
   
   def icomment(self, comment_str, print_to_input_script=True, 
                print_to_logfile=False):
      
      if not isinstance(comment_str, str):
         raise ValueError('"comment_str" must be a string')
      else:
         comment_str = '# ' + '\n# '.join([x.strip() for x in comment_str.split('\n') 
                                           if x.strip()!=''])
               
      if not isinstance(print_to_logfile, bool):
         raise ValueError('Print option "print_to_logfile" must be a boolean')
         
      if not isinstance(print_to_input_script, bool):
         raise ValueError('Print option "print_to_input_script" must be a boolean')
         
      if not (print_to_input_script or print_to_logfile):
         raise ValueError('Both "print_to_input_script" and "print_to_logfile" '+
                          'cannot be False, because in that case this method '+
                          'has no use')
         
      if print_to_input_script and (not self.write_input_script):
         raise RuntimeError('You wanted to print the comment to the '+
                            'input script but you have not provided the '+
                            'input script path')
         
      if print_to_logfile:
         screen = 'yes'
      else:
         screen = 'no'
               
      if self.write_input_script:
         if print_to_input_script:
            if self.append_to_input_file:
               self.lmp.command('print "{0}" append {1} screen {2}'.format(comment_str,
                                                               self.input_file_path,
                                                               screen))
            else:            
               self.lmp.command('print "{0}" file {1} screen {2}'.format(comment_str,
                                                               self.input_file_path,
                                                               screen))
         else:
            self.lmp.command('print "{0}" screen yes'.format(comment_str))
      else:
         self.lmp.command('print "{0}" screen yes'.format(comment_str))
            
   #########################################################################
   
   def iwhitespace(self, nlines=1):
      
      if not isinstance(nlines, int):
         raise ValueError('Number of lines of whitespace "nlines" must be an '+
                          'integer')
         
      if self.write_input_script:
         if self.append_to_input_file:
            self.lmp.command('print "{0}" append {1} screen no'.format('\n'*(nlines-1),
                                                         self.input_file_path))
         else:            
            self.lmp.command('print "{0}" file {1} screen no'.format('\n'*(nlines-1),
                                                         self.input_file_path))            
            
   #########################################################################
   
   def ipwd(self, wd=None):
      
      import os
      
      self.icomment('Absolute path of current working directory on {0}'.format(
                     self.computer_sys_desc))
      self.icomment('** Uncomment the following line to run this input script '+
                    'from any other directory on {0}'.format(
                       self.computer_sys_desc))
      if wd is None:
         self.icommand('shell cd {0}'.format(os.getcwd()))
      else:
         if not isinstance(wd, str):
            raise ValueError('Working directory "wd" must be a string')
         elif not os.path.isdir(wd):
            raise ValueError('Requested working directory does not exist')
         else:
            self.icommand('shell cd {0}'.format(os.path.abspath(wd)))
      self.input_file_path = os.path.relpath(self.input_file_path, 
                                             start=wd)
      self.iwhitespace()      
      
      
   #########################################################################
   
   
   
