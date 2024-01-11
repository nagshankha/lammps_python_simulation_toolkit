from empty_class import EmptyClass

class BufferManager(EmptyClass):
   
   def __init__(self):
      raise RuntimeError('class "BufferManager" must not be initialised, ONLY inherited')
      
   ################################
   
   def addBuffer(self, bf):
      self.bf = bf
            
   ################################
   
   def add2bf(self, input_keys='all', exclude=None):
      if input_keys == 'all':
         if not exclude == None:
            if not isinstance(exclude, list):
               raise ValueError('If not None, "exclude" must be a list of strings')
            elif not all(list(map(lambda x: isinstance(x, str), exclude))):
               raise ValueError('Entries in "exclude" must be strings')
            elif len(set(exclude)-set(self.__dict__.keys())) != 0:
               raise ValueError('The following members of the current objects asked NOT to be '+
                                'incorporated in current Buffer instance are absent: {0}'.format(
                                   set(exclude)-set(self.__dict__.keys())))
            else:
               input_keys = self.__dict__.keys()
               self.bf.__dict__.update(dict([(x,self.__dict__[x]) for x in input_keys 
                                             if x not in exclude]))
         else:
            self.bf.__dict__.update(self.__dict__)
      else:
         if not isinstance(input_keys, list):
            raise ValueError('Members of current class that needed to be incorporated ' +
                             'as member of "bf" must be provided in a list of strings')
         elif len(input_keys) == 0:
            raise ValueError('No variables to update the Buffer instance')
         elif not all(list(map(lambda x: isinstance(x, str), input_keys))):
            raise ValueError('Entries in "input_keys" must be strings')
         elif len(set(input_keys)-set(self.__dict__.keys())) != 0:
            raise ValueError('The following members of the current objects asked to be '+
                             'incorporated in current Buffer instance are absent: {0}'.format(
                                   set(input_keys)-set(self.__dict__.keys())))
         else:
            self.bf.__dict__.update(dict([(x,self.__dict__[x]) for x in input_keys]))
            
   ###########################################################################
         
   def del_from_bf(self, input_keys):
      if not isinstance(input_keys, list):
         raise ValueError('Entries to be deleted from bf must be provided in a ' +
                          'list of strings')
      elif len(input_keys) == 0:
         raise ValueError('No variable names are provided to delete from the '+
                          'Buffer instance')
      elif not all(list(map(lambda x: isinstance(x, str), input_keys))):
         raise ValueError('Entries in "input_keys" must be strings')
      elif len(set(input_keys)-set(self.bf.__dict__.keys())) != 0:
         raise ValueError('The following members asked to delete are not in '+
                          'the current Buffer instance: {0}'.format(
                                set(input_keys)-set(self.bf.__dict__.keys())))
      else:
         input_keys = list(set(input_keys))
      
      
      list(map(self.bf.__delitem__, filter(self.bf.__contains__, input_keys)))
           
   ###########################################################################   
     
   
