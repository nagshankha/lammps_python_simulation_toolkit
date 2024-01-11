from empty_class import EmptyClass
import numpy as np

class Toolbox(EmptyClass):
   
   def __init__(self):
      raise RuntimeError('class "Toolbox" must not be initialised, ONLY inherited')
      
##########################################################################
   
   def create_unique_randomstate_across_processors(self):
      
      if self.comm.rank == 0:
         rng = np.random.default_rng()
      else:
         rng = None
         
      rng = self.comm.bcast(rng, root=0)
      
      return rng
   
##########################################################################

   def mkdir(self, path, verbose=False):
      
      import os
      
      if not isinstance(path, str):
         raise ValueError('"path" must be a string')
      elif len(path.split('..')) != 1:
         raise ValueError(' makedirs() will become confused if the path elements '+
                          'to create include pardir (eg. ".." on UNIX systems).')
      elif os.path.isdir(path):
         if self.comm.rank == 0:
            if verbose:
               print('Path "{0}" already exists'.format(path))
         self.comm.Barrier()
      else:
         if self.comm.rank == 0:
            os.makedirs(path)
         self.comm.Barrier()
         
##########################################################################
         
