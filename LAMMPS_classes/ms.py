from simulations import Simulations
import warnings
from buffer import Buffer
import numpy as np
from read_atomistic_system_in_LAMMPS import LoadSystem
import os

class MolecularStatics(Simulations):
   
   def __init__(self, bf:Buffer, thermo_list, thermo_freq=50, boxrelax_dirs=None, 
                couple='none', ms_diagnotic_dumps_location='./MS_diagnotic_dumps/',
                        default_LAMMPS_timestep=False):
      
      self._thermo_list = thermo_list
      Simulations.__init__(self, bf, None, thermo_freq, boxrelax_dirs, 
                           couple)
      
      self.ms_diagnotic_dumps_location = ms_diagnotic_dumps_location
      
      self.bf.run0()
      self.bf.write_dump_lmp('all', self.ms_diagnotic_dumps_location, 
                             'initial_structure_for_minimization.mpiio.dump', 
                             ['id', 'type', 'x', 'y', 'z', 'ix', 'iy', 'iz',
                              'fx', 'fy', 'fz'])     
   
      self._minimization_attempts_count = 0
      
      self.__dict__['_default_timesteps'] = {'lj': 0.005, # tau 
                                             'real': 1.0, # fs
                                             'metal': 0.001, # ps
                                             'si': 1.0e-8, # s (10 ns)
                                             'cgs': 1.0e-8, # s (10 ns)
                                             'electron': 0.001, # fs
                                             'micro': 2.0, # micro-seconds
                                             'nano': 0.00045 # ns
                                             }
      
      if default_LAMMPS_timestep:
         self._timestep = self._default_timesteps[self.bf.LAMMPS_units]
      else:
         self._timestep = 10*self._default_timesteps[self.bf.LAMMPS_units]
         
      self.bf.icommand(f'timestep {self._timestep}')
      
      self._update_stored_state()
                 
   ############################################################
   
   def dump_structure_at_initialization_of_minimization(self):
      
      self.bf.icomment('Dumping structure at the initialization of minimization '+
                       'with all fixes defined.\n This will shows the forces on '+
                       'atoms exerted due to fixes like setforce or addforce for '+
                       'example.\n However for minimization allowing for pressure '+
                       'relaxation, the fix box/relax will be applied after this '+
                       'dump command.')
      
      self.bf.write_regular_dump_files('dump_structure_at_minimization_initialization', 'all', 
                                       10, self.ms_diagnotic_dumps_location, 
                                       'structure_at_minimization_initialization.mpiio.dump', 
                                       ['id', 'type', 'x', 'y', 'z', 'ix', 'iy', 'iz',
                                        'fx', 'fy', 'fz'],
                                        dump_modify_args={'first': 'yes'})
      
      self.bf.icommand('min_style cg')
      self.bf.icommand('minimize 0.0 1e-8 1 200000')
         
      self.bf.icommand('undump dump_structure_at_minimization_initialization')
      
      self.bf.iwhitespace(2)
   
   ############################################################
   
   def _update_stored_state(self):
      
      self.bf.run0()
      
      if self.bf.lmp.has_id('variable', 'fmax0'):
         self.bf.icommand('variable fmax0 delete')
         self.bf.icommand('variable fmax0 equal $(fmax)')
      else:
         self.bf.icommand('variable fmax0 equal $(fmax)')
      
      if self.bf.lmp.has_id('fix', 'store_state'):        
         self.bf.icommand('unfix store_state')
         self.bf.icommand('fix store_state all store/state 0 x y z ix iy iz com no')
      else:
         self.bf.icommand('fix store_state all store/state 0 x y z ix iy iz com no')
         
      self.bf.iwhitespace(2)
      
   
   ############################################################
   
   def _restore_stored_state(self):
      
      if self.bf.lmp.has_id('fix', 'store_state'):  
      
         self.bf.icommand('variable x0 atom f_store_state[1]')
         self.bf.icommand('variable y0 atom f_store_state[2]')
         self.bf.icommand('variable z0 atom f_store_state[3]')
         self.bf.icommand('variable ix0 atom f_store_state[4]')
         self.bf.icommand('variable iy0 atom f_store_state[5]')
         self.bf.icommand('variable iz0 atom f_store_state[6]')  
         image_str = ' '.join(np.where(self.bf.lmp.extract_box()[-2], 
                                       ['v_ix0', 'v_iy0', 'v_iz0'], [0,0,0]))
         self.bf.icommand('set atom * x v_x0 y v_y0 z v_z0 '+
                          'image ' + image_str + ' vx 0.0 vy 0.0 vz 0.0')
         
         self.bf.icommand('variable x0 delete')
         self.bf.icommand('variable y0 delete')
         self.bf.icommand('variable z0 delete')
         self.bf.icommand('variable ix0 delete')
         self.bf.icommand('variable iy0 delete')
         self.bf.icommand('variable iz0 delete') 
      else:
         raise RuntimeError('There is no stored state to restore')
         
      self.bf.iwhitespace(2)
   
   ############################################################
   
   def nve_relax(self, nsteps=300, timestep=None, temp_tolerance=5, 
                 conf_check_freq=10):
      
      if timestep is not None:
         self.bf.icommand(f'timestep {timestep}')
         
      if conf_check_freq % 10 != 0:
         raise RuntimeError('"conf_check_freq" must be a multiple of 10')
         
      self.bf.icommand('variable temp_per_atom atom 0.5*mass*(vx^2+vy^2+vz^2)'+
                       '/1.5/8.617333*1e5')
      self.bf.icommand('compute max_temp_per_atom all reduce max v_temp_per_atom')
      self.bf.icommand(f'variable hot_atoms atom v_temp_per_atom>{temp_tolerance}')
      self.bf.icommand('group grp_hot_atoms dynamic all var hot_atoms')
      self.bf.icommand('variable num_hot_atoms equal count(grp_hot_atoms)')
         
      thermo_list = ['step', 'time', 'pe', 'ke', 'press', 'temp', 'fmax', 'fnorm',
                     'v_num_unrelaxed_atoms', 'c_max_temp_per_atom', 'v_num_hot_atoms']
      self.bf.icommand('thermo {0}'.format(10))
      self.bf.icommand('thermo_style custom {0}'.format(' '.join(thermo_list)))
      self.bf.icommand('thermo_modify flush yes')
      
      self.bf.run0()
      
      self.bf.icommand('variable my_Temp equal temp')      
      
      self.bf.icommand('set group all vx 0.0 vy 0.0 vz 0.0')
      
      self.bf.run0()
      self._update_stored_state()
      self.bf.icomment('Current fnorm = {0} eV/A'.format(
                                   self.bf.lmp.get_thermo('fnorm')),
                                   print_to_logfile=True)
      self.bf.icommand('variable temp0 equal $(temp)')
         
      self.bf.icommand('fix nve_relax all nve/limit 0.5')
      self.bf.icommand(f'fix check_temp_tol all halt 1 v_my_Temp > {temp_tolerance} '+
                       'error continue message yes')
      self.bf.icommand(f"run {nsteps} every {conf_check_freq} "+
                       "'if \"$(fmax) < ${fmax0}\" then "+
                       "\"variable fmax0 delete\" "+
                       "\"variable fmax0 equal $(fmax)\" "+
                       "\"variable temp0 delete\" "+
                       "\"variable temp0 equal $(temp)\" "+
                       "\"unfix store_state\" "+
                       "\"fix store_state all store/state 0 x y z ix iy iz com no\" '",
                       enclosing_quotes='"""')
      self.bf.icomment('Temperature of the last stored state = {0} K'.format(
                                   self.bf.lmp.extract_variable('temp0')),
                                   print_to_logfile=True)      
      self._restore_stored_state()
      self.bf.icommand('unfix nve_relax')
      self.bf.icommand('unfix check_temp_tol')
      self.bf.icommand('variable my_Temp delete')
      self.bf.icommand('variable temp0 delete')
      
      self.bf.icommand('group grp_hot_atoms delete')
      self.bf.icommand('variable temp_per_atom delete')
      self.bf.icommand('uncompute max_temp_per_atom')
      self.bf.icommand('variable hot_atoms delete')
      self.bf.icommand('variable num_hot_atoms delete')
      
      self.update_thermo(self._thermo_list, self._thermo_freq)
      self.bf.run0()
      
      if timestep is not None:
         self.bf.icommand(f'timestep {self._timestep}')
         
      self.bf.iwhitespace()      
      self.bf.icomment('Current fmax = {0} eV/A'.format(
                                   self.bf.lmp.get_thermo('fmax')),
                                   print_to_logfile=True)
      self.bf.icomment('Current fnorm = {0} eV/A'.format(
                                   self.bf.lmp.get_thermo('fnorm')),
                                   print_to_logfile=True)
      self.bf.icomment('Number of unrelaxed atoms = {0} ({1} %)'.format(
                                   int(self.bf.lmp.extract_variable(
                                         'num_unrelaxed_atoms')),
                                   (self.bf.lmp.extract_variable(
                                         'num_unrelaxed_atoms')/
                                   self.bf.lmp.get_natoms()*100)),
                                   print_to_logfile=True)
         
      self.bf.iwhitespace(2)
   
   ############################################################   
   
      
   def run(self, fmax_tol=1e-8, p_tol1=0.5, p_tol2=1.0, press_conv_crit = 'press',
           mode='L'):
      
      # the mode can be either "L" (for large systems) or "S" (for small systems)
      # The mode will slightly change the minimization steps to make it faster
      # What qualifies as large or small is upto the user to decide, from his/her
      # experience and/or trial and error.
      # CG and SD typically converges in less number of minimization steps, however each 
      # minimization step takes more time than a MD step (bottleneck for very
      # large simulation cell sizes) and often CG fails in line search.
      # FIRE, on the other hand, is damped dynamics and almost always converges
      # but can take very many steps.
      # So for smaller systems we would like to use CG or SD more, and for 
      # large systems we would prefer FIRE.
      
      self.bf.icommand(f'variable unrelaxed_atoms atom fx>{fmax_tol}||'+
                       f'fy>{fmax_tol}||fz>{fmax_tol}')
      self.bf.icommand('group grp_unrelaxed_atoms dynamic all var unrelaxed_atoms')
      self.bf.icommand('variable num_unrelaxed_atoms equal count(grp_unrelaxed_atoms)')
      
      self.update_thermo(thermo_list=self._thermo_list+['v_num_unrelaxed_atoms'],
                         thermo_freq=None )
      
      self.bf.run0()
      self._update_stored_state()
      
      if mode not in ['S', 'L']:
         raise RuntimeError('"mode" for minimization can either be "S" or "L" '+
                            'for small or large system sizes respectively')
   
      if self.boxrelax_dirs is None:
         
         self.bf.icomment('#####################################################\n'+
                          '## Minimization keeping periodic box lengths fixed ##\n'+
                          '#####################################################', 
                          print_to_logfile=True)
         self.bf.iwhitespace()
         self.bf.icomment('Current fmax = {0} eV/A'.format(
                                   self.bf.lmp.get_thermo('fmax')),
                                   print_to_logfile=True)
         self.bf.icomment('Current fnorm = {0} eV/A'.format(
                                   self.bf.lmp.get_thermo('fnorm')),
                                   print_to_logfile=True)
         self.bf.icomment('Current potential energy = {0} eV'.format(
                                   self.bf.lmp.get_thermo('pe')),
                                   print_to_logfile=True)
         self.bf.icomment('Current pressure = {0} bars'.format(
                                   self.bf.lmp.get_thermo('press')),
                                   print_to_logfile=True)
         self.bf.icomment('Current pressure components = '+
                       '[{0} {1} {2} {3} {4} {5}] bars'.format(
                        self.bf.lmp.get_thermo('pxx'), self.bf.lmp.get_thermo('pyy'),
                        self.bf.lmp.get_thermo('pzz'), self.bf.lmp.get_thermo('pxy'),
                        self.bf.lmp.get_thermo('pxz'), self.bf.lmp.get_thermo('pyz')
                        ),  print_to_logfile=True)
         self.bf.iwhitespace()
         flag = 1
         while self.bf.lmp.get_thermo('fmax') > fmax_tol:
            if flag > 10:
               break
            else:
               if mode=='S':
                  self.minimize_w_fmax_crit('cg', fmax_tol=fmax_tol)
                  if self.bf.lmp.get_thermo('fmax') > fmax_tol:
                     self.minimize_w_fmax_crit('quickmin', maxiter=600, fmax_tol=fmax_tol)
                  if self.bf.lmp.get_thermo('fmax') > fmax_tol:
                     self.minimize_w_fmax_crit('fire', maxiter=200, fmax_tol=fmax_tol)
               elif mode=='L':
                  self.minimize_w_fmax_crit('cg', maxiter=200, fmax_tol=fmax_tol)
                  if self.bf.lmp.get_thermo('fmax') > fmax_tol:
                     self.minimize_w_fmax_crit('quickmin', maxiter=3000, fmax_tol=fmax_tol)
                  if self.bf.lmp.get_thermo('fmax') > fmax_tol:
                     self.minimize_w_fmax_crit('fire', maxiter=1000, fmax_tol=fmax_tol)
               flag = flag+1
         if self.bf.lmp.get_thermo('fmax') > fmax_tol:
            self.minimize_w_fmax_crit('fire', fmax_tol=fmax_tol)
         if self.bf.lmp.get_thermo('fmax') > fmax_tol:
            self.bf.icomment('Unable to minimize the system such that '+
                             'fmax is below {0} eV/A. '.format(fmax_tol)+
                             'Current value of fmax = {0} eV/A'.format(
                                   self.bf.lmp.get_thermo('fmax')),
                             print_to_logfile=True)
            exit_code = 0
            self.bf.iwhitespace()
         else:
            self.bf.icomment('Energy minimized below fmax tolerance of '+
                             '{0} eV/A. '.format(fmax_tol)+
                             'Current value of fmax = {0} eV/A'.format(
                                   self.bf.lmp.get_thermo('fmax')),
                             print_to_logfile=True)
            exit_code = 1
            self.bf.iwhitespace()
      else:
         self.bf.icomment('#####################################################\n'+
                          '## Minimization allowing periodic box lengths/tilts  ###\n'+
                          '### ({0}) to change allowing for ###\n'.format(
                                       ', '.join(self.boxrelax_dirs)) +
                          '#### pressures allowing these directions to relax ####\n' +
                          '#######################################################', 
                          print_to_logfile=True)
         self.bf.iwhitespace()
         self.bf.icomment('Current fmax = {0} eV/A'.format(
                                   self.bf.lmp.get_thermo('fmax')),
                                   print_to_logfile=True)
         self.bf.icomment('Current fnorm = {0} eV/A'.format(
                                   self.bf.lmp.get_thermo('fnorm')),
                                   print_to_logfile=True)
         self.bf.icomment('Current potential energy = {0} eV'.format(
                                   self.bf.lmp.get_thermo('pe')),
                                   print_to_logfile=True)
         self.bf.icomment('Current pressure = {0} bars'.format(
                                   self.bf.lmp.get_thermo('press')),
                                   print_to_logfile=True)
         self.bf.icomment(( 'Current pressure components (bars) \n'+
                           f'pxx = {self.bf.lmp.get_thermo("pxx")}, ' +
                           f'pyy = {self.bf.lmp.get_thermo("pyy")}, ' +
                           f'pzz = {self.bf.lmp.get_thermo("pzz")}, \n' +
                           f'pxy = {self.bf.lmp.get_thermo("pxy")}, ' +
                           f'pxz = {self.bf.lmp.get_thermo("pxz")}, ' +
                           f'pyz = {self.bf.lmp.get_thermo("pyz")}' ),
                           print_to_logfile=True)
         self.bf.iwhitespace()
         if p_tol1 >= p_tol2:
            warnings.warn('"p_tol1" must be less than "p_tol2". So the values '+
                          'are exchanged!')
            p_tol2, p_tol1 = (p_tol1, p_tol2)
            
         self.minimize_with_box_relax(mode, p_tol1, p_tol2, press_conv_crit, 
                                      fmax_tol=fmax_tol)
         
         if ( (not self.pressure_convergence(press_conv_crit, p_tol2)) or
              (self.bf.lmp.get_thermo('fmax') > fmax_tol) ):
            exit_code = 0
         else:
            exit_code = 1
         
      self._timestep = self._default_timesteps[self.bf.LAMMPS_units]
      self.bf.icommand(f'timestep {self._timestep}')
         
      self.bf.icommand('set group all vx 0.0 vy 0.0 vz 0.0')
      
      self.bf.icommand('variable fmax0 delete')
      self.bf.icommand('unfix store_state')      
      
      self.bf.run0()
      
      return exit_code
   
#########################################################################
   
   
   def minimize_w_fmax_crit(self, min_style, fmax_tol=1e-8, maxiter=200000):
      
      def make_maxiter_list(maxiter):
         l = [500]
         while maxiter != 0:
            if l[-1] >= maxiter:
               l[-1] = maxiter
               break
            elif maxiter-l[-1]<100:
               l[-1] = maxiter
               break
            else:
               maxiter = maxiter - l[-1]
               if l[-1] > 100:
                  l.extend([l[-1] - 100])
               else:
                  l.extend([100])
                      
         return l
      
      maxiter_list = make_maxiter_list(maxiter)
      if np.sum(maxiter_list) != maxiter:
         raise RuntimeError('Error in forming maxiter_list')
      
      self.bf.icomment('Commencing {0} minimization with '.format(min_style) +
                       'fmax tolerance of {0} eV/A \n and '.format(fmax_tol) +  
                       'maximum iterations of {0}:'.format(maxiter),
                       print_to_input_script=False,
                       print_to_logfile=True)
      self.bf.icommand('min_style {0}'.format(min_style))
      self.bf.icommand('min_modify norm inf')
      min_stuck = 0
      current_fmax = self.bf.lmp.get_thermo('fmax')
      for i, ml in enumerate(maxiter_list):
         self.bf.icommand('minimize 0.0 {0} {1} 200000'.format(fmax_tol, ml))
         if ((self.bf.lmp.get_thermo('fmax')-current_fmax) < -fmax_tol) :
            if self.bf.lmp.get_thermo('fmax') > fmax_tol:
               current_fmax = self.bf.lmp.get_thermo('fmax')
               self._update_stored_state()
               if i == len(maxiter_list) - 1:
                  self.bf.icomment(f'fmax has decreased by more than {fmax_tol} eV/A '+
                                   f'in last {ml} steps of the maximum {maxiter} steps. \n'+
                                   f'Change in fmax in last {ml} steps = '+
                                   f'{self.bf.lmp.get_thermo("fmax") - current_fmax} eV/A')
            else:
               break
         elif self.bf.lmp.get_thermo('fmax') <= fmax_tol:
            break
         else:
            if i == len(maxiter_list)-1:
               min_stuck = 1
            self.bf.iwhitespace()
            self.bf.icomment(f'fmax has decreased by less than {fmax_tol} eV/A '+
                             f'(or has increased) in last {ml} steps. \n'+
                             (f'So minimization is suspended before {maxiter} steps. \n'
                              if (i == len(maxiter_list)-1) else '')+
                             f'Change in fmax in last {ml} steps = '+
                             f'{self.bf.lmp.get_thermo("fmax") - current_fmax} eV/A')
            self.bf.icomment('Restoring last stored state of the system')
            self._restore_stored_state()
            self.bf.run0()
            self.bf.icomment('Current fmax = {0} eV/A'.format(
                                   self.bf.lmp.get_thermo('fmax')),
                                   print_to_logfile=True)
            self.bf.icomment('Current fnorm = {0} eV/A'.format(
                                   self.bf.lmp.get_thermo('fnorm')),
                                   print_to_logfile=True)
            break
      
      if min_stuck == 1:
         self.bf.iwhitespace()
         self.bf.icomment('Running 300 NVE MD runs on the current unrelaxed system.\n '+
                          'Part of the excess potential energy above minimum will '+
                          'be converted into kinetic energy, \n thus minimizing  '+
                          'potential energy and lower fmax (possibly). Also the MD '+
                          'might induce adequate fluctuations \n to drive the '+
                          'system out of the unchanging fmax configuration before '+
                          'subsequent minimization attempts')
         self.nve_relax(timestep=self._default_timesteps[self.bf.LAMMPS_units])
         current_fmax = self.bf.lmp.get_thermo('fmax')
         self.bf.iwhitespace()
         if self.boxrelax_dirs is None:
            self.bf.icommand('min_style hftn')
         else:
            self.bf.icommand('min_style cg')
         self.bf.icommand('min_modify norm inf')
         if min_style in ['cg', 'sd', 'hftn']:            
            for i, ml in enumerate(maxiter_list):
               self.bf.icommand('minimize 0.0 {0} {1} 2000'.format(fmax_tol, ml))
               if ((self.bf.lmp.get_thermo('fmax')-current_fmax) < -fmax_tol) :
                  if self.bf.lmp.get_thermo('fmax') > fmax_tol:
                     current_fmax = self.bf.lmp.get_thermo('fmax')
                     self._update_stored_state()
                  else:
                     break
               elif ((self.bf.lmp.get_thermo('fmax') <= fmax_tol) or 
                     (i == len(maxiter_list)-1)):
                  break
               else:
                  self.bf.iwhitespace()
                  self.bf.icomment(f'fmax has decreased by less than {fmax_tol} eV/A '+
                             f'(or has increased) in last {ml} steps. '+
                             f'So minimization is suspended before {maxiter} steps. '+
                             f'Change in fmax in last {ml} steps = '+
                             f'{self.bf.lmp.get_thermo("fmax") - current_fmax} eV/A')
                  self.bf.icomment('Restoring last stored state of the system')
                  self._restore_stored_state()
                  self.bf.run0()
                  break
         else:
            self.bf.icommand('minimize 0.0 {0} {1} 2000'.format(fmax_tol, 300))
            self.bf.icomment('Current fmax = {0} eV/A'.format(
                                   self.bf.lmp.get_thermo('fmax')),
                                   print_to_logfile=True)
            self.bf.icomment('Current fnorm = {0} eV/A'.format(
                                   self.bf.lmp.get_thermo('fnorm')),
                                   print_to_logfile=True)
            self.bf.iwhitespace()
            if self.bf.lmp.get_thermo('fmax') > self.bf.lmp.extract_variable('fmax0'):
               self.bf.icomment('Restoring last stored state of the system')
               self._restore_stored_state()
               self.bf.run0()
               self.bf.icomment('Current fmax = {0} eV/A'.format(
                                      self.bf.lmp.get_thermo('fmax')),
                                      print_to_logfile=True)
               self.bf.icomment('Current fnorm = {0} eV/A'.format(
                                      self.bf.lmp.get_thermo('fnorm')),
                                      print_to_logfile=True)
               self.bf.iwhitespace()
               self.bf.icommand('min_style {0}'.format(min_style))
               self.bf.icommand('min_modify norm inf')
               for i, ml in enumerate(maxiter_list):
                  self.bf.icommand('minimize 0.0 {0} {1} 200000'.format(fmax_tol, ml))
                  if ((self.bf.lmp.get_thermo('fmax')-current_fmax) < -fmax_tol) :
                     if self.bf.lmp.get_thermo('fmax') > fmax_tol:
                        current_fmax = self.bf.lmp.get_thermo('fmax')
                        self._update_stored_state()
                     else:
                        break
                  elif ((self.bf.lmp.get_thermo('fmax') <= fmax_tol) or 
                        (i == len(maxiter_list)-1)):
                     break
                  else:
                     self.bf.iwhitespace()
                     self.bf.icomment(f'fmax has decreased by less than {fmax_tol} eV/A '+
                                f'(or has increased) in last {ml} steps. '+
                                f'So minimization is suspended before {maxiter} steps. '+
                                f'Change in fmax in last {ml} steps = '+
                                f'{self.bf.lmp.get_thermo("fmax") - current_fmax} eV/A')
                     self.bf.icomment('Restoring last stored state of the system')
                     self._restore_stored_state()
                     self.bf.run0()
                     break
              
      self.bf.iwhitespace()
      self.bf.run0()
      self.bf.icomment('Current fmax = {0} eV/A'.format(
                                   self.bf.lmp.get_thermo('fmax')),
                                   print_to_logfile=True)
      self.bf.icomment('Current fnorm = {0} eV/A'.format(
                                   self.bf.lmp.get_thermo('fnorm')),
                                   print_to_logfile=True)
      self.bf.icomment('Current potential energy = {0} eV'.format(
                                self.bf.lmp.get_thermo('pe')),
                                print_to_logfile=True)
      self.bf.icomment('Number of unrelaxed atoms = {0} ({1} %)'.format(
                                   int(self.bf.lmp.extract_variable(
                                         'num_unrelaxed_atoms')),
                                   (self.bf.lmp.extract_variable(
                                         'num_unrelaxed_atoms')/
                                   self.bf.lmp.get_natoms()*100)),
                                   print_to_logfile=True)
      self.bf.icomment('Current pressure = {0} bars'.format(
                                   self.bf.lmp.get_thermo('press')),
                                   print_to_logfile=True)
      self.bf.icomment(( 'Current pressure components (bars) \n'+
                                 f'pxx = {self.bf.lmp.get_thermo("pxx")}, ' +
                                 f'pyy = {self.bf.lmp.get_thermo("pyy")}, ' +
                                 f'pzz = {self.bf.lmp.get_thermo("pzz")}, \n' +
                                 f'pxy = {self.bf.lmp.get_thermo("pxy")}, ' +
                                 f'pxz = {self.bf.lmp.get_thermo("pxz")}, ' +
                                 f'pyz = {self.bf.lmp.get_thermo("pyz")}' ),
                                 print_to_logfile=True)
      self.bf.iwhitespace()
      
      self._minimization_attempts_count += 1
      
      self.bf.write_dump_lmp('all', self.ms_diagnotic_dumps_location, 
                             ( 'minimization_attempt_{0}_with_{1}'.format(
                               self._minimization_attempts_count, min_style ) + 
                               '.mpiio.dump' ), 
                             ['id', 'type', 'x', 'y', 'z', 'ix', 'iy', 'iz',
                              'fx', 'fy', 'fz'])
   
############################################################

   def minimize_with_box_relax(self, mode, p_tol1, p_tol2, press_conv_crit, 
                               fmax_tol=1e-8):
   
      max_trials_outer = 3
      max_trials_inner = 15
      
      p_relax_success = False
      
      for ot in range(max_trials_outer):
         
         if ot == 0:
            p_tol = p_tol1
         else:
            p_tol = p_tol2
            
         if ( self.pressure_convergence(press_conv_crit, p_tol) and
              (self.bf.lmp.get_thermo('fmax') < fmax_tol) ):
            if ot == 0:
               self.bf.icomment('Already, energy minimized and pressure relaxed \n'+
                                'below fmax tolerance of {0} ev/A and \n'.format(
                                      fmax_tol)+
                                'pressure tolerance of {0} bars. \n'.format(p_tol)+
                                'So no need of any minimization!',
                                print_to_logfile=True)
               self.bf.iwhitespace()
            else:
               self.bf.icomment('Both energy minimized and pressure relaxed \n'+
                                'below fmax tolerance of {0} ev/A and \n'.format(
                                      fmax_tol)+
                                'pressure tolerance of {0} bars. \n'.format(p_tol),
                                print_to_logfile=True)
               self.bf.icomment('Current pressure = {0} bars'.format(
                                 self.bf.lmp.get_thermo('press')),
                                 print_to_logfile=True)
               self.bf.icomment(( 'Current pressure components (bars) \n'+
                                 f'pxx = {self.bf.lmp.get_thermo("pxx")}, ' +
                                 f'pyy = {self.bf.lmp.get_thermo("pyy")}, ' +
                                 f'pzz = {self.bf.lmp.get_thermo("pzz")}, \n' +
                                 f'pxy = {self.bf.lmp.get_thermo("pxy")}, ' +
                                 f'pxz = {self.bf.lmp.get_thermo("pxz")}, ' +
                                 f'pyz = {self.bf.lmp.get_thermo("pyz")}' ),
                                 print_to_logfile=True)
               self.bf.icomment('Current fmax = {0} eV/A'.format(
                                 self.bf.lmp.get_thermo('fmax')),
                                 print_to_logfile=True)
               self.bf.icomment('Current potential energy = {0} eV'.format(
                                         self.bf.lmp.get_thermo('pe')),
                                         print_to_logfile=True)
               self.bf.iwhitespace()
            break
                     
         for i in range(max_trials_inner):
            
            if mode == 'L':
               if (ot == 0) and (i == 0):
                  if press_conv_crit in ['press', 'pxx', 'pyy', 'pzz', 
                                         'pxy', 'pxz', 'pyz']:
                     self.bf.icommand(f'variable abs_Pr equal abs({press_conv_crit})')
               self.bf.icommand('fix halt_at_relaxed_pressure all halt '+
                                '1 v_abs_Pr < {0} '.format(p_tol)+
                                'error continue message yes')
            self.bf.icommand('fix boxrelax all box/relax {0}couple {1} '.format(
                             ' 0.0 '.join(self.boxrelax_dirs)+' 0.0 ', 
                             self.couple) + 'vmax 0.0005 nreset 100')
            self.minimize_w_fmax_crit('cg', fmax_tol=fmax_tol)
                        
            if self.pressure_convergence(press_conv_crit, p_tol):
               p_relax_success = True
               self.bf.iwhitespace()
               self.bf.icomment('Pressure relaxed in {0} '.format(
                                                    (ot*max_trials_inner)+i+1)+
                                'box/relax minimizations',
                                print_to_logfile=True)
               self.bf.icomment('Current values of pressure components to be relaxed (bars) \n '+
                                ', '.join([(f"p{x*2} = "+'{0}'.format(self.bf.lmp.get_thermo(f'p{x*2}')))
                                           if len(x) == 1 else
                                           (f"p{x} = "+'{0}'.format(self.bf.lmp.get_thermo(f'p{x}')))
                                           for x in self.boxrelax_dirs]),
                                print_to_logfile=True)
               self.bf.iwhitespace()
               self.bf.icommand('unfix boxrelax')
               if mode == 'L':
                  if self.bf.lmp.has_id('fix', 'halt_at_relaxed_pressure'):
                     self.bf.icommand('unfix halt_at_relaxed_pressure')
            else:
               self.bf.icomment('Pressure not relaxed below {0} bars '.format(
                                                                         p_tol)+
                                'after {0} '.format((ot*max_trials_inner)+i+1)+
                                'box/relax minimizations', print_to_logfile=True)
               self.bf.icomment('Current values of pressure components to be relaxed (bars) \n '+
                                ', '.join([(f"p{x*2} = "+'{0}'.format(self.bf.lmp.get_thermo(f'p{x*2}')))
                                           if len(x) == 1 else
                                           (f"p{x} = "+'{0}'.format(self.bf.lmp.get_thermo(f'p{x}')))
                                           for x in self.boxrelax_dirs]),
                                print_to_logfile=True)
               if i>5 and i<10:
                  if mode == 'L':
                     self.minimize_w_fmax_crit('sd', fmax_tol=fmax_tol, maxiter=200)
                     self.minimize_w_fmax_crit('cg', fmax_tol=fmax_tol, maxiter=200)
                  else:
                     self.minimize_w_fmax_crit('sd', fmax_tol=fmax_tol)
                     self.minimize_w_fmax_crit('cg', fmax_tol=fmax_tol)
               self.bf.icommand('unfix boxrelax')
               if mode == 'L':
                  if self.bf.lmp.has_id('fix', 'halt_at_relaxed_pressure'):
                     self.bf.icommand('unfix halt_at_relaxed_pressure')
            
            if not p_relax_success:
               if mode == 'L':
                  self.minimize_w_fmax_crit('fire', fmax_tol=fmax_tol)
               else:
                  self.minimize_w_fmax_crit('fire', fmax_tol=fmax_tol, 
                                            maxiter=1000)
            else:
               if self.bf.lmp.get_thermo('fmax') < fmax_tol:
                  self.bf.icomment('Energy also minimized during '+
                                   'box/relax minimizations',
                                   print_to_logfile=True)
                  self.bf.icomment('Current fmax = {0} eV/A'.format(
                                   self.bf.lmp.get_thermo('fmax')),
                                   print_to_logfile=True)
                  break
               else:
                  self.bf.icomment('Energy NOT minimized during '+
                                   'box/relax minimizations below \n'+
                                   'fmax tolerance of {0} ev/A. '.format(fmax_tol)+
                                   'So we will run FIRE minimization.',
                                   print_to_logfile=True)
                  self.bf.icomment('Current fmax = {0} eV/A'.format(
                                   self.bf.lmp.get_thermo('fmax')),
                                   print_to_logfile=True)
                  self.minimize_w_fmax_crit('fire', fmax_tol=fmax_tol)
                  
                  if not self.pressure_convergence(press_conv_crit, p_tol):
                     p_relax_success = False
                     self.bf.icomment('Pressure have increase above {0} bars \n'.format(
                                                                         p_tol)+
                                      'after FIRE minimization. So we need to repeat \n'+
                                      'box/relax minimizations',
                                      print_to_logfile=True)
                     self.bf.icomment('Current values of pressure components to be relaxed (bars) \n '+
                                      ', '.join([(f"p{x*2} = "+'{0}'.format(self.bf.lmp.get_thermo(f'p{x*2}')))
                                           if len(x) == 1 else
                                           (f"p{x} = "+'{0}'.format(self.bf.lmp.get_thermo(f'p{x}')))
                                                 for x in self.boxrelax_dirs]),
                                      print_to_logfile=True)
                  elif self.bf.lmp.get_thermo('fmax') < fmax_tol:
                     break
                  
   def pressure_convergence(self, criterion, p_tol):
      
      press_strs = ['press', 'pxx', 'pyy', 'pzz', 'pxy', 'pxz', 'pyz']
      if criterion in press_strs:
         return abs(self.bf.lmp.get_thermo(criterion)) < p_tol
      elif hasattr(criterion, '__call__'):
         return criterion([self.bf.lmp.get_thermo(x) for x in press_strs],
                           p_tol)
      else:
         raise ValueError('"criterion" must be a LAMMPS pressure strings or '+
                          'a function taking a list and a float and evaluating '+
                          'to a boolean')
      
      
      
############################################################   
                     
   def __setattr__(self, name, value):
      
      Simulations.__setattr__(self, name, value)
      
      if name == 'ms_diagnotic_dumps_location':
         if not isinstance(value, str):
            raise ValueError('"ms_diagnotic_dumps_location" takes a path '+
                             'which must be provided in a string')
         elif value[-1] != '/':
            raise ValueError('A path to a directory must end with a "/"')
         else:
            self.bf.mkdir_with_LAMMPS(value)
            self.__dict__[name] = value
               
      elif name == '_minimization_attempts_count':
         if isinstance(value, int) and (value >= 0):
            self.__dict__[name] = value
         else:
            raise ValueError('Private member variable "_minimization_attempts_count" '+
                             'must be a positive integer')
            
      elif name == '_timestep':
         if isinstance(value, float) and (value > 0):
            self.__dict__[name] = value
         else:
            raise ValueError('Private member variable "_timestep" '+
                             'must be a positive float')
      else:
         pass
      
      
############################################################   
############################################################   

         
      
class RestartMolecularStatics(MolecularStatics, LoadSystem):
   
   def __init__(self, ms_diagnotic_dumps_location,
                restart_dump_filename=None):
      
      self.ms_diagnotic_dumps_location = ms_diagnotic_dumps_location
      self.restart_dump_filepath = restart_dump_filename
      
   ############################################################
      
   def initiate_restart_system(self, bf:Buffer, n_atom_types:int, log_filename='none', 
                               logfile_folder='./', LAMMPS_units = 'metal', 
                               LAMMPS_atom_style = 'atomic', wd = None):
      
      ls = LoadSystem(bf, self.restart_dump_filepath, n_atom_types)
      ls.initiate_LAMMPS_system(log_filename=log_filename, folder=logfile_folder,
                                LAMMPS_units = LAMMPS_units, 
                                LAMMPS_atom_style = LAMMPS_atom_style,
                                wd = wd)
      ls.read_from_dump(['x', 'y', 'z', 'ix', 'iy', 'iz'])
      self.bf = bf      
      
   ############################################################
   
   def initiate_simulation_env(self, thermo_list, thermo_freq=50, 
                               boxrelax_dirs=None, couple='none',
                               default_LAMMPS_timestep=False):
      
      Simulations.__init__(self, self.bf, thermo_list, thermo_freq, 
                           boxrelax_dirs, couple)
      
      self.__dict__['_default_timesteps'] = {'lj': 0.005, # tau 
                                             'real': 1.0, # fs
                                             'metal': 0.001, # ps
                                             'si': 1.0e-8, # s (10 ns)
                                             'cgs': 1.0e-8, # s (10 ns)
                                             'electron': 0.001, # fs
                                             'micro': 2.0, # micro-seconds
                                             'nano': 0.00045 # ns
                                             }
      
      if default_LAMMPS_timestep:
         self._timestep = self._default_timesteps[self.bf.LAMMPS_units]
      else:
         self._timestep = 10*self._default_timesteps[self.bf.LAMMPS_units]
         
      self.bf.icommand(f'timestep {self._timestep}')
      
      self._update_stored_state()
      
   ############################################################   
         
   def __setattr__(self, name, value):
      
      import re, glob
      
      if name == 'ms_diagnotic_dumps_location':
         if not isinstance(value, str):
            raise ValueError('"ms_diagnotic_dumps_location" takes a path '+
                             'which must be provided in a string')
         elif value[-1] != '/':
            raise ValueError('A path to a directory must end with a "/"')
         elif not os.path.isdir(value):
            raise ValueError('The directory pointing to the restart dump file '+
                             'does not exist')
         else:
            self.__dict__[name] = value
            
      elif name == 'restart_dump_filepath':
         
         if value is None:           
            file_list = [x.split('/')[-1] for x in 
                                  glob.glob(self.ms_diagnotic_dumps_location+
                                  'minimization_attempt_*')]
            self.__dict__[name] = (self.ms_diagnotic_dumps_location+
                                   file_list[np.argmax(list(map(int, 
                                       [x[slice(*re.search(r'\d+', x).span())] 
                                        for x in  file_list])))] )
            self._minimization_attempts_count = max(list(map(int, 
                                       [x[slice(*re.search(r'\d+', x).span())] 
                                        for x in  file_list])))
         
         else:
            if not isinstance(value, str):
               raise ValueError('"restart_dump_filename" must be a string if '+
                                'not None')
            elif not os.path.exists(self.ms_diagnotic_dumps_location+value):
               raise ValueError(f'File "{value}" does not exist in the MS diagnostic '+
                                f'dump folder "{self.ms_diagnotic_dumps_location}"')
            elif re.search(r'minimization_attempt_\d',value) is None:
               raise ValueError('"restart_dump_filename" must have the pattern '+
                                '"minimization_attempt_\d"')
            else:
               self.__dict__[name] = self.ms_diagnotic_dumps_location+value
               self._minimization_attempts_count = int(value[slice(
                                               *re.search(r'\d+',value).span())])
            
         
               
      else:
         MolecularStatics.__setattr__(self, name, value)
      
