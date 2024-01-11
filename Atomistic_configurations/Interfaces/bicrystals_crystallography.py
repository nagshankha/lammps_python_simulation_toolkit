import numpy as np
import miscellaneous_computes_crystallography as misc_crys
import sympy
from scipy.special import comb
from sympy.solvers.diophantine.diophantine import diop_linear
from sympy.core.containers import Tuple
import fractions
import itertools
import os



class EmptyClass:
   pass


class BicrystalCrystallography(EmptyClass):

   def __init__(self, rot_axis, rot_angle, z_dir, x_dir, lattice, rot_angle_conv = 'deg'):

      # z_dir is the normal to the boundary with respect to one of the crytals
      # The z-axis of the simulation cell will be directed towards z_dir
      # x_dir is the direction on the boundary plane which will align with the
      # x-axis of the simulation cell. x_dir is defined with respect to the 
      # same crystal which is used to define z_dir  
      
      self.primitive_vecs = lattice['primitive_vecs']  
      self.lat_name = lattice['lat_name']
      
      if rot_angle_conv == 'deg':
         self.rot_angle = np.deg2rad(rot_angle)
      elif rot_angle_conv == 'rad':
         pass
      else:
         raise ValueError('Rotation angle must be in degrees ("deg") or radians ("rad")')
         
      if not all([isinstance(x, np.ndarray) for x in [rot_axis, z_dir, x_dir]]):
         raise ValueError('All the input directions must be np.ndarray')
      elif not all(np.array([rot_axis.dtype, z_dir.dtype, x_dir.dtype]) == int):
         raise ValueError('All input directions must be integer arrays like Miller indices, '+
                          'primarily to ensure they are not irrational')
      else:
         self.rot_axis = rot_axis
         if not np.isclose(np.dot(z_dir, x_dir), 0):
            raise RuntimeError('z_dir and x_dir must be perpendicular')
         else:
            self.z_dir_crys1 = z_dir
            self.x_dir_crys1 = x_dir   
               
      self.y_dir_crys1 = np.cross(self.z_dir_crys1, self.x_dir_crys1)
      self.y_dir_crys1 = misc_crys.get_miller_indices(self.y_dir_crys1)
      
      self.orient_crys1 = np.c_[self.x_dir_crys1, self.y_dir_crys1, self.z_dir_crys1].T
      
      self.crys_struc = misc_crys.CrystalStructure(self.primitive_vecs)
      self.multiplicity_crys1, self.spacing_crys1 = self.crys_struc.latticeInterplanarSpacingnDisplacements(
                                        self.orient_crys1, return_relative_displacements=False,
                                        return_multiplicity=True, return_lattice_spacing=True)[-2:]    
         
      self.z_n_x_dirs_other_crystal()
      
      # np.sqrt(x) always returns a float even if it is an integer. This 
      # function checks whether the output is an integer (like when x=9) or not
      test_int = lambda x: str(np.sqrt(x)).split('.')[1] == '0'
      
      # Checking whether the lattice spacings of crystal 1 and crystal 2 are
      # in rational proportion along x direction
      sum_sq_x_dir_crys1 = np.sum(self.x_dir_crys1**2)
      sum_sq_x_dir_crys2 = np.sum(self.x_dir_crys2**2)
      gcd_sum_sq_x_dir_crys1n2 = np.gcd(sum_sq_x_dir_crys1, sum_sq_x_dir_crys2)
      sum_sq_x_dir_crys1_gcd_norm = int( sum_sq_x_dir_crys1/
                                         gcd_sum_sq_x_dir_crys1n2 )
      sum_sq_x_dir_crys2_gcd_norm = int( sum_sq_x_dir_crys2/ 
                                         gcd_sum_sq_x_dir_crys1n2 )
      if not all( map(test_int, [sum_sq_x_dir_crys1_gcd_norm, 
                                 sum_sq_x_dir_crys2_gcd_norm] ) ):
         raise RuntimeError('The lattice spacings of crystal 1 and crystal 2 ' +
                            'along x direction are not in rational proportion')
      else:
         self.x_spacing_ratio = misc_crys.get_miller_indices(np.array(
                               [self.spacing_crys1[0], self.spacing_crys2[0]])**2)
         if not all( map(test_int, self.x_spacing_ratio) ):
            raise RuntimeError('Elements of the x_spacing_ratio at this stage '+
                               'must be squares of integers')
         else:
            self.x_spacing_ratio = np.sqrt(self.x_spacing_ratio).astype(int)[::-1]
         
         
      # Checking whether the lattice spacings of crystal 1 and crystal 2 are
      # in rational proportion along y direction
      sum_sq_y_dir_crys1 = np.sum(self.y_dir_crys1**2)
      sum_sq_y_dir_crys2 = np.sum(self.y_dir_crys2**2)
      gcd_sum_sq_y_dir_crys1n2 = np.gcd(sum_sq_y_dir_crys1, sum_sq_y_dir_crys2)
      sum_sq_y_dir_crys1_gcd_norm = int( sum_sq_y_dir_crys1/
                                         gcd_sum_sq_y_dir_crys1n2 )
      sum_sq_y_dir_crys2_gcd_norm = int( sum_sq_y_dir_crys2/ 
                                         gcd_sum_sq_y_dir_crys1n2 )
      if not all( map(test_int, [sum_sq_y_dir_crys1_gcd_norm, 
                                 sum_sq_y_dir_crys2_gcd_norm] ) ):
         raise RuntimeError('The lattice spacings of crystal 1 and crystal 2 ' +
                            'along y direction are not in rational proportion')
      else:
         self.y_spacing_ratio = misc_crys.get_miller_indices(np.array(
                               [self.spacing_crys1[1], self.spacing_crys2[1]])**2)
         if not all( map(test_int, self.y_spacing_ratio) ):
            raise RuntimeError('Elements of the y_spacing_ratio at this stage '+
                               'must be squares of integers')
         else:
            self.y_spacing_ratio = np.sqrt(self.y_spacing_ratio).astype(int)[::-1]
         
      
    
################################################################

   def z_n_x_dirs_other_crystal(self):

      rot_axis = self.rot_axis/np.linalg.norm(self.rot_axis)
      epsilon = np.zeros((3,3,3))
      epsilon[0,1,2] = 1; epsilon[1,2,0] = 1; epsilon[2,0,1] = 1 
      epsilon[2,1,0] = -1; epsilon[0,2,1] = -1; epsilon[1,0,2] = -1
      self.rot_mat = ( ( np.cos(self.rot_angle)*np.eye(3) ) + 
                     ( ( 1 - np.cos(self.rot_angle) ) * np.outer(rot_axis,rot_axis) ) -
                     ( np.sin(self.rot_angle) * np.einsum('ijk,k->ij', epsilon, rot_axis) ) )
                     
      if not np.allclose(np.linalg.inv(self.rot_mat), self.rot_mat.T):
         print('Faulty rotation matrix:')
         print(self.rot_mat)
         raise RuntimeError('Inverse of a rotation matrix always equals its transpose')
    
      self.z_dir_crys2 = np.dot(self.rot_mat.T, self.z_dir_crys1)    
      self.x_dir_crys2 = np.dot(self.rot_mat.T, self.x_dir_crys1)
      self.y_dir_crys2 = np.cross(self.z_dir_crys2, self.x_dir_crys2)
      self.z_dir_crys2, self.x_dir_crys2, self.y_dir_crys2 = misc_crys.get_miller_indices(
            np.c_[self.z_dir_crys2, self.x_dir_crys2, self.y_dir_crys2].T)
      
      self.orient_crys2 = np.c_[self.x_dir_crys2, self.y_dir_crys2, self.z_dir_crys2].T
      
      self.multiplicity_crys2, self.spacing_crys2 = self.crys_struc.latticeInterplanarSpacingnDisplacements(
                                        self.orient_crys2, return_relative_displacements=False,
                                        return_multiplicity=True, return_lattice_spacing=True)[-2:] 
      
#################################################################


   def CSL_n_motifs(self, multiple_prim_vecs_pairs = False, verbose=False):
   
      # If multiple_prim_vecs_pairs == True, then it would likely be more computationally expensive

      interplanar_spacings = ( np.array([self.spacing_crys1[2], self.spacing_crys2[2]])/ 
                               np.array([self.multiplicity_crys1[2], self.multiplicity_crys2[2]]) )
      lat_vol = abs(np.dot(self.primitive_vecs[0], np.cross(self.primitive_vecs[1], self.primitive_vecs[2])))
      sublat_area = lat_vol/interplanar_spacings
      p_vecs_sublat = self.crys_struc.get_subLattice_primitive_vecs_routine2(np.c_[self.z_dir_crys1, self.z_dir_crys2].T)[0]
      p_vecs_sublat[1] = np.dot(self.rot_mat, p_vecs_sublat[1].T).T
      cross_prod_p_vecs_1 = np.cross(p_vecs_sublat[0,0,:], p_vecs_sublat[0,1,:])
      cross_prod_p_vecs_1 = cross_prod_p_vecs_1/np.linalg.norm(cross_prod_p_vecs_1)
      if not ( np.allclose([np.dot(p_vecs_sublat[1,0,:], cross_prod_p_vecs_1)/np.linalg.norm(p_vecs_sublat[1,0,:]), 
                            np.dot(p_vecs_sublat[1,1,:], cross_prod_p_vecs_1)/np.linalg.norm(p_vecs_sublat[1,1,:])], 
                           0.0, atol=1e-3) ):
         print(p_vecs_sublat)
         print(np.dot(p_vecs_sublat[1,0,:], np.cross(p_vecs_sublat[0,0,:], p_vecs_sublat[0,1,:])))
         print(np.dot(p_vecs_sublat[1,1,:], np.cross(p_vecs_sublat[0,0,:], p_vecs_sublat[0,1,:])))
         raise RuntimeError('The primitive vectors of the boundary planes are not coplanar yet!')
         
      vec1_norm = np.linalg.norm(p_vecs_sublat[0,0,:]); vec2_norm = np.linalg.norm(p_vecs_sublat[0,1,:])      
      vec3_norm = np.linalg.norm(p_vecs_sublat[1,0,:]); vec4_norm = np.linalg.norm(p_vecs_sublat[1,1,:])
      cos_12 = np.dot(p_vecs_sublat[0,0,:], p_vecs_sublat[0,1,:]) / vec1_norm / vec2_norm
      cos_13 = np.dot(p_vecs_sublat[0,0,:], p_vecs_sublat[1,0,:]) / vec1_norm / vec3_norm
      cos_14 = np.dot(p_vecs_sublat[0,0,:], p_vecs_sublat[1,1,:]) / vec1_norm / vec4_norm
      vec1_2D = np.array([1,0.]); vec2_2D = np.array([cos_12, np.sqrt(1-(cos_12**2))])
      vec3_2D = np.array([cos_13, np.sqrt(1-(cos_13**2))]); vec4_2D = np.array([cos_14, np.sqrt(1-(cos_14**2))])
      vec1_2D = vec1_2D*vec1_norm; vec2_2D = vec2_2D*vec2_norm
      vec3_2D = vec3_2D*vec3_norm; vec4_2D = vec4_2D*vec4_norm
      
      #Taking vec1_2D and vec2_2D as primitive vecs for the 2D lattice, we will now express
      #vec3_2D and vec4_2D in terms of the former vectors respectively and store them in the 
      #the rows of vec3_vec4_inprim in order.
      
      vec3_vec4_inprim = np.linalg.solve( np.c_[vec1_2D, vec2_2D], np.c_[vec3_2D, vec4_2D] ).T
      
      # Creating numpy ufuncs of functions of the fractions class
      frac_array = np.frompyfunc(lambda x: fractions.Fraction(x).limit_denominator(100), 1, 1)
      num_array = np.frompyfunc(lambda x: x.numerator, 1, 1)
      den_array = np.frompyfunc(lambda x: x.denominator, 1, 1)
      
      # Creating fractions of every component of coeff and 
      # extracting the corresponding numerators and denominators
      vec3_vec4_inprim_frac = frac_array(vec3_vec4_inprim)
      vec3_vec4_inprim_num = num_array(vec3_vec4_inprim_frac).astype(int)
      vec3_vec4_inprim_den = den_array(vec3_vec4_inprim_frac).astype(int)
      
      cx_1 = vec3_vec4_inprim_num[0,0]*vec3_vec4_inprim_den[1,0]
      cx_2 = vec3_vec4_inprim_num[1,0]*vec3_vec4_inprim_den[0,0]
      cx_3 = -vec3_vec4_inprim_den[0,0]*vec3_vec4_inprim_den[1,0]
      
      cy_1 = vec3_vec4_inprim_num[0,1]*vec3_vec4_inprim_den[1,1]
      cy_2 = vec3_vec4_inprim_num[1,1]*vec3_vec4_inprim_den[0,1]
      cy_3 = -vec3_vec4_inprim_den[0,1]*vec3_vec4_inprim_den[1,1]
      
      if (cx_3 == 0) or (cy_3 == 0):
         raise RuntimeError('The denominators cannot be zero')
      
      if ((cx_1 == 0) and (cx_2 == 0)) or ((cy_1 == 0) and (cy_2 == 0)):
         raise RuntimeError('The primitive vectors of one of the boundary planes (from the 2 crystals) are collinear')
         
      if ((cx_1 == 0) and (cy_1 == 0)) or ((cx_2 == 0) and (cy_2 == 0)):
         raise RuntimeError('One of the primitive vectors vec3_2D or vec4_2D is a zero vector... Weird!!!')
         
      x, y, z, w = sympy.symbols(['x', 'y', 'z', 'w'])
      
      t_0_vals = np.array([-3,-2,-1,0,1,2,3])
      
      if any(np.array([cx_1,cx_2]) == 0) and any(np.array([cy_1,cy_2]) == 0):
         if cx_1 == 0:
            res1 = diop_linear( (cx_2*y)-(cx_3*z) ); res2 = diop_linear( (cy_1*x)-(cy_3*z) )
         else:
            res1 = diop_linear( (cx_1*y)-(cx_3*z) ); res2 = diop_linear( (cy_2*x)-(cy_3*z) )
         free_sym = list(res1[0].free_symbols); free_sym.sort(key=str)
         res1_arr = np.array([res1.subs(list(zip(free_sym, (i)))) for i in t_0_vals for j in t_0_vals])
         res2_arr = np.array([res2.subs(list(zip(free_sym, (j)))) for i in t_0_vals for j in t_0_vals])
               
      elif any(np.array([cx_1,cx_2,cy_1,cy_2]) == 0):
         coeff_order = np.array([[0,1,2], [0,2,1], [1,0,2], [2,0,1], [1,2,0], [2,1,0]])
         coeff_arr = np.array([[cx_1,cx_2,cx_3], [cy_1,cy_2,cy_3]])
         zero_loc = np.where(coeff_arr[:,:-1] == 0)
         if zero_loc[0].item()==0:
            res1 = diop_linear( np.dot(coeff_arr[0], [x,y,z]), param='m' )
            free_sym1 = list(res1[0].free_symbols)
            res2_list = [Tuple(*np.array(diop_linear(np.dot(coeff_arr[1][c_ord], [x,y,z]), param='n'))[np.argsort(c_ord)]) 
                         for c_ord in coeff_order]
            free_syms2 = list(set.union(*[res2_val[count].free_symbols for res2_val in res2_list for count in [0,1,2]]))
            free_syms2.sort(key=str)
            res1_arr = []; res2_arr = []
            for res2_list_val in res2_list:
               res3_expr = res1[0] - res2_list_val[1-zero_loc[1].item()]
               res3_expr_coeffs = [res3_expr.coeff(count) for count in free_sym1+free_syms2]
               if len(res3_expr_coeffs) != 3:
                  raise RuntimeError('There must be three entries in res3_expr_coeffs when a primitive vector of '+
                                     'one crystal is parallel to a primitive vector of the other crystal')
               if res3_expr_coeffs[0] == 0:
                  raise RuntimeError('the coefficient for res1 cannot be zero')
               elif all(np.array(res3_expr_coeffs[1:]) == 0):
                  raise RuntimeError('all the coefficients for res2_list_val cannot be zero')
               elif any(np.array(res3_expr_coeffs[1:]) == 0):
                  res3 = diop_linear(res3_expr)
                  free_sym3 = np.array(list(res3[0].free_symbols)).item()
                  interm_list = [res3.subs(free_sym3, count) for count in t_0_vals]
                  res1_arr = res1_arr + [res1.subs(free_sym1[0], interm[0]) for interm in interm_list for count in t_0_vals]
                  some_ind = np.nonzero(res3_expr_coeffs[1:])[0].item()
                  res2_arr = res2_arr + [res2_list_val.subs([(free_syms2[some_ind], interm[1]), (free_syms2[1-some_ind], count)]) 
                                         for count in t_0_vals for interm in interm_list]
               else:
                  res3_list = [Tuple(*np.array(diop_linear(np.dot(np.array(res3_expr_coeffs)[c_ord], [x,y,z])))[np.argsort(c_ord)]) 
                               for c_ord in coeff_order]
                  free_sym3 = list(set.union(*[res3[count].free_symbols for res3 in res3_list for count in [0,1,2]]))
                  free_sym3.sort(key=str)
                  interm_list = [res3.subs(list(zip(free_sym3, count))) for res3 in res3_list for count in itertools.permutations(t_0_vals,2)]
                  res1_arr = res1_arr + [res1.subs(free_sym1[0], interm[0]) for interm in interm_list]
                  res2_arr = res2_arr + [res2_list_val.subs(list(zip(free_syms2, interm[1:]))) for interm in interm_list]
                  
         elif zero_loc[0].item()==1:
            res1_list = [Tuple(*np.array(diop_linear(np.dot(coeff_arr[0][c_ord], [x,y,z]), param='m'))[np.argsort(c_ord)]) 
                         for c_ord in coeff_order]
            free_syms1 = list(set.union(*[res1_val[count].free_symbols for res1_val in res1_list for count in [0,1,2]]))
            free_syms1.sort(key=str)
            res2 = diop_linear( np.dot(coeff_arr[1], [x,y,z]), param='n' )
            free_sym2 = list(res2[0].free_symbols)
            res1_arr = []; res2_arr = []
            for res1_list_val in res1_list:
               res3_expr = res1_list_val[1-zero_loc[1].item()] - res2[0] 
               res3_expr_coeffs = [res3_expr.coeff(count) for count in free_syms1+free_sym2]
               if len(res3_expr_coeffs) != 3:
                  raise RuntimeError('There must be three entries in res3_expr_coeffs when a primitive vector of '+
                                     'one crystal is parallel to a primitive vector of the other crystal')
               if res3_expr_coeffs[-1] == 0:
                  raise RuntimeError('the coefficient for res2 cannot be zero')
               elif all(np.array(res3_expr_coeffs[:-1]) == 0):
                  raise RuntimeError('all the coefficients for res1_list_val cannot be zero')
               elif any(np.array(res3_expr_coeffs[:-1]) == 0):
                  res3 = diop_linear(res3_expr)
                  free_sym3 = np.array(list(res3[0].free_symbols)).item()
                  interm_list = [res3.subs(free_sym3, count) for count in t_0_vals]
                  some_ind = np.nonzero(res3_expr_coeffs[:-1])[0].item()
                  res1_arr = res1_arr + [res1_list_val.subs([(free_syms1[some_ind], interm[0]), (free_syms1[1-some_ind], count)]) 
                                         for count in t_0_vals for interm in interm_list]
                  res2_arr = res2_arr + [res2.subs(free_sym2[0], interm[1]) for interm in interm_list for count in t_0_vals]                  
               else:
                  res3_list = [Tuple(*np.array(diop_linear(np.dot(np.array(res3_expr_coeffs)[c_ord], [x,y,z])))[np.argsort(c_ord)]) 
                               for c_ord in coeff_order]
                  free_sym3 = list(set.union(*[res3[count].free_symbols for res3 in res3_list for count in [0,1,2]]))
                  free_sym3.sort(key=str)
                  interm_list = [res3.subs(list(zip(free_sym3, count))) for res3 in res3_list for count in itertools.permutations(t_0_vals,2)]
                  res1_arr = res1_arr + [res1_list_val.subs(list(zip(free_syms1, interm[:-1]))) for interm in interm_list]
                  res2_arr = res2_arr + [res2.subs(free_sym2[0], interm[1]) for interm in interm_list]
         res1_arr = np.array(res1_arr); res2_arr = np.array(res2_arr) 
                  
      else:
         coeff_order = np.array([[0,1,2], [0,2,1], [1,0,2], [2,0,1], [1,2,0], [2,1,0]])
         coeff_order4 = np.array(list(itertools.permutations([0,1,2,3])))
         coeff_arr = np.array([[cx_1,cx_2,cx_3], [cy_1,cy_2,cy_3]])
         res1_list = [Tuple(*np.array(diop_linear(np.dot(coeff_arr[0][c_ord], [x,y,z]), param='m'))[np.argsort(c_ord)]) 
                      for c_ord in coeff_order]
         free_syms1 = list(set.union(*[res1_val[count].free_symbols for res1_val in res1_list for count in [0,1,2]]))
         free_syms1.sort(key=str)
         res2_list = [Tuple(*np.array(diop_linear(np.dot(coeff_arr[1][c_ord], [x,y,z]), param='n'))[np.argsort(c_ord)]) 
                      for c_ord in coeff_order]
         free_syms2 = list(set.union(*[res2_val[count].free_symbols for res2_val in res2_list for count in [0,1,2]]))
         free_syms2.sort(key=str)
         res1_arr = []; res2_arr = []
         for res1_list_val in res1_list:
            for res2_list_val in res2_list:
               res3_expr = res1_list_val[1] - res2_list_val[1]
               res3_expr_coeffs = [res3_expr.coeff(count) for count in free_syms1+free_syms2]
               if len(res3_expr_coeffs) != 4:
                  raise RuntimeError('There must be four entries in res3_expr_coeffs when the primitive vectors of '+
                                     'the two intersecting crystals are not parallel to each other')
               if all(np.array(res3_expr_coeffs[:-2]) == 0):
                  raise RuntimeError('all the coefficients for res1_list_val cannot be zero')
               elif all(np.array(res3_expr_coeffs[2:]) == 0):
                  raise RuntimeError('all the coefficients for res2_list_val cannot be zero')
               elif any(np.array(res3_expr_coeffs[:-2]) == 0) and any(np.array(res3_expr_coeffs[2:]) == 0):
                  res3 = diop_linear(res3_expr)
                  free_sym3 = np.array(list(res3[0].free_symbols)).item()
                  interm_list1 = [res3.subs(free_sym3, count) for count in t_0_vals]
                  some_ind1 = np.nonzero(res3_expr_coeffs[:-2])[0].item()
                  some_ind2 = np.nonzero(res3_expr_coeffs[2:])[0].item()
                  interm_res1_arr = [res1_list_val.subs(free_syms1[some_ind1], interm[0]) for interm in interm_list1]
                  interm_res2_arr = [res2_list_val.subs(free_syms2[some_ind2], interm[1]) for interm in interm_list1]
                  for i in np.arange(len(interm_list1)):
                     res3_2 = diop_linear(interm_res1_arr[i][0]-interm_res2_arr[i][0])
                     free_sym3_2 = np.array(list(res3_2[0].free_symbols)).item()
                     interm_list2 = [res3_2.subs(free_sym3_2, count) for count in t_0_vals]
                     res1_arr = res1_arr + [interm_res1_arr[i].subs(free_syms1[1-some_ind1], interm[0]) for interm in interm_list2]
                     res2_arr = res2_arr + [interm_res2_arr[i].subs(free_syms2[1-some_ind2], interm[1]) for interm in interm_list2]                     
               elif any(np.array(res3_expr_coeffs[:-2]) == 0):
                  res3_list = [Tuple(*np.array(diop_linear(np.dot(np.array(res3_expr_coeffs)[np.nonzero(res3_expr_coeffs)][c_ord], [x,y,z])))[np.argsort(c_ord)]) 
                               for c_ord in coeff_order]
                  free_sym3 = list(set.union(*[res3[count].free_symbols for res3 in res3_list for count in [0,1,2]]))
                  free_sym3.sort(key=str)
                  interm_list = [res3.subs(list(zip(free_sym3, count))) for res3 in res3_list for count in itertools.permutations(t_0_vals,2)]
                  some_ind = np.nonzero(res3_expr_coeffs[:-2])[0].item()
                  interm_res1_arr = [res1_list_val.subs(free_syms1[some_ind], interm[0]) for interm in interm_list]
                  interm_res2_arr = [res2_list_val.subs(list(zip(free_syms2, interm[1:]))) for interm in interm_list]
                  for i in np.arange(len(interm_list)):
                     res3_2 = diop_linear(interm_res1_arr[i][0]-interm_res2_arr[i][0])[0]
                     if res3_2 is None:
                        continue
                     else:
                        res3_2 = int(res3_2)
                     res1_arr = res1_arr + [interm_res1_arr[i].subs(free_syms1[1-some_ind], res3_2)]
                     res2_arr = res2_arr + [interm_res2_arr[i]]
               elif any(np.array(res3_expr_coeffs[2:]) == 0):
                  res3_list = [Tuple(*np.array(diop_linear(np.dot(np.array(res3_expr_coeffs)[np.nonzero(res3_expr_coeffs)][c_ord], [x,y,z])))[np.argsort(c_ord)]) 
                               for c_ord in coeff_order]
                  free_sym3 = list(set.union(*[res3[count].free_symbols for res3 in res3_list for count in [0,1,2]]))
                  free_sym3.sort(key=str)
                  interm_list = [res3.subs(list(zip(free_sym3, count))) for res3 in res3_list for count in itertools.permutations(t_0_vals,2)]
                  interm_res1_arr = [res1_list_val.subs(list(zip(free_syms1, interm[:-1]))) for interm in interm_list]
                  some_ind = np.nonzero(res3_expr_coeffs[2:])[0].item()
                  interm_res2_arr = [res2_list_val.subs(free_syms2[some_ind], interm[-1]) for interm in interm_list]
                  for i in np.arange(len(interm_list)):
                     res3_2 = diop_linear(interm_res1_arr[i][0]-interm_res2_arr[i][0])[0]
                     if res3_2 is None:
                        continue
                     else:
                        res3_2 = int(res3_2)
                     res1_arr = res1_arr + [interm_res1_arr[i]]
                     res2_arr = res2_arr + [interm_res2_arr[i].subs(free_syms2[1-some_ind], res3_2)]
               else:
                  res3_list = [Tuple(*np.array(diop_linear(np.dot(np.array(res3_expr_coeffs)[c_ord], [x,y,z,w])))[np.argsort(c_ord)]) 
                               for c_ord in coeff_order4]
                  free_sym3 = list(set.union(*[res3[count].free_symbols for res3 in res3_list for count in [0,1,2,3]]))
                  free_sym3.sort(key=str)
                  interm_list = [res3.subs(list(zip(free_sym3, count))) for res3 in res3_list for count in itertools.permutations(t_0_vals,3)]
                  res1_arr = res1_arr + [res1_list_val.subs(list(zip(free_syms1, interm[:-2]))) for interm in interm_list]
                  res2_arr = res2_arr + [res2_list_val.subs(list(zip(free_syms2, interm[2:]))) for interm in interm_list]  
         res1_arr = np.array(res1_arr).astype(int); res2_arr = np.array(res2_arr).astype(int) 
         
      int_arr0 = np.c_[res1_arr[:,-1], res2_arr[:,-1]]
      int_arr0 = int_arr0[np.any(int_arr0 != 0, axis=1)]
      int_arr_gcd = np.gcd.reduce(int_arr0, axis=1)
      int_arr_gcd_divided = ((int_arr0.T/ int_arr_gcd).T).astype(int)
      unique_dirs, unique_dirs_inv_inds = np.unique(int_arr_gcd_divided, return_inverse=True, axis=0)
      int_arr = []
      for count, ud in enumerate(unique_dirs):
         int_arr_gcd1 = int_arr_gcd[unique_dirs_inv_inds==count]
         if not np.all(int_arr_gcd1 % np.min(int_arr_gcd1) == 0):
            raise RuntimeError("Coincident sites must form a lattice")
         else:
            int_arr = int_arr + [ud*np.min(int_arr_gcd1)]
      int_arr = np.array(int_arr)
      
      if int_arr.dtype != int:
         raise ValueError('"int_arr" must be an integer array')
         
      if verbose:
         print('We have selected {0} independent coincident lattice vectors'.format(len(int_arr)))
         print('We will now browse through {0} combinations of pairs of those lattice vectors '.format(int(comb(len(int_arr),2)))+
               'to find the pair(s) of vectors which has the minimum area ---> those pairs will form the CSL')  
                     
      poss_vecs_2D = np.dot( np.c_[vec1_2D, vec2_2D], int_arr.T ).T
      poss_vecs_3D = np.dot( p_vecs_sublat[0,:,:].T, int_arr.T ).T
      
      comb_inds = np.array(list(itertools.combinations(np.arange(len(poss_vecs_2D)), 2)))
      comb_inds = comb_inds[np.invert(np.all(np.isclose(poss_vecs_2D[comb_inds[:,0]], -poss_vecs_2D[comb_inds[:,1]]), axis=1))]
      tmp_area = abs(np.cross(poss_vecs_2D[comb_inds[:,0]], poss_vecs_2D[comb_inds[:,1]], axis=1))
      CSL_area = np.min(tmp_area)
      sel_pairs = comb_inds[np.isclose(tmp_area, CSL_area)]
      CSL_prim_vecs_2D = poss_vecs_2D[sel_pairs]
      CSL_prim_vecs_3D_crys1 = poss_vecs_3D[sel_pairs]
      CSL_prim_vecs_3D_crys2 = np.dot(self.rot_mat.T, 
                                      CSL_prim_vecs_3D_crys1.reshape(
                                      len(CSL_prim_vecs_3D_crys1)*2, 3).T).T.reshape(-1,2,3)
      if np.isclose(CSL_area, np.max(sublat_area)):
         pass
      elif CSL_area < np.max(sublat_area):
         print('Faulty CSL area = {0} which is less than the '.format(CSL_area)+
               'maximum of the individual sublattice areas = {0}'.format(np.max(sublat_area)))
         print('Corresponding CSL primitive vector pairs in 2D = {0}\n'.format(CSL_prim_vecs_2D)+
               'and in 3D (for crys1) = {0}\n'.format(CSL_prim_vecs_3D_crys1)+
               'and in 3D (for crys2) = {0}'.format(CSL_prim_vecs_3D_crys2))
         raise RuntimeError('CSL area cannot be less than the maximum of the individual sublattice areas')
      
      if not multiple_prim_vecs_pairs:
         cos_pair = ( np.sum(CSL_prim_vecs_2D[:,0,:] * CSL_prim_vecs_2D[:,1,:], axis=1) / 
                      np.linalg.norm(CSL_prim_vecs_2D[:,0,:], axis=1) /
                      np.linalg.norm(CSL_prim_vecs_2D[:,1,:], axis=1) )
         sel_pair = np.isclose(cos_pair, np.min(abs(cos_pair)))
         
         CSL_prim_vecs_2D = CSL_prim_vecs_2D[sel_pair]
         CSL_prim_vecs_3D_crys1 = CSL_prim_vecs_3D_crys1[sel_pair]
         CSL_prim_vecs_3D_crys2 = CSL_prim_vecs_3D_crys2[sel_pair]
         
         if np.count_nonzero(sel_pair) != 1:
            sel_pair = ( ( (np.sum(CSL_prim_vecs_3D_crys1[:,0,:]*self.x_dir_crys1, axis=1) > 0) |
                           (np.sum(CSL_prim_vecs_3D_crys1[:,0,:]*self.y_dir_crys1, axis=1) > 0) ) &
                         ( (np.sum(CSL_prim_vecs_3D_crys1[:,1,:]*self.x_dir_crys1, axis=1) > 0) |
                           (np.sum(CSL_prim_vecs_3D_crys1[:,1,:]*self.y_dir_crys1, axis=1) > 0) ) )
            if np.count_nonzero(sel_pair) == 0:
               CSL_prim_vecs_2D = CSL_prim_vecs_2D[0]
               CSL_prim_vecs_3D_crys1 = CSL_prim_vecs_3D_crys1[0]
               CSL_prim_vecs_3D_crys2 = CSL_prim_vecs_3D_crys2[0]
            else:
               CSL_prim_vecs_2D = CSL_prim_vecs_2D[sel_pair][0]
               CSL_prim_vecs_3D_crys1 = CSL_prim_vecs_3D_crys1[sel_pair][0]
               CSL_prim_vecs_3D_crys2 = CSL_prim_vecs_3D_crys2[sel_pair][0]
      
         
      self.CSL_prim_vecs_2D =  CSL_prim_vecs_2D
      self.CSL_prim_vecs_3D_crys1 =  CSL_prim_vecs_3D_crys1
      self.CSL_prim_vecs_3D_crys2 =  CSL_prim_vecs_3D_crys2
      self.CSL_area = CSL_area
      
#################################################################      

   def write_info(self, filename, folder = './'):

      if not isinstance(folder, str):
         raise ValueError('"folder" should be a string input')
      elif folder[-1] != '/':
         raise ValueError('"folder" should end with a "/"')
      if not isinstance(filename, str):
         raise ValueError('"filename" should be a string input')
      if not os.path.isdir(folder):
         os.makedirs(folder)
      f = open(folder+filename, 'w')
      f.writelines(['############ Grain boundary information file ############\n\n\n']
                 + ['****** Crystal orientation and boundary plane ******\n\n']
                 + ['(rotation axis, rotation angle) = '+
                    '({0}, {1:.2f} deg) \n'.format(self.rot_axis, np.rad2deg(self.rot_angle))]
                 + ['(boundary plane normal of crystal 1, boundary plane normal of crystal 2) = '+
                    '({0}, {1}) \n'.format(self.z_dir_crys1, self.z_dir_crys2)]
                 + ['*Note: The "rotation axis" and the "boundary plane normal of crystal 1" are '+
                    'defined with respect to the same orthonormal basis (say B1) \n']
                 + ['*Note: The "boundary plane normal of crystal 2" is defined with respect to a '+
                    'orthonormal basis (say B2), which is rotated with respect to B1 about "rotation axis" '+
                    'by "rotation angle" \n\n']
                 + ['-- Orthogonal directions on the boundary plane (User-specified, arbitrary) --\n']
                 + ['x: {0} with respect to B1\n'.format(self.x_dir_crys1)+
                    '   {0} with respect to B2\n'.format(self.x_dir_crys2)]
                 + ['y: {0} with respect to B1\n'.format(self.y_dir_crys1)+
                    '   {0} with respect to B2\n\n\n'.format(self.y_dir_crys2)]
                 + ['****** Crystallographic information ******\n\n']
                 + ['{0} lattice\n'.format(self.lat_name)]
                 + ['Primitive vectors = [{0},\n'.format(self.primitive_vecs[0])+
                    '                     {0},\n'.format(self.primitive_vecs[1])+
                    '                     {0}]\n'.format(self.primitive_vecs[2])]
                 + ['*Note: Both the crystals on either side of the boundary plane belong to the same lattice \n\n']
                 + ['-- Lattice spacing and multiplicity along different directions --\n']
                 + ['x{0}: lattice spacing = {1}, multiplicity = {2}\n'.format(self.x_dir_crys1, self.spacing_crys1[0], self.multiplicity_crys1[0])]
                 + ['x{0}: lattice spacing = {1}, multiplicity = {2}\n'.format(self.x_dir_crys2, self.spacing_crys2[0], self.multiplicity_crys2[0])]
                 + ['y{0}: lattice spacing = {1}, multiplicity = {2}\n'.format(self.y_dir_crys1, self.spacing_crys1[1], self.multiplicity_crys1[1])]
                 + ['y{0}: lattice spacing = {1}, multiplicity = {2}\n'.format(self.y_dir_crys2, self.spacing_crys2[1], self.multiplicity_crys2[1])]
                 + ['z{0}: lattice spacing = {1}, multiplicity = {2}\n'.format(self.z_dir_crys1, self.spacing_crys1[2], self.multiplicity_crys1[2])]
                 + ['z{0}: lattice spacing = {1}, multiplicity = {2}\n'.format(self.z_dir_crys2, self.spacing_crys2[2], self.multiplicity_crys2[2])]
                 + ['*Note: All lattice spacings are expressed in terms of lattice scaling \n\n']
                 + ['** {0} lattice spacing along x{1} for crystal 1'.format(self.x_spacing_ratio[0], self.x_dir_crys1)+
                    ' equals {0} lattice spacing along x{1} for crystal 2\n'.format(self.x_spacing_ratio[1], self.x_dir_crys2)]
                 + ['** {0} lattice spacing along y{1} for crystal 1'.format(self.y_spacing_ratio[0], self.y_dir_crys1)+
                    ' equals {0} lattice spacing along y{1} for crystal 2'.format(self.y_spacing_ratio[1], self.y_dir_crys2)]
                  )
                  
      f.close()
               

   
#################################################################
      
      
   def __setattr__(self, name, value):
      self.__dict__[name] = value
      
      
#################################################################


