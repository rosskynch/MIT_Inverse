/*
 *    An MIT forward solver code based on the deal.II (www.dealii.org) library.
 *    Copyright (C) 2013-2015 Ross Kynch & Paul Ledger, Swansea Unversity.
 * 
 *    This program is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program.  If not, see <http://www.gnu.org/licenses/>.    
*/

#ifndef BACKGROUNDFIELD_H
#define BACKGROUNDFIELD_H

// deal.II includes:
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>

// std includes:
#include <complex>
#include <algorithm>
#include <math.h>

// boost includes:
#include <boost/math/special_functions/legendre.hpp>
#include <boost/math/special_functions/bessel.hpp>

// complex bessel library:
#include <complex_bessel.h>

// My includes:
#include <curlfunction.h>
#include <all_data.h>
#include <myvectortools.h>

using namespace dealii;


// TODO: Make all of the functions of type EddyCurrentFunction.
//       Will need to add a few extra functions (e.g. the zero_xxx boolean functions).

namespace backgroundField
{
  /* REMOVED FOR NOW, NOT NEEDED
  // TODO: Move to conductingSphereDipole class.
  template<int dim>
  std_cx11::array<double, dim> cartesian_to_spherical(Point<dim> &position)
  
  template<int dim>
  Point<dim> spherical_to_cartesian(std_cx11::array<double, dim> &position)
  */
  
  // Dipole Field as a source term.
  // The source is communicated via the RHS of the equation
  // and use zero far field conditions.
  template<int dim>
  struct DipoleAsSourceData
  {
    double coil_radius;
    double current;
    Tensor<1,dim> coil_direction;
    Point<dim> coil_position;
  };
  template<int dim>
  class DipoleAsSource : public EddyCurrentFunction<dim>
  {
  public:
    DipoleAsSource(const DipoleAsSourceData<dim> &data);
    
    void rhs_value_list (const std::vector<Point<dim> > &points,
                         std::vector<Vector<double> >   &values,
                         const types::material_id &mat_id) const;

    bool zero_rhs() const {return false;}
  private:
    const DipoleAsSourceData<dim> data;
    double current_factor;
  };

  // Dipole field centred at a point
  template<int dim>
  class DipoleSource : public EddyCurrentFunction<dim>
  {
  public:
    DipoleSource(const Point<dim> &input_source_point,
                 const Tensor<1, dim> &input_coil_direction);
        
    void vector_value_list (const std::vector<Point<dim> > &points,
                            std::vector<Vector<double> >   &values,
                            const types::material_id &mat_id) const;
        
    void curl_value_list (const std::vector<Point<dim> > &points,
                          std::vector<Vector<double> >   &values,
                          const types::material_id &mat_id) const;
                          
    bool zero_vector() const { return false;}
    bool zero_curl() const { return false;}
                                        
  private:
    Point<dim> source_point;
    Tensor<1, dim> coil_direction;
  };
  
  // Field for a conducting sphere in a uniform background field.
  // includes a function to compute the perturbed field.
  // First, a data structure to store the key parameters:
  struct conductingSphereData
  {
    double sphere_radius;
    double sigma;
    double mu_c;
    double mu_n;
    double omega;
    
    std::vector<Vector<double>> uniform_field;
  };
  // Class
  template<int dim>
  class conductingSphere : public EddyCurrentFunction<dim>
  {
  public:
    // Constructor
    conductingSphere(const conductingSphereData &data_in);
    
    // Copy constructor
    conductingSphere(const conductingSphere<dim> &source);
    
    // Assignment operator
    conductingSphere& operator= (const conductingSphere<dim> &source);
        
    void vector_value_list (const std::vector<Point<dim> > &points,
                            std::vector<Vector<double> >   &value_list,
                            const types::material_id &mat_id) const;
        
    void curl_value_list (const std::vector<Point<dim> > &points,
                          std::vector<Vector<double> >   &value_list,
                          const types::material_id &mat_id) const;
                          
    void perturbed_field_value_list (const std::vector< Point<dim> > &points,
                                     std::vector<Vector<double> > &value_list,
                                     const types::material_id &mat_id) const;
    
    void check_spherical_coordinates(const std::vector< Point<dim> > &points,
                                     std::vector<Vector<double> > &value_list) const;
                                     
    bool zero_vector() const { return false;}
    bool zero_curl() const { return false;}
    bool zero_perturbed() const { return false;}
                                        
  private:
    conductingSphereData data;
    double constant_p;
    std::complex<double> constant_C;
    std::complex<double> constant_D;
    double constant_B_magnitude;
  };
  
  // Field for a conducting sphere with a dipole-generated field.
  // includes a function to compute the perturbed field.
  // First, a data structure to store the key parameters:
  template <int dim>
  struct conductingSphereDipoleData
  {
    // Geometry info:
    // Note, the values of h0, r0 and theta0 all derive from coil_radius and coil_centre.
    // So they must be handled carefully when setting up this struct.
    double sphere_radius;
    double coil_radius;
    Tensor<1,dim> coil_direction; // direction of coil from centre of sphere
    double theta0; // angular span of the coil.
    double h0; // radial distance from sphere centre to coil centre.
    double r0; // radial distance from sphere centre to outer edge of coil.
    
    // Material info:
    double sigma; // Conductivity of sphere.
    double mu; // Permeability of sphere & background.
    double omega; // angular frequency.
    double I0; // current through the coil.
  };
  // Class
  template<int dim>
  class conductingSphereDipole : public EddyCurrentFunction<dim>
  {
  public:
    // Constructor
    conductingSphereDipole(const conductingSphereDipoleData<dim> &data_in);
    
    // Copy constructor
    conductingSphereDipole(const conductingSphereDipole<dim> &source);
    
    // Assignment operator
    conductingSphereDipole& operator= (const conductingSphereDipole<dim> &source);
        
    void vector_value_list (const std::vector<Point<dim> > &points,
                            std::vector<Vector<double> >   &value_list,
                            const types::material_id &mat_id) const;
                            
    void scattered_field_value_list (const std::vector<Point<dim> > &points,
                                     std::vector<Vector<double> >   &value_list,
                                     const types::material_id &mat_id) const;

    bool zero_vector() const { return false;}
    bool zero_scattered() const { return false;}
  private:
    conductingSphereDipoleData<dim> data;
    std::complex<double> constant_alpha;
    std::complex<double> constant_gamma;
    
    // Functions needed.
    unsigned int factorial(const unsigned int &n) const;
    double calcTHETA_n (const unsigned int &n,
                        const double &theta) const;
    std::complex<double> calc_bessel_n_plus_half(const unsigned int &n,
                                                 const std::complex<double> &x) const;
    // Rotational information:
    bool use_rotation = false;
    Tensor<1,dim> rotation_axis;
    double rotation_angle;
    FullMatrix<double> rotate_forward;
  };
  
  // Copy of the above but ignores the scattered field due to the sphere.
  template<int dim>
  class SphericalDipole : public EddyCurrentFunction<dim>
  {
  public:
    // Constructor
    SphericalDipole(const conductingSphereDipoleData<dim> &data_in);
    
    // Copy constructor
    SphericalDipole(const SphericalDipole<dim> &source);
    
    // Assignment operator
    SphericalDipole& operator= (const SphericalDipole<dim> &source);
        
    void vector_value_list (const std::vector<Point<dim> > &points,
                            std::vector<Vector<double> >   &value_list,
                            const types::material_id &mat_id) const;

    bool zero_vector() const { return false;}
  private:
    conductingSphereDipoleData<dim> data;
    std::complex<double> constant_alpha;
    std::complex<double> constant_gamma;
    
    // Functions needed.
    double calcTHETA_n (const unsigned int &n,
                        const double &theta) const;
    // Rotational information:
    bool use_rotation = false;
    Tensor<1,dim> rotation_axis;
    double rotation_angle;
    FullMatrix<double> rotate_forward;
  };
  
  // Field for a conducting object in a uniform background field.
  // Specifically, for an object where the polarization tensor is known.
  // This allows us to output values via the perturbed_field_value_list, 
  // using the pertubation tensor which must be read in as input.
  // Takes as input:
  // - a uniform field (3 real, 3 imag values), so [0][:] = real part.
  //                                               [1][:] = imag part.
  // - a polarization tensor [2][3][3], so [0][:][:] = real part
  //                                       [1][:][:] = imag part.
  template<int dim>
  class conductingObject_polarization_tensor : public EddyCurrentFunction<dim>
  {
  public:
    conductingObject_polarization_tensor(const std::vector<Vector<double> > &uniform_field,
                                         const std::vector<FullMatrix<double> > &polarizationTensor);
    
    void curl_value_list (const std::vector<Point<dim> > &points,
                          std::vector<Vector<double> >   &value_list,
                          const types::material_id &mat_id) const;

    void perturbed_field_value_list (const std::vector< Point<dim> > &points,
                                     std::vector<Vector<double> > &value_list,
                                     const types::material_id &mat_id) const;

    bool zero_curl() const { return false;}
    bool zero_perturbed() const { return false;}
                                        
  private:
    const std::vector< Vector<double> > uniform_field;
    const std::vector<FullMatrix<double> > polarizationTensor;
  };
  // Wave propagation
  // E = p*exp(i*k*x), x in R^3
  //
  // with p orthogonal to k, omega = |k| & |p|=1.
  template<int dim>
  class WavePropagation : public EddyCurrentFunction<dim> 
  {
  public:
    WavePropagation(Vector<double> &k_wave,
                    Vector<double> &p_wave);
    
    void vector_value_list (const std::vector<Point<dim> > &points,
                            std::vector<Vector<double> >   &values,
                            const types::material_id &mat_id) const;
                            

    void curl_value_list(const std::vector<Point<dim> > &points,
                         std::vector<Vector<double> > &value_list,
                         const types::material_id &mat_id) const;
                         
    bool zero_vector() const { return false;}
    bool zero_curl() const { return false;}
    
  private:  
    Vector<double> k_wave;
    Vector<double> p_wave;
  };
  
  // Function for testing only.
  // Solution is:
  // A = (x^2, y^2, z^2).
  // curlA = (0, 0, 0).
  template<int dim>
  class polynomialTest : public EddyCurrentFunction<dim>
  {
  public:
    void vector_value_list (const std::vector<Point<dim> > &points,
                            std::vector<Vector<double> >   &values,
                            const types::material_id &mat_id) const;

    void rhs_value_list (const std::vector<Point<dim> > &points,
                         std::vector<Vector<double> > &value_list,
                         const types::material_id &mat_id) const;
    
    bool zero_vector() const { return false;}
    bool zero_rhs() const { return false;}
  private:  
    Vector<double> k_wave;
    Vector<double> p_wave;
  };

  // Very simple uniform background
  // This returns zero in the vector_value part
  // and the uniform field specified in the curl_value part.
  template<int dim>
  class curlUniformField : public EddyCurrentFunction<dim>
  {
  public:
    curlUniformField(const std::vector< Vector<double> > &uniform_field);

    void vector_value_list (const std::vector<Point<dim> > &points,
                            std::vector<Vector<double> >   &value_list) const;

    void curl_value_list (const std::vector<Point<dim> > &points,
                          std::vector<Vector<double> >   &value_list) const;
                          
    bool zero_rhs() const { return false;}

  private:
    const std::vector< Vector<double> > uniform_field;
  };
  
  
  // Class for the team benchmark 7 problem. Only non-zero part is the
  // RHS, the BCs are zero for both Dirichlet and Neumann.
  template<int dim>
  class TEAMBenchmark7 : public EddyCurrentFunction<dim>
  {
  public:
    TEAMBenchmark7(const Point<dim> &coil_centre,
                   const std::vector<Point<dim>> &corner_centres,
                   const types::material_id coil_mat_id);

    void rhs_value_list (const std::vector<Point<dim> > &points,
                         std::vector<Vector<double> >   &values,
                         const types::material_id &mat_id) const;
                         
    bool zero_rhs() const { return false;}
  private:
    const double current_magnitude = 1.0968e6;
    Point<dim> coil_centre;
    std::vector<Point<dim>> corner_centres;
    
    types::material_id coil_mat_id;
    
    unsigned int get_quadrant(const Point<dim> &p) const;
    Point<dim> get_tangent_by_quadrant(const unsigned int &coil_quadrant,
                                       const unsigned int &corner_quadrant,
                                       const double &corner_angle) const;
  };
}
#endif