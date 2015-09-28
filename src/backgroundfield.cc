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

#include <backgroundfield.h>

using namespace dealii;

namespace backgroundField
{
  /* REMOVED FOR NOW, NOT NEEDED
  // TODO: Move inside the conductingspheredipole class.
  // Function returning spherical polar co-ords for cartesian input:
  // For dim==2, will return only the r and phi components.
  template<int dim>
  std_cx11::array<double, dim> cartesian_to_spherical(Point<dim> &position)
  {
    // Convert (x,y,z) to (r,theta,phi), spherical polars.
    std_cx11::array<double,dim> output;
    output[0] = sqrt(position.square()); // r
    output[1] = atan2(position(1),position(0); // phi
    if (dim == 3)
    {
      output[2] = acos((position(2)/output[0])); // theta
    }
    return output;
  }
  
  // Function returning cartesian co-ords for spherical polar input:
  template<int dim>
  Point<dim> spherical_to_cartesian(std_cx11::array<double, dim> &position)
  {
    Point<dim> output;
    
    output[0] = position[0]*sin(position[2])*cos(position[1]); 
    output[0] = position[0]*sin(position[2])*sin(position[1]);
    if (dim ==3)
    {
      output[2] = position[0]*cos(position[2]);
    }
      
    
    return output;
  }
  */
  
  // DIPOLEASSOURCE
  template<int dim>
  DipoleAsSource<dim>::DipoleAsSource (const DipoleAsSourceData<dim> &data)
  :
  data(data)
  {
    current_factor = -15.0*data.current*numbers::PI*data.coil_radius*data.coil_radius;
  }
  
  template<int dim>
  void DipoleAsSource<dim>::rhs_value_list (const std::vector<Point<dim> > &points,
                                            std::vector<Vector<double> >   &value_list,
                                            const types::material_id &mat_id) const
  {
    Assert(value_list.size() == points.size(), ExcDimensionMismatch(value_list.size(), points.size()));

    for (unsigned int k=0; k<points.size(); ++k)
    {
      // Note: Under the latest deal.II version, it should possible to handle this all with Tensor<x,dim>.
      //       However, not 100% sure the Tensor updates are all ready in current version, so bypass with
      //       Vector<double> & FullMatrix where needed.
      const Point<dim> &p = points[k];
      const Point<dim> shifted_point = Point<dim> (p - data.coil_position);
      const double rad = p.distance(data.coil_position);
      
      // avoid singularity at centre of coil
      if (rad < 1e-5)
      {
        value_list[k] = 0.;
      }
      else
      {
        // first calculate \hat{r}
        Vector<double> rhat_vector(dim);
        Tensor<1,dim> rhat_tensor(dim);
        for (unsigned int d=0; d<dim; ++d)
        {
          rhat_vector(d) = shifted_point(d)/rad;
          rhat_tensor[d] = rhat_vector(d);
        }
        
        // calculate the matrix due to \hat{r} \otimes \hat{r}
        FullMatrix<double> rhat_outer_product;
        rhat_outer_product.outer_product(rhat_vector, rhat_vector);
        
        // calculate the result of (\hat{r} \otimes \hat{r})*m
        Vector<double> temp_vector(dim);
        Vector<double> coil_direction(dim);
        for (unsigned int d=0; d<dim; ++d)
        {
          coil_direction(d) = data.coil_direction[d];
        }
        
        // NOTE: the direction is unit valued, the scaling factor is incorporated into current_factor.
        rhat_outer_product.vmult(temp_vector,
                                 coil_direction);
        // convert the vector to Tensor so it can be used with cross_product
        Tensor<1,dim> temp_tensor;
        for (unsigned int d=0; d<dim; ++d)
        {
          temp_tensor[d] = temp_vector(d);
        }
        // calculate the cross product \hat{r} \times (\hat{r} \otimes \hat{r})*m
        Tensor<1,dim> crossproductresult;
        cross_product(crossproductresult, rhat_tensor, temp_tensor);
        // Finally transfer over to output list.
        for (unsigned int d=0; d<dim; ++d)
        {
          value_list[k](d) = crossproductresult[d]*current_factor/(rad*rad*rad*rad);
          // zero imaginary part.
          value_list[k](d+dim) = 0.;
        }
      }
    }
  }
  template class DipoleAsSource<3>;
  
  // DIPOLESOURCE
  template<int dim>
  DipoleSource<dim>::DipoleSource(const Point<dim> &input_source_point,
                                  const Tensor<1, dim> &input_coil_direction)
  :
  source_point(input_source_point),
  coil_direction(input_coil_direction)
  {}
  
  // members:
  template <int dim>
  void DipoleSource<dim>::vector_value_list (const std::vector<Point<dim> > &points,
                                             std::vector<Vector<double> > &value_list,
                                             const types::material_id &mat_id) const
  {
    Assert(value_list.size() == points.size(), ExcDimensionMismatch(value_list.size(), points.size()));
        
    Tensor<1,dim> shifted_point;
    Tensor<1,dim> result; 
    for (unsigned int k=0; k<points.size(); ++k)
    {
      const Point<dim> &p = points[k];
      /* Work out the vector (stored as a tensor so we can use cross_product)
       * from the source point to the current point, p
       */
      for (unsigned int i = 0; i < dim; ++i)
      {
        shifted_point[i] = p(i) - source_point(i);
      }
      double rad = p.distance(source_point);
      double factor = EquationData::constant_mu0*1.0/(4.0*numbers::PI*rad*rad*rad);
      
      cross_product(result, coil_direction, shifted_point);
      result *= factor;
      for (unsigned int i = 0; i < dim; ++i)
      {
        // Real
        value_list[k](i) = result[i];
        // Imaginary
        value_list[k](i+dim) = 0.0;
      }
    }
  }
  template <int dim>
  void DipoleSource<dim>::curl_value_list (const std::vector<Point<dim> > &points,
                                           std::vector<Vector<double> > &value_list,
                                           const types::material_id &mat_id) const
  {
    Assert(value_list.size() == points.size(), ExcDimensionMismatch(value_list.size(), points.size()));
    
    FullMatrix<double> rhat(dim);
    FullMatrix<double> D2G(dim);
    FullMatrix<double> eye(IdentityMatrix(3));
    Vector<double> result(dim);
    // create a vector-version of the tensor coil_direction.
    // There may be a far better way to deal with this.... (TODO)
    // e.g. Use Tensor<2,dim> for the matrices instead, making the whole thing more tensor based.
    Vector<double> coil_direction(dim);
    for (unsigned int i = 0; i < dim; ++i)
    {
      coil_direction(i) = coil_direction[i];
    }

    Tensor<1,dim> shifted_point;
    Vector<double> scaled_vector(dim);

    for (unsigned int k=0; k<points.size(); ++k)
    {
      const Point<dim> &p = points[k];
      // Work out the vector (stored as a tensor so we can use cross_product)
      // from the source point to the current point, p
      for (unsigned int i = 0; i < dim; ++i)
      {
        shifted_point[i] = p(i) - source_point(i);
      }
      double rad = p.distance(source_point);
      double factor = 1.0/(4.0*numbers::PI*rad*rad*rad);

      // Construct D2G
      for (unsigned int i = 0; i < dim; ++i)
      {
        scaled_vector(i)=shifted_point[i]/rad;
      }
      rhat.outer_product(scaled_vector,scaled_vector);
      D2G=0;
      D2G.add(3.0,rhat,-1.0,eye);

      D2G.vmult(result, coil_direction);

      result *= factor;
      for (unsigned int i=0;i<dim;i++)
      {
        // Real
        value_list[k](i) = result[i];
        // Imaginary
        value_list[k](i+dim) = 0.0;
      }
    }
  }
  template class DipoleSource<3>;
  // END DIPOLESOURCE
  
  // CONDUCTINGSPHERE
  // Constructor
  template<int dim>
  conductingSphere<dim>::conductingSphere(const conductingSphereData &data_in)
  :
  data(data_in)
  {
    // Expect input in scaled variables, s.t. mu_r = mu/mu0.
    // However, the solution requires mu, so multiply by mu0.
    const double mu_c = data.mu_c*EquationData::constant_mu0;
    const double mu_n = data.mu_n*EquationData::constant_mu0;
    
    constant_B_magnitude = sqrt(data.uniform_field[0].norm_sqr() + data.uniform_field[1].norm_sqr());
    
    // calculate the required constants:    
    constant_p = data.sigma*mu_c*data.omega;
    std::complex<double> temp(0, constant_p);
    std::complex<double> v = sqrt(temp)*data.sphere_radius;
    
    // Calculate some useful terms:
//     temp = sqrt(2.0/(numbers::PI*v));
//     std::complex<double> besselhalf_plus = temp*sinh(v);
//     std::complex<double> besselhalf_minus = temp*cosh(v);
    // alternatively use the bessel library (with wrappers to fortran)
    std::complex<double> besselhalf_plus = sp_bessel::besselI(0.5, v);
    std::complex<double> besselhalf_minus = sp_bessel::besselI(-0.5, v);

    // If conducting & non-conducting region have same mu, then things simplify:
    if (data.mu_n == data.mu_c)
    {
      constant_C = 3.0*pow(sqrt(data.sphere_radius),3)/(v*besselhalf_plus);
      constant_D = pow(data.sphere_radius,3)*(3.0*v*besselhalf_minus - (3.0 + v*v)*besselhalf_plus)
                   / ( v*v*besselhalf_plus );
    }
    else
    {    
       constant_C = 3.0*mu_c*v*pow(sqrt(data.sphere_radius),3)
                    / ( (mu_c - mu_n)*v*besselhalf_minus + (mu_n*(1.0 + v*v) - mu_c)*besselhalf_plus );
       constant_D = pow(data.sphere_radius,3)*( (2.0*mu_c + mu_n)*v*besselhalf_minus - (mu_n*(1.0 + v*v) + 2.0*mu_c)*besselhalf_plus )
                    / ( (mu_c - mu_n)*v*besselhalf_minus + (mu_n*(1.0 + v*v)-mu_c)*besselhalf_plus );
    }
    // Debugging (can compare to matlab code)
//     std::cout << constant_C.real() << " + "<<constant_C.imag() << "i" << std::endl;
//     std::cout << constant_D.real() << " + "<<constant_D.imag() << "i" << std::endl;
  }
  
  // Copy constructor
  template<int dim>
  conductingSphere<dim>::conductingSphere(const conductingSphere<dim> &source)
  :
  data(source.data)
  {
    // Copy over the member objects    
    constant_p = source.constant_p;
    constant_C = source.constant_C;
    constant_D = source.constant_D;
    constant_B_magnitude = source.constant_B_magnitude;
  }
  
  // Assignment operator:
  template<int dim>
  conductingSphere<dim>& conductingSphere<dim>::operator= (const conductingSphere<dim> &source)
//   :
//   data(source.data)
  // How to copy across the data part?
  {
    constant_p = source.constant_p;
    constant_C = source.constant_C;
    constant_D = source.constant_D;
    constant_B_magnitude = source.constant_B_magnitude;
    
    return *this;
  }
  
  template <int dim>
  void conductingSphere<dim>::vector_value_list (const std::vector<Point<dim> > &points,
                                             std::vector<Vector<double> > &value_list,
                                             const types::material_id &mat_id) const
  {
    Assert(value_list.size() == points.size(), ExcDimensionMismatch(value_list.size(), points.size()));
    
    for (unsigned int k=0; k<points.size(); ++k)
    {
      const Point<dim> &p = points[k];
      double r, theta, phi;
      std::complex<double> factor;
      // Convert (x,y,z) to (r,theta,phi), spherical polars.
      r = sqrt(p.square());
      theta = (r > 0) ? acos(p(2)/r) : 0.0;
      phi = atan2(p(1),p(0));
      
      if (r < 1e-15)
      {
        factor=0.0;
        value_list[k]=0;
      }
      else if (r < data.sphere_radius)
      {
        std::complex<double> temp(0, constant_p);
        std::complex<double> v=sqrt(temp)*r;
        
        // Calculate the required bessel function values:
//         std::complex<double> bessel3halfs_plus = sqrt(2.0/(numbers::PI*v))*(cosh(v) - (1.0/v)*sinh(v));
        // alternatively use the bessel library (with wrappers to fortran)
        std::complex<double> bessel3halfs_plus = sp_bessel::besselI(1.5, v);
        
        factor = 0.5*constant_B_magnitude*constant_C*bessel3halfs_plus*sin(theta)/sqrt(r);
      }
      else
      {
        factor = 0.5*constant_B_magnitude*(r + constant_D/(r*r))*sin(theta);
      }
      // Convert back to cartesian
      // & split real/imaginary parts:
      value_list[k](0) = -factor.real()*sin(phi);
      value_list[k](1) = factor.real()*cos(phi);
      value_list[k](2) = 0.0;
      value_list[k](3) = -factor.imag()*sin(phi);
      value_list[k](4) = factor.imag()*cos(phi);
      value_list[k](5) = 0.0;
    }
  }
  
  template <int dim>
  void conductingSphere<dim>::curl_value_list (const std::vector<Point<dim> > &points,
                                             std::vector<Vector<double> > &value_list,
                                             const types::material_id &mat_id) const
  {
    Assert(value_list.size() == points.size(), ExcDimensionMismatch(value_list.size(), points.size()));
    
    for (unsigned int k=0; k<points.size(); ++k)
    {
      
      // Use the perturbed values:
      /*
      for (unsigned int d=0; d<dim; ++d)
      {
        value_list[k](d) = perturbed_values[k](d) + data.uniform_field_re(d);
        value_list[k](d+dim) = perturbed_values[k](d+dim) + data.uniform_field_im(d);
      }
      */
      const Point<dim> &p = points[k];
      double r, theta, phi;
      // Convert (x,y,z) to (r,theta,phi), spherical polars.
      r = sqrt(p.square());
      theta = (r > 0) ? acos(p(2)/r) : 0.0;
      phi = atan2(p(1),p(0));
      
      
      
      if (r < 1e-15)
      {
        value_list[k]=0;
      }
      else if (r < data.sphere_radius)
      {
        std::complex<double> temp(0, constant_p);
        const std::complex<double> sqrt_ip = sqrt(temp);
        const std::complex<double> v = sqrt_ip*r;
        
        // Calculate the required bessel function values:
        temp = sqrt(2.0/(numbers::PI*v));
//         const std::complex<double> besselhalf_plus = temp*sinh(v);
//         const std::complex<double> bessel3halfs_plus = temp*(cosh(v) - (1.0/v)*sinh(v));
//         const std::complex<double> bessel5halfs_plus = temp*( (1.0 + 3.0/(v*v))*sinh(v) - (3.0/v)*cosh(v));
        
        const std::complex<double> besselhalf_plus = sp_bessel::besselI(0.5, v);
        const std::complex<double> bessel3halfs_plus = sp_bessel::besselI(1.5, v);
        const std::complex<double> bessel5halfs_plus = sp_bessel::besselI(2.5, v);
        
        
        const std::complex<double> factor_r
        = (1.0/sqrt(r*r*r))*constant_B_magnitude*constant_C*bessel3halfs_plus*cos(theta);
        
        const std::complex<double> factor_theta
        = -(constant_B_magnitude*constant_C*sin(theta))
        *( bessel3halfs_plus/sqrt(r) + sqrt(r)*sqrt_ip*( besselhalf_plus + bessel5halfs_plus ) )/(4.0*r);
          
          // Convert to cartesian:
        const std::complex<double> factor_x = factor_r*sin(theta)*cos(phi) + factor_theta*cos(theta)*cos(phi);
        const std::complex<double> factor_y = factor_r*sin(theta)*sin(phi) + factor_theta*cos(theta)*sin(phi);
        const std::complex<double> factor_z = factor_r*cos(theta) - factor_theta*sin(theta);
        
        value_list[k](0) = factor_x.real();
        value_list[k](1) = factor_y.real();
        value_list[k](2) = factor_z.real();
        value_list[k](3) = factor_x.imag();
        value_list[k](4) = factor_y.imag();
        value_list[k](5) = factor_z.imag();
      }
      else // r > data.sphere_radius
      {
        const std::complex<double> factor_r = constant_B_magnitude*(1.0 + constant_D/(r*r*r))*cos(theta);
        const std::complex<double> factor_theta = -constant_B_magnitude*(1.0 - constant_D/(2.0*r*r*r))*sin(theta);
        // Convert to cartesian:
        const std::complex<double> factor_x = factor_r*sin(theta)*cos(phi) + factor_theta*cos(theta)*cos(phi);
        const std::complex<double> factor_y = factor_r*sin(theta)*sin(phi) + factor_theta*cos(theta)*sin(phi);
        const std::complex<double> factor_z = factor_r*cos(theta) - factor_theta*sin(theta);
        
        value_list[k](0) = factor_x.real();
        value_list[k](1) = factor_y.real();
        value_list[k](2) = factor_z.real();
        value_list[k](3) = factor_x.imag();
        value_list[k](4) = factor_y.imag();
        value_list[k](5) = factor_z.imag();
      }
    }
  }
  template<int dim>
  void conductingSphere<dim>::perturbed_field_value_list (const std::vector< Point<dim> > &points,
                                                          std::vector<Vector<double> > &value_list,
                                                          const types::material_id &mat_id) const
  {
    // Returns the value of the perturbed field:
    // H_{p} = H - H_{0}
    //
    // In general, this is only valid outside of the object, so return 0
    // for any position within the object.
    //
    // TODO: Assume that the centre of the object is (0,0,0) for now
    
    Assert(value_list.size() == points.size(), ExcDimensionMismatch(value_list.size(), points.size()));

    FullMatrix<double> rhat(dim);
    FullMatrix<double> D2G(dim);
    FullMatrix<double> eye(IdentityMatrix(3));
    std::vector<Vector<double>> result(2, Vector<double> (dim));
    
    Vector<double> scaled_vector(dim);
    
    for (unsigned int k=0; k<points.size(); ++k)
    {
      const Point<dim> &p = points[k];

      double r = sqrt(p.square());
      if (r - data.sphere_radius < 1e-10)
      {
        for (unsigned int i=0; i<dim; ++i)
        {
          // TODO: add the formula for the inside of the sphere.
          value_list[k](i) = 0.0;
          value_list[k](i+dim) = 0.0;
        }
      }
      else
      {        
        double factor = 1.0/(4.0*numbers::PI*r*r*r);
        // Construct D2G
        for (unsigned int i=0; i<dim; ++i)
        {
          scaled_vector(i)=p(i)/r;
        }
        rhat.outer_product(scaled_vector,scaled_vector);
        D2G=0;
        D2G.add(3.0,rhat,-1.0,eye);
        
        D2G.vmult(result[0], data.uniform_field[0]);
        D2G.vmult(result[1], data.uniform_field[1]); // D2G is real valued so no extra terms.
        
        result[0] *= factor;
        result[1] *= factor;
        // Now multiply by 2*pi*polarization_tensor
        // NOTE: the polarization tensor is diagonal: M = constant_D*identityMatrix.
        //       and it is complex valued.
        for (unsigned int i=0; i<dim; ++i)
        {
          value_list[k](i) = 2.0*numbers::PI*(constant_D.real()*result[0](i) - constant_D.imag()*result[1](i));
          value_list[k](i+dim) = 2.0*numbers::PI*(constant_D.imag()*result[0](i) + constant_D.real()*result[1](i));
        }
      }
    }
  }
  template<int dim>
  void conductingSphere<dim>::check_spherical_coordinates(const std::vector< Point<dim> > &points,
                                                          std::vector<Vector<double> > &value_list) const
  {
    Assert(value_list.size() == points.size(), ExcDimensionMismatch(value_list.size(), points.size()));
    
    for (unsigned int k=0; k<points.size(); ++k)
    {
      const Point<dim> &p = points[k];
      double r, theta, phi;
      // Convert (x,y,z) to (r,theta,phi), spherical polars.
      r = sqrt(p.square());
//       theta = atan2(p(1),hypot(p(0),p(2)));
      theta = acos(p(2)/r);
      phi = atan2(p(1),p(0));
      value_list[k](0)=r;
      value_list[k](1)=theta;
      value_list[k](2)=phi;
    }
  }

  template class conductingSphere<3>;
  // END CONDUCTINGSPHERE
  
  // CONDUCTINGSPHEREDIPOLE
  // Data instantiation
  template struct conductingSphereDipoleData<3>;
  // Constructor
  template<int dim>
  conductingSphereDipole<dim>::conductingSphereDipole(const conductingSphereDipoleData<dim> &data_in)
  :
  data(data_in),
  rotate_forward(dim,dim)
  {
    // Asserts
    // Check the chosen coil location makes sense:
    Assert(data.h0 > data.sphere_radius,
           StandardExceptions::ExcMessage("backgroundField::conductingSphereDipole: h0 must be greater than sphere_radius"));
    
    // Check the chosen coil direction makes sense (must have length >0)
    Assert(data.coil_direction.norm() > 0.0,
           StandardExceptions::ExcMessage("backgroundField::conductingSphereDipole: coil_direction.norm() must be greater than zero"));
    
    // Calculate constant values
    // alpha = (1-i)*sqrt(omega*mu*sigma/2)
    // gamma = i*(omega*mu*I0*r0*sin(theta0)
    constant_alpha = std::complex<double> (1.0, 1.0);
    constant_alpha *= sqrt(data.omega*data.mu*data.sigma/2.0);
    constant_gamma
    = std::complex<double> (0.0, data.omega*data.mu*data.I0*data.r0*sin(data.theta0));
    
//     std::cout << "CONSTANTS: " << constant_alpha << " " << constant_gamma << std::endl;

    // Allow for arbitrary coil location:
    // This requires that we calculate the angle and axis of rotation from
    // the given coil direction to the reference coil direction.
    // we'll assume the direction and position are the same for now (TODO)
    const Tensor <1,dim> reference_coil_direction({0.0, 0.0, 1.0});
    
    Tensor<1,dim> normalised_coil_direction(data.coil_direction/data.coil_direction.norm());
    // note: reference_coil_direction already normalised.
    rotation_angle = acos(normalised_coil_direction*reference_coil_direction);
    
    cross_product(rotation_axis,
                  normalised_coil_direction,
                  reference_coil_direction);
    if (rotation_axis.norm() > 1e-15)
    {
      rotation_axis /= rotation_axis.norm();
    }
    else
    {
      // We know that the reference axis is [0,0,1], so just require any axis perpendicular to this.
      // Choose [0,1,0].
      rotation_axis = Tensor<1,dim> ({0.0,1.0,0.0});
    }
    // Also store the rotation matrix:
    rotate_forward = MyVectorTools::rodrigues_rotation_matrix(rotation_axis,
                                                              rotation_angle);

    // Store if rotation is needed.
    if (abs(rotation_angle) < 1e-15)
    {
      use_rotation = false;
    }
    else
    {
      use_rotation = true;
    }
    //debug:
//     std::cout << "rotate? " << use_rotation << std::endl;
//     std::cout << "axis " << rotation_axis << " | angle " << rotation_angle << std::endl;
  }
  
  // Copy constructor
  template<int dim>
  conductingSphereDipole<dim>::conductingSphereDipole(const conductingSphereDipole<dim> &source)
  :
  data(source.data)
  {
    // Copy over the member objects
    constant_alpha = source.constant_alpha;
    constant_gamma = source.constant_gamma;
    use_rotation = source.use_rotation;
    rotation_axis = source.rotation_axis;
    rotation_angle = source.rotation_angle;
    rotate_forward = source.rotate_forward;
  }
  
  // Assignment operator:
  template<int dim>
  conductingSphereDipole<dim>& conductingSphereDipole<dim>::operator= (const conductingSphereDipole<dim> &source)
  {
    // Copy over the member objects
    constant_alpha = source.constant_alpha;
    constant_gamma = source.constant_gamma;
    use_rotation = source.use_rotation;
    rotation_axis = source.rotation_axis;
    rotation_angle = source.rotation_angle;
    rotate_forward = source.rotate_forward;
    
    data = source.data; // Is there a better way to copy data struct??
    
    return *this;
  }
  template <int dim>
  unsigned int conductingSphereDipole<dim>::factorial(const unsigned int &n) const
  {
    unsigned int nm1 = n-1;
    return ( n <= 1 ? 1 : n*factorial(nm1));
  }
  
  template<int dim>
  double conductingSphereDipole<dim>::calcTHETA_n (const unsigned int &n,
                                                   const double &theta) const
  {
    double THETA_n
    = sqrt((2.0*(double)n + 1.0)/ (2.0*(double)n*((double)n+1.0)))
    *boost::math::legendre_p((int)n, 1, cos(theta));
    
    return THETA_n;
  }
  // TODO: remove, shouldnt be needed any longer (assuming the bessel fortran wrapper library is installed).
  template<int dim>
  std::complex<double> conductingSphereDipole<dim>::calc_bessel_n_plus_half(const unsigned int &n,
                                                                            const std::complex<double> &x) const
  // 2nd attempt:
  {
    // avoid blow up:
    const double tol = 1e-16;
    if (std::norm(x) < tol)
    {
      return std::complex<double> (0.0, 0.0);
    }
    
    if (n==0)
    {
      return sqrt(2.0/(numbers::PI*x))*sinh(x);
    }
    else if (n==1)
    {
      return sqrt(2.0/(numbers::PI*x))*(cosh(x) - (1.0/x)*sinh(x));
    }
    else
    {
      const unsigned int nm2 = n-2;
      const unsigned int nm1 = n-1;
      // recurrance is:
      // I_{n} = I_{n-2}(x) - (2*(n-1)/x)*I_{n-1}(x).
      const std::complex<long double> temp1 = (std::complex<long double>)calc_bessel_n_plus_half(nm2, x);
      const std::complex<long double> temp2 = (std::complex<long double>)calc_bessel_n_plus_half(nm1, x);
      const std::complex<long double> coeff = (2.0*(long double)n - 1.0)/(std::complex<long double>)x;
      const std::complex<long double> temp_output = temp1 - coeff*temp2;
      std::complex<double> output = (std::complex<double>) temp_output;
//       = calc_bessel_n_plus_half(nm2, x) - *calc_bessel_n_plus_half(nm1,x);
      return output;
    }
  }

// This version has issues at n>8 because of the factorial function.
//   {
//     // Computes the result of I_{n+1/2} (i.e. the Modified bessel function of the first kind).
//     // Had to implement this because boost does not contain these functions for complex arguments.
//     std::complex<double> temp(0.0, 0.0);
//     for (unsigned int s=0; s<n+1; ++s)
//     {
//       std::cout << "fact: " << n << " " << s << " " << (double)factorial(n+s) << std::endl;
//       temp
//       +=
//       (( s%2==0 ? 1.0 : -1.0 )*exp(x) - ( n%2==0 ? 1.0 : -1.0 )*exp(-x))*(double)factorial(n+s)
//       / ( (double)factorial(s)*(double)factorial(n-s)*pow(2.0*x,s) );      
//     }
//     temp *= (1.0/sqrt(2.0*numbers::PI*x));
//     return temp;
//   }
  
  template <int dim>
  void conductingSphereDipole<dim>::vector_value_list (const std::vector<Point<dim> > &points,
                                             std::vector<Vector<double> > &value_list,
                                             const types::material_id &mat_id) const
  {
    Assert(value_list.size() == points.size(), ExcDimensionMismatch(value_list.size(), points.size()));
    
    
    
    // Loop over all points:
    for (unsigned int k=0; k<points.size(); ++k)
    {
      const Point<dim> &point_in = points[k];
      Point<dim> p;
      if (use_rotation)
      {
        // If the coil has been rotated then we need to perform a change of basis so that
        // the new polar axis passes through the rotated coil.
        // A change of basis by a rotation is given by v' = R^{-1}*v.
        // where v is the vector in the original basis, v' is the vector in the new basis
        // and R is the rotation matrix.
        // TODO: double check we haven't got R and R^-1 backwards here and below.
        Vector<double> temp_in (dim);
        for (unsigned int d=0; d<dim; ++d)
        {
          temp_in(d) = point_in(d);
        }
        Vector<double> temp_out (dim);
        rotate_forward.vmult(temp_out,temp_in);
        for (unsigned int d=0; d<dim; ++d)
        {
          p(d) = temp_out(d);
        }
      }
      else
      {
        p = point_in;
      }
      // Convert to spherical polars:
      // NOTE: we copy the convention in the paper.
      // This should be done according to the physics convention (ISO standard).
      // TODO: confirm this is the correct ordering.
      const double r = sqrt(p.square());
      const double theta = acos(p(2)/r);
      const double phi = atan2(p(1),p(0));
      
      // Storage for the components:
      std::complex<double> E_r(0.0, 0.0);
      std::complex<double> E_theta(0.0, 0.0);
      std::complex<double> E_phi(0.0, 0.0);
      
      const double a_over_r0 = data.sphere_radius/data.r0;
      const double a_over_r = data.sphere_radius/r;
      const std::complex<double> alphaa = constant_alpha*data.sphere_radius;
      
      // TODO: make truncation dependent on some tolerance
      // and fix the bessel function problem:
      // Need to avoid computing besseli with x >> n
      // am unable to find a quick way to implement a robust method at the moment.
      const unsigned int maxN = 20;
      // avoid singularity at r = 0
      if (r < 1e-15)
      {
        value_list[k]=0;
      }
      // inside sphere:
      else if (r < data.sphere_radius)
      {
        // Parameters not dependent upon n:
        const std::complex<double> alphar = constant_alpha*r;
        
        std::complex<double> sum_total(0.0,0.0);
        
        // Infinite sum, for now simply truncate.
        for (unsigned int n=1; n<maxN+1; ++n)
        {
          // Using my functions (commented out as they seem unstable):
//           const std::complex<double> bessel_n_plus_half
//           = calc_bessel_n_plus_half(n, alphar);
//           const unsigned int nm1 = n-1;
//           const std::complex<double> bessel_n_minus_half
//           = calc_bessel_n_plus_half(nm1, alphaa);
          
          // Using library linked to FORTRAN
          const double np12 = (double)n+0.5;
          const double nm12 = (double)n-0.5;
          const std::complex<double> bessel_n_plus_half = sp_bessel::besselI(np12, alphar);
          const std::complex<double> bessel_n_minus_half  = sp_bessel::besselI(nm12, alphaa);

//           std::cout << "BESSEL INPUT: " << n << " " << alphar << std::endl;
//           std::cout << "BESSEL " << bessel_n_plus_half << " " <<  bessel_n_minus_half  <<std::endl;
          
          const double THETA_N0 = calcTHETA_n(n, data.theta0);
          const double THETA_N = calcTHETA_n(n, theta);
          
          
          sum_total
          += bessel_n_plus_half*pow(a_over_r0, n+1)*THETA_N0*THETA_N/bessel_n_minus_half;
          
//           std::cout << "INPUT: "
//           << alphar << " "
//           << alphaa << std::endl;
//           
//           
//           std::cout << "CUMSUM PARTS: " << bessel_n_plus_half << " "
//           << pow(a_over_r0, n+1) << " "
//           << THETA_N0 << " "
//           << THETA_N << " "
//           << bessel_n_minus_half
//           << std::endl;
          
          
//           std::cout << "CUMSUM: " << n << " " << sum_total << std::endl;
        }
        E_phi = constant_gamma*sum_total/(sqrt(data.sphere_radius*r)*alphaa);
        // Convert back to cartesian
        // & split real/imaginary parts:
        // NOTE: we know E_r and E_theta = 0, which simplifies things:
        const double phi_to_x_convert = -sin(phi);
        const double phi_to_y_convert = cos(phi);
        Vector<double> temp_re (dim);
        Vector<double> temp_im (dim);
        temp_re(0) = E_phi.real()*phi_to_x_convert;
        temp_re(1) = E_phi.real()*phi_to_y_convert;
        temp_re(2) = 0.0;
        temp_im(0) = E_phi.imag()*phi_to_x_convert;
        temp_im(1) = E_phi.imag()*phi_to_y_convert;
        temp_im(2) = 0.0;
        
        // Finally deal with any rotation required:
        // Note, rotation matrices are orthogonal, so can use vmult
        // i.e. no need for inverse.
        Vector<double> E_re(dim);
        Vector<double> E_im(dim);
        if (use_rotation)
        {
          rotate_forward.Tvmult(E_re,temp_re);
          rotate_forward.Tvmult(E_im,temp_im);
        }
        else
        {
          E_re = temp_re;
          E_im = temp_im;
        }
        for (unsigned int d=0; d<dim; ++d)
        {
          value_list[k](d) = E_re(d);
          value_list[k](d+dim) = E_im(d);
        }

      }
      // outside sphere.
      else
      {
        // Must first calc E_s, then add to second infinite sum
        const double r_gt = std::max(r,data.r0);
        const double r_lt = std::min(r,data.r0);
        const double r_lt_over_r_gt = r_lt/r_gt;
        const std::complex<double> alphaa = constant_alpha*data.sphere_radius;
//         std::complex<double> alphar = constant_alpha*r;
        std::complex<double> sum_total_Es(0.0,0.0);
        std::complex<double> sum_total_Esc(0.0,0.0);
        // Infinite sum, for now simply truncate.
        for (unsigned int n=1; n<maxN+1; ++n)
        {
          const double THETA_N0 = calcTHETA_n(n, data.theta0);
          const double THETA_N = calcTHETA_n(n, theta);

          sum_total_Es += (1.0/(2.0*(double)n + 1.0))*(pow(r_lt_over_r_gt, n)/r_gt)*THETA_N0*THETA_N;

          // Using my functions (commented out as they seem unstable):
          // TODO: remove
//           const std::complex<double> bessel_n_plus_half
//           = calc_bessel_n_plus_half(n, alphaa);
//           const unsigned int nm1 = n-1;
//           const std::complex<double> bessel_n_minus_half
//           = calc_bessel_n_plus_half(nm1, alphaa);
          
          // Using library linked to FORTRAN
          const double np12 = (double)n+0.5;
          const double nm12 = (double)n-0.5;
          const std::complex<double> bessel_n_plus_half = sp_bessel::besselI(np12, alphaa);
          const std::complex<double> bessel_n_minus_half  = sp_bessel::besselI(nm12, alphaa);
          
          sum_total_Esc
          += (bessel_n_plus_half/(alphaa*bessel_n_minus_half) - 1.0/(2.0*(double)n + 1.0))
          *pow(a_over_r, n)
          *pow(a_over_r0, n+1)
          *THETA_N0*THETA_N;
        }
        E_phi = constant_gamma*sum_total_Es + (constant_gamma/r)*sum_total_Esc;
        // Convert back to cartesian
        // & split real/imaginary parts:
        // NOTE: we know E_r and E_theta = 0, which simplifies things:
        const double phi_to_x_convert = -sin(phi);
        const double phi_to_y_convert = cos(phi);
        Vector<double> temp_re (dim);
        Vector<double> temp_im (dim);
        temp_re(0) = E_phi.real()*phi_to_x_convert;
        temp_re(1) = E_phi.real()*phi_to_y_convert;
        temp_re(2) = 0.0;
        temp_im(0) = E_phi.imag()*phi_to_x_convert;
        temp_im(1) = E_phi.imag()*phi_to_y_convert;
        temp_im(2) = 0.0;
        
        // Finally deal with any rotation required:
        // Note, rotation matrices are orthogonal, so can use Tvmult
        // i.e. no need for inverse.
        Vector<double> E_re(dim);
        Vector<double> E_im(dim);
        if (use_rotation)
        {
          rotate_forward.Tvmult(E_re,temp_re);
          rotate_forward.Tvmult(E_im,temp_im);
        }
        else
        {
          E_re = temp_re;
          E_im = temp_im;
        }
        for (unsigned int d=0; d<dim; ++d)
        {
          value_list[k](d) = E_re(d);
          value_list[k](d+dim) = E_im(d);
        }
      }      
      
      // output the spherical coord values:
//       value_list[k](0) = E_r.real();
//       value_list[k](1) = E_theta.real();
//       value_list[k](2) = E_phi.real();
//       value_list[k](3) = E_r.imag();
//       value_list[k](4) = E_theta.imag();
//       value_list[k](5) = E_phi.imag();
    }
  }
  template <int dim>
  void conductingSphereDipole<dim>::scattered_field_value_list (const std::vector<Point<dim> > &points,
                                             std::vector<Vector<double> > &value_list,
                                             const types::material_id &mat_id) const
  {
    Assert(value_list.size() == points.size(), ExcDimensionMismatch(value_list.size(), points.size()));
    // Loop over all points:
    for (unsigned int k=0; k<points.size(); ++k)
    {
      const Point<dim> &point_in = points[k];
      Point<dim> p;
      if (use_rotation)
      {
        // If the coil has been rotated then we need to perform a change of basis so that
        // the new polar axis passes through the rotated coil.
        // A change of basis by a rotation is given by v' = R^{-1}*v.
        // where v is the vector in the original basis, v' is the vector in the new basis
        // and R is the rotation matrix.
        // TODO: double check we haven't got R and R^-1 backwards here and below.
        Vector<double> temp_in (dim);
        for (unsigned int d=0; d<dim; ++d)
        {
          temp_in(d) = point_in(d);
        }
        Vector<double> temp_out (dim);
        rotate_forward.vmult(temp_out,temp_in);
        for (unsigned int d=0; d<dim; ++d)
        {
          p(d) = temp_out(d);
        }
      }
      else
      {
        p = point_in;
      }
      // Convert to spherical polars:
      // NOTE: we copy the convention in the paper.
      // This should be done according to the physics convention (ISO standard).
      // TODO: confirm this is the correct ordering.
      const double r = sqrt(p.square());
      const double theta = acos(p(2)/r);
      const double phi = atan2(p(1),p(0));
      
      // Storage for the components:
      std::complex<double> E_r(0.0, 0.0);
      std::complex<double> E_theta(0.0, 0.0);
      std::complex<double> E_phi(0.0, 0.0);
      
      // TODO: make truncation dependent on some tolerance
      // and fix the bessel function problem:
      // Need to avoid computing besseli with x >> n
      // am unable to find a quick way to implement a robust method at the moment.
      const unsigned int maxN = 20;
      
      // Not valid inside the sphere
      if (r < data.sphere_radius)
      {
        value_list[k]=0.0;
      }
      else
      {
        // Constants which appear:
        const double a_over_r0 = data.sphere_radius/data.r0;
        const double a_over_r = data.sphere_radius/r;
        const std::complex<double> gamma_over_r = constant_gamma/r;
        const std::complex<double> alphaa = constant_alpha*data.sphere_radius;
       
        const double r_gt = std::max(r,data.r0);
        const double r_lt = std::min(r,data.r0);
        
        std::complex<double> sum_total_Esc(0.0,0.0);
        // Infinite sum, for now simply truncate.
        for (unsigned int n=1; n<maxN+1; ++n)
        {
          const double THETA_N0 = calcTHETA_n(n, data.theta0);
          const double THETA_N = calcTHETA_n(n, theta);

          const double np12 = n+0.5;
          const double nm12 = n-0.5;
          const std::complex<double> bessel_n_plus_half = sp_bessel::besselI(np12, alphaa);
          const std::complex<double> bessel_n_minus_half  = sp_bessel::besselI(nm12, alphaa);
          
//           const std::complex<double> bessel_n_plus_half
//           = calc_bessel_n_plus_half(n, alphaa);
// 
//           const unsigned int nm1 = n-1;
//           const std::complex<double> bessel_n_minus_half
//           = calc_bessel_n_plus_half(nm1, alphaa);
          
          sum_total_Esc
          += (bessel_n_plus_half/(alphaa*bessel_n_minus_half) - (1.0/(2.0*(double)n + 1.0)))
          *pow(a_over_r, n)*pow(a_over_r0, n+1)
          *THETA_N0*THETA_N;
        }
        E_phi = gamma_over_r*sum_total_Esc;
        // Convert back to cartesian
        // & split real/imaginary parts:
        // NOTE: we know E_r and E_theta = 0, which simplifies things:
        const double phi_to_x_convert = -sin(phi);
        const double phi_to_y_convert = cos(phi);
        Vector<double> temp_re (dim);
        Vector<double> temp_im (dim);
        temp_re(0) = E_phi.real()*phi_to_x_convert;
        temp_re(1) = E_phi.real()*phi_to_y_convert;
        temp_re(2) = 0.0;
        temp_im(0) = E_phi.imag()*phi_to_x_convert;
        temp_im(1) = E_phi.imag()*phi_to_y_convert;
        temp_im(2) = 0.0;
        
        // Finally deal with any rotation required:
        // Note, rotation matrices are orthogonal, so can use Tvmult
        // i.e. no need for inverse.
        Vector<double> E_re(dim);
        Vector<double> E_im(dim);
        if (use_rotation)
        {
          rotate_forward.Tvmult(E_re,temp_re);
          rotate_forward.Tvmult(E_im,temp_im);
        }
        else
        {
          E_re = temp_re;
          E_im = temp_im;
        }
        for (unsigned int d=0; d<dim; ++d)
        {
          value_list[k](d) = E_re(d);
          value_list[k](d+dim) = E_im(d);
        }
      }
      
      // output the spherical coord values:
//       value_list[k](0) = E_r.real();
//       value_list[k](1) = E_theta.real();
//       value_list[k](2) = E_phi.real();
//       value_list[k](3) = E_r.imag();
//       value_list[k](4) = E_theta.imag();
//       value_list[k](5) = E_phi.imag();
    }
  }
  template class conductingSphereDipole<3>;
  // END CONDUCTINGSPHEREDIPOLE
  
  // CLASS SPHERICALDIPOLE
  // Constructor
  template<int dim>
  SphericalDipole<dim>::SphericalDipole(const conductingSphereDipoleData<dim> &data_in)
  :
  data(data_in),
  rotate_forward(dim,dim)
  {
    // Asserts
    // Check the chosen coil location makes sense:
    Assert(data.h0 > data.sphere_radius,
           StandardExceptions::ExcMessage("backgroundField::SphericalDipole: h0 must be greater than sphere_radius"));
    
    // Check the chosen coil direction makes sense (must have length >0)
    Assert(data.coil_direction.norm() > 0.0,
           StandardExceptions::ExcMessage("backgroundField::SphericalDipole: coil_direction.norm() must be greater than zero"));
    
    // Calculate constant values
    // alpha = (1-i)*sqrt(omega*mu*sigma/2)
    // gamma = i*(omega*mu*I0*r0*sin(theta0)
    constant_alpha = std::complex<double> (1.0, 1.0);
    constant_alpha *= sqrt(data.omega*data.mu*data.sigma/2.0);
    constant_gamma
    = std::complex<double> (0.0, data.omega*data.mu*data.I0*data.r0*sin(data.theta0));
    
//     std::cout << "CONSTANTS: " << constant_alpha << " " << constant_gamma << std::endl;

    // Allow for arbitrary coil location:
    // This requires that we calculate the angle and axis of rotation from
    // the given coil direction to the reference coil direction.
    // we'll assume the direction and position are the same for now (TODO)
    const Tensor <1,dim> reference_coil_direction({0.0, 0.0, 1.0});
    
    Tensor<1,dim> normalised_coil_direction(data.coil_direction/data.coil_direction.norm());
    // note: reference_coil_direction already normalised.
    rotation_angle = acos(normalised_coil_direction*reference_coil_direction);
    
    cross_product(rotation_axis,
                  normalised_coil_direction,
                  reference_coil_direction);
    if (rotation_axis.norm() > 1e-15)
    {
      rotation_axis /= rotation_axis.norm();
    }
    else
    {
      // We know that the reference axis is [0,0,1], so just require any axis perpendicular to this.
      // Choose [0,1,0].
      rotation_axis = Tensor<1,dim> ({0.0,1.0,0.0});
    }
    // Also store the rotation matrix:
    rotate_forward = MyVectorTools::rodrigues_rotation_matrix(rotation_axis,
                                                              rotation_angle);

    // Store if rotation is needed.
    if (abs(rotation_angle) < 1e-15)
    {
      use_rotation = false;
    }
    else
    {
      use_rotation = true;
    }
    //debug:
//     std::cout << "rotate? " << use_rotation << std::endl;
//     std::cout << "axis " << rotation_axis << " | angle " << rotation_angle << std::endl;
  }
  
  // Copy constructor
  template<int dim>
  SphericalDipole<dim>::SphericalDipole(const SphericalDipole<dim> &source)
  :
  data(source.data)
  {
    // Copy over the member objects
    constant_alpha = source.constant_alpha;
    constant_gamma = source.constant_gamma;
    use_rotation = source.use_rotation;
    rotation_axis = source.rotation_axis;
    rotation_angle = source.rotation_angle;
    rotate_forward = source.rotate_forward;
  }
  
  // Assignment operator:
  template<int dim>
  SphericalDipole<dim>& SphericalDipole<dim>::operator= (const SphericalDipole<dim> &source)
  {
    // Copy over the member objects
    constant_alpha = source.constant_alpha;
    constant_gamma = source.constant_gamma;
    use_rotation = source.use_rotation;
    rotation_axis = source.rotation_axis;
    rotation_angle = source.rotation_angle;
    rotate_forward = source.rotate_forward;
    
    data = source.data; // Is there a better way to copy data struct??
    
    return *this;
  }
  template<int dim>
  double SphericalDipole<dim>::calcTHETA_n (const unsigned int &n,
                                                   const double &theta) const
  {
    double THETA_n
    = sqrt((2.0*(double)n + 1.0)/ (2.0*(double)n*((double)n+1.0)))
    *boost::math::legendre_p((int)n, 1, cos(theta));
    
    return THETA_n;
  }
  template <int dim>
  void SphericalDipole<dim>::vector_value_list (const std::vector<Point<dim> > &points,
                                                std::vector<Vector<double> > &value_list,
                                                const types::material_id &mat_id) const
  {
    Assert(value_list.size() == points.size(), ExcDimensionMismatch(value_list.size(), points.size()));
    // Loop over all points:
    for (unsigned int k=0; k<points.size(); ++k)
    {
      const Point<dim> &point_in = points[k];
      Point<dim> p;
      if (use_rotation)
      {
        // If the coil has been rotated then we need to perform a change of basis so that
        // the new polar axis passes through the rotated coil.
        // A change of basis by a rotation is given by v' = R^{-1}*v.
        // where v is the vector in the original basis, v' is the vector in the new basis
        // and R is the rotation matrix.
        // TODO: double check we haven't got R and R^T backwards here and below.
        Vector<double> temp_in (dim);
        for (unsigned int d=0; d<dim; ++d)
        {
          temp_in(d) = point_in(d);
        }
        Vector<double> temp_out (dim);
        rotate_forward.vmult(temp_out,temp_in);
        for (unsigned int d=0; d<dim; ++d)
        {
          p(d) = temp_out(d);
        }
      }
      else
      {
        p = point_in;
      }
      // Convert to spherical polars:
      // NOTE: we copy the convention in the paper.
      // This should be done according to the physics convention (ISO standard).
      const double r = sqrt(p.square());
      const double theta = acos(p(2)/r);
      const double phi = atan2(p(1),p(0));
      
      // Storage for the components:
      std::complex<double> E_r(0.0, 0.0);
      std::complex<double> E_theta(0.0, 0.0);
      std::complex<double> E_phi(0.0, 0.0);
      
      const double a_over_r0 = data.sphere_radius/data.r0;
      const double a_over_r = data.sphere_radius/r;
      const std::complex<double> alphaa = constant_alpha*data.sphere_radius;
      
      // TODO: make truncation dependent on some tolerance
      const unsigned int maxN = 20;
      // avoid singularity at r = 0
      if (r < 1e-10)
      {
        value_list[k]=0;
      }
      // inside sphere:
      else if (r < data.sphere_radius)
      {
        // Parameters not dependent upon n:
        const std::complex<double> alphar = constant_alpha*r;
        
        std::complex<double> sum_total(0.0,0.0);
        
        // Infinite sum, for now simply truncate.
        for (unsigned int n=1; n<maxN+1; ++n)
        {
          // Using library linked to FORTRAN
          const double np12 = (double)n+0.5;
          const double nm12 = (double)n-0.5;
          const std::complex<double> bessel_n_plus_half = sp_bessel::besselI(np12, alphar);
          const std::complex<double> bessel_n_minus_half  = sp_bessel::besselI(nm12, alphaa);
          
          const double THETA_N0 = calcTHETA_n(n, data.theta0);
          const double THETA_N = calcTHETA_n(n, theta);
          
          sum_total
          += bessel_n_plus_half*pow(a_over_r0, n+1)*THETA_N0*THETA_N/bessel_n_minus_half;
        }
        E_phi = constant_gamma*sum_total/(sqrt(data.sphere_radius*r)*alphaa);
        // Convert back to cartesian
        // & split real/imaginary parts:
        // NOTE: we know E_r and E_theta = 0, which simplifies things:
        const double phi_to_x_convert = -sin(phi);
        const double phi_to_y_convert = cos(phi);
        Vector<double> temp_re (dim);
        Vector<double> temp_im (dim);
        temp_re(0) = E_phi.real()*phi_to_x_convert;
        temp_re(1) = E_phi.real()*phi_to_y_convert;
        temp_re(2) = 0.0;
        temp_im(0) = E_phi.imag()*phi_to_x_convert;
        temp_im(1) = E_phi.imag()*phi_to_y_convert;
        temp_im(2) = 0.0;
        
        // Finally deal with any rotation required:
        // Note, rotation matrices are orthogonal, so can use vmult
        // i.e. no need for inverse.
        Vector<double> E_re(dim);
        Vector<double> E_im(dim);
        if (use_rotation)
        {
          rotate_forward.Tvmult(E_re,temp_re);
          rotate_forward.Tvmult(E_im,temp_im);
        }
        else
        {
          E_re = temp_re;
          E_im = temp_im;
        }
        for (unsigned int d=0; d<dim; ++d)
        {
          value_list[k](d) = E_re(d);
          value_list[k](d+dim) = E_im(d);
        }
      }
      // outside sphere.
      else
      {
        // Must first calc E_s, then add to second infinite sum
        const double r_gt = std::max(r,data.r0);
        const double r_lt = std::min(r,data.r0);
        const double r_lt_over_r_gt = r_lt/r_gt;
        std::complex<double> sum_total_Esrc(0.0,0.0);
        // Infinite sum, for now simply truncate.
        for (unsigned int n=1; n<maxN+1; ++n)
        {
          const double THETA_N0 = calcTHETA_n(n, data.theta0);
          const double THETA_N = calcTHETA_n(n, theta);

          sum_total_Esrc
          += (1.0/(2.0*(double)n + 1.0))*(pow(r_lt_over_r_gt, n)/r_gt)*THETA_N0*THETA_N;
        }
        E_phi = constant_gamma*sum_total_Esrc;
        // Convert back to cartesian
        // & split real/imaginary parts:
        // NOTE: we know E_r and E_theta = 0, which simplifies things:
        const double phi_to_x_convert = -sin(phi);
        const double phi_to_y_convert = cos(phi);
        Vector<double> temp_re (dim);
        Vector<double> temp_im (dim);
        temp_re(0) = E_phi.real()*phi_to_x_convert;
        temp_re(1) = E_phi.real()*phi_to_y_convert;
        temp_re(2) = 0.0;
        temp_im(0) = E_phi.imag()*phi_to_x_convert;
        temp_im(1) = E_phi.imag()*phi_to_y_convert;
        temp_im(2) = 0.0;
        
        // Finally deal with any rotation required:
        // Note, rotation matrices are orthogonal, so can use Tvmult
        Vector<double> E_re(dim);
        Vector<double> E_im(dim);
        if (use_rotation)
        {
          rotate_forward.Tvmult(E_re,temp_re);
          rotate_forward.Tvmult(E_im,temp_im);
        }
        else
        {
          E_re = temp_re;
          E_im = temp_im;
        }
        for (unsigned int d=0; d<dim; ++d)
        {
          value_list[k](d) = E_re(d);
          value_list[k](d+dim) = E_im(d);
        }
      }
    }
  }
  template class SphericalDipole<3>;
  // END SPHERICALDIPOLE
  
  // conductingObject_polarization_tensor
  template<int dim>
  conductingObject_polarization_tensor<dim>
  ::conductingObject_polarization_tensor(const std::vector<Vector<double> > &uniform_field,
                                         const std::vector<FullMatrix<double> > &polarizationTensor)
  :
  uniform_field(uniform_field),
  polarizationTensor(polarizationTensor)  
  {
  }
  
  
  template <int dim>
  void conductingObject_polarization_tensor<dim>
  ::curl_value_list (const std::vector<Point<dim> > &points,
                     std::vector<Vector<double> > &value_list,
                     const types::material_id &mat_id) const
  {
    Assert(value_list.size() == points.size(), ExcDimensionMismatch(value_list.size(), points.size()));
    
    // No analytical solution, just set uniform far field conditions. 
    // TODO: switch over to using the perturbation tensor to calculate hte
    //       perturbed field, then add the uniform field.
    
    std::vector<Vector<double> > perturbed_value_list(value_list.size(), Vector<double> (dim+dim));
    perturbed_field_value_list(points, perturbed_value_list, mat_id);
    
    for (unsigned int k=0; k<points.size(); ++k)
    {
      for (unsigned int d=0; d<dim; ++d)
      {
        // Seems to only work when we treat the BC as n x curl E = 
        // TODO: work out why using curl(A) = B (computed from perturbation tensor formula) doesn't work.
        
        value_list[k](d) = perturbed_value_list[k](d) + uniform_field[0](d);
        //EquationData::constant_mu0*
        value_list[k](d+dim) = perturbed_value_list[k](d+dim) + uniform_field[1](d);
      }
    }
  }
  template<int dim>
  void conductingObject_polarization_tensor<dim>::perturbed_field_value_list (const std::vector< Point<dim> > &points,
                                                        std::vector<Vector<double> > &value_list,
                                                        const types::material_id &mat_id) const
  {
    // Returns the value of the perturbed field:
    // H_{p} = H - H_{0}
    //
    // TODO: Assume that the centre of the object is (0,0,0) for now
    
    Assert(value_list.size() == points.size(), ExcDimensionMismatch(value_list.size(), points.size()));

    FullMatrix<double> rhat(dim);
    FullMatrix<double> D2G(dim);
    FullMatrix<double> eye(IdentityMatrix(3));
    std::vector<Vector<double>> D2Gresult(2, Vector<double> (dim));
    std::vector<Vector<double>> PTresult(2, Vector<double> (dim));
    
    Vector<double> scaled_vector(dim);
    
    for (unsigned int k=0; k<points.size(); ++k)
    {
      const Point<dim> &p = points[k];
      
      double r = sqrt(p.square());
      // Avoid singular point
      if (r < 1e-10)
      {
        value_list[k]=0.0;
      }
      else
      {
        // perturbed field = D2G*M*H0 = V2*V1...
        // where:
        // V1 = M*H0, (Note [0] -> real part, [1] -> imag part).
        // V1[0] = M[0]*H0[0] - M[1]*H0[1];
        // V1[1] = M[0]*H-[1] + M[1]*H0[0];
        //
        // V2 = D2G*V1 (D2G is real valued).
        // V2[0] = D2G*V1[0];
        // V2[1] = D2G*V1[1];
        double factor = 1.0/(4.0*numbers::PI*r*r*r);
        // Construct D2G
        for (unsigned int i=0; i<dim; ++i)
        {
          scaled_vector(i)=p(i)/r;
        }
        rhat.outer_product(scaled_vector,scaled_vector);
        
        // TODO: fix so that complex uniform field would work (don't use it for now, so not an immediate problem).
        polarizationTensor[0].vmult(PTresult[0], uniform_field[0]);
        //polarizationTensor[0].vmult(PTresult[1], uniform_field[1]);
        
        polarizationTensor[1].vmult(PTresult[1], uniform_field[0]);
        //polarizationTensor[1].vmult_add(PTresult[0], uniform_field[1]);
        
        D2G=0;
        D2G.add(3.0,rhat,-1.0,eye);
        
        D2G.vmult(D2Gresult[0], PTresult[0]);
        D2G.vmult(D2Gresult[1], PTresult[1]); // D2G is real valued so no extra terms.
        
        D2Gresult[0] *= factor;
        D2Gresult[1] *= factor;
        
        for (unsigned int i=0; i<dim; ++i)
        {
          value_list[k](i) = D2Gresult[0](i);
          value_list[k](i+dim) = D2Gresult[1](i);
        }
      }
    }
  }

  template class conductingObject_polarization_tensor<3>; 
  // END CONDUCTINGCUBE
  
  // WAVE PROPAGATION
  template<int dim>
  WavePropagation<dim>::WavePropagation(Vector<double> &k_wave,
                                        Vector<double> &p_wave)
  :
  k_wave(k_wave),
  p_wave(p_wave)
  {
    // Note: need k orthogonal to p
    //       with |k| < 1, |p| = 1.
    // Examples:
    // horizontal wave.
    //   k_wave(0) = 0.0;
    //   k_wave(1) = 0.1;
    //   k_wave(2) = 0.0;
    //   p_wave(0) = 1.0;
    //   p_wave(1) = 0.0;
    //   p_wave(2) = 0.0;
    // diagonal wave.
    //   k_wave(0) = -0.1;
    //   k_wave(1) = -0.1;
    //   k_wave(2) = 0.2;
    //   p_wave(0) = 1./sqrt(3.);
    //   p_wave(1) = 1./sqrt(3.);
    //   p_wave(2) = 1./sqrt(3.);
  
    // Make sure input is sane:
    Assert (k_wave.size() == 3, ExcDimensionMismatch (k_wave.size(), 3));
    Assert (k_wave.size() == k_wave.size(), ExcDimensionMismatch (k_wave.size(), k_wave.size()));
    const double delta = 1e-10;
    Assert (abs(p_wave.norm_sqr()-1.0) < delta, ExcIndexRange(p_wave.norm_sqr(), 1, 1));
  } 

  template <int dim>
  void WavePropagation<dim>::vector_value_list (const std::vector<Point<dim> > &points,
                                                std::vector<Vector<double> > &value_list,
                                                const types::material_id &mat_id) const
  {
    Assert(value_list.size() == points.size(), ExcDimensionMismatch(value_list.size(), points.size()));

    double exponent;
    for (unsigned int i=0; i<points.size(); ++i)
    {
      const Point<dim> &p = points[i];
      exponent = k_wave(0)*p(0)+k_wave(1)*p(1)+k_wave(2)*p(2);
      for (unsigned int d=0; d<dim; ++d)
      {
        // Real:
        value_list[i](d) = ( p_wave(d) )*std::cos(exponent);
        // Imaginary:
        value_list[i](d+dim) = ( p_wave(d) )*std::sin(exponent);
      }
    }
  }
  template <int dim>
  void WavePropagation<dim>::curl_value_list(const std::vector<Point<dim> > &points,
                                             std::vector<Vector<double> > &value_list,
                                             const types::material_id &mat_id) const
  {
    Assert(value_list.size() == points.size(), ExcDimensionMismatch(value_list.size(), points.size()));
    
    double exponent;
    for (unsigned int i=0; i<points.size(); ++i)
    {
      const Point<dim> &p = points[i];
      exponent = k_wave(0)*p(0)+k_wave(1)*p(1)+k_wave(2)*p(2);
      // Real:
      value_list[i](0) = -( k_wave(1)*p_wave(2) - k_wave(2)*p_wave(1) )*std::sin(exponent);
      value_list[i](1) = -( k_wave(2)*p_wave(0) - k_wave(0)*p_wave(2) )*std::sin(exponent);
      value_list[i](2) = -( k_wave(0)*p_wave(1) - k_wave(1)*p_wave(0) )*std::sin(exponent);
      // Imaginary:
      value_list[i](3) =  ( k_wave(1)*p_wave(2) - k_wave(2)*p_wave(1) )*std::cos(exponent);
      value_list[i](4) =  ( k_wave(2)*p_wave(0) - k_wave(0)*p_wave(2) )*std::cos(exponent);
      value_list[i](5) =  ( k_wave(0)*p_wave(1) - k_wave(1)*p_wave(0) )*std::cos(exponent);
    }
  }
  template class WavePropagation<3>; 
  // END WAVE PROPAGATION
  
  // POLYNOMIALTEST:
  // No constructor needed.
  template <int dim>
  void polynomialTest<dim>::vector_value_list (const std::vector<Point<dim> > &points,
                                              std::vector<Vector<double> > &value_list,
                                              const types::material_id &mat_id) const
  {
    Assert(value_list.size() == points.size(), ExcDimensionMismatch(value_list.size(), points.size()));

    for (unsigned int i=0; i<points.size(); ++i)
      {
        const Point<dim> &p = points[i];

        /* quadratic: */
        value_list[i](0) = p(0)*p(0);
        value_list[i](1) = p(1)*p(1);
        value_list[i](2) = p(2)*p(2);
        value_list[i](3) = p(0)*p(0);
        value_list[i](4) = p(1)*p(1);
        value_list[i](5) = p(2)*p(2);
      }
  }
  template <int dim>
  void polynomialTest<dim>::rhs_value_list (const std::vector<Point<dim> > &points,
                                            std::vector<Vector<double> > &value_list,
                                            const types::material_id &mat_id) const
  {
    Assert(value_list.size() == points.size(), ExcDimensionMismatch(value_list.size(), points.size()));
    for (unsigned int i=0; i<points.size(); ++i)
      {
        const Point<dim> &p = points[i];

        /* quadratic: */
        value_list[i](0) = p(0)*p(0);
        value_list[i](1) = p(1)*p(1);
        value_list[i](2) = p(2)*p(2);
        value_list[i](3) = p(0)*p(0);
        value_list[i](4) = p(1)*p(1);
        value_list[i](5) = p(2)*p(2);
      }
  }
  // END POLYNOMIALTEST
  template class polynomialTest<3>;
  
  // curlUniformField
  template<int dim>
  curlUniformField<dim>::curlUniformField(const std::vector<Vector<double> > &uniform_field)
  :
  uniform_field(uniform_field)
  {
  }

  template <int dim>
  void curlUniformField<dim>::vector_value_list (const std::vector<Point<dim> > &points,
                                                 std::vector<Vector<double> > &value_list) const
  {
    Assert(value_list.size() == points.size(), ExcDimensionMismatch(value_list.size(), points.size()));

    // Not required, just set to zero.
    for (unsigned int k=0; k<points.size(); ++k)
    {
      value_list[k]=0.0;
    }
  }

  template <int dim>
  void curlUniformField<dim>::curl_value_list (const std::vector<Point<dim> > &points,
                                               std::vector<Vector<double> > &value_list) const
  {
    Assert(value_list.size() == points.size(), ExcDimensionMismatch(value_list.size(), points.size()));

    // Simply set the field to be the uniform field given in the constructor.
    for (unsigned int k=0; k<points.size(); ++k)
    {
      for (unsigned int d=0; d<dim; ++d)
      {
        // TODO: Make sure this doesn't need to be scaled by some factor
        value_list[k](d) = uniform_field[0](d);
        value_list[k](d+dim) = uniform_field[1](d);
      }
    }
  }
  template class curlUniformField<3>;
  // END curlUniformField
  
  // TEAMBenchmark7
  template<int dim>
  TEAMBenchmark7<dim>::TEAMBenchmark7(const Point<dim> &coil_centre,
                                      const std::vector<Point<dim>> &corner_centres,
                                      const types::material_id coil_mat_id)
  :
  coil_centre(coil_centre),
  corner_centres(corner_centres),
  coil_mat_id(coil_mat_id)
  {
  }
  
  template<int dim>
  void TEAMBenchmark7<dim>::rhs_value_list(const std::vector<Point<dim> > &points,
                                           std::vector<Vector<double> >   &values,
                                           const types::material_id &mat_id) const
  {
    Assert(values.size() == points.size(), ExcDimensionMismatch(values.size(), points.size()));
    // First, avoid any work if the we're outside the coil material.
    if (mat_id != coil_mat_id)
    {
      for (unsigned int k=0; k<points.size(); ++k)
      {
        values[k]=0;
      }
    }
    else
    {
      for (unsigned int k=0; k<points.size(); ++k)
      {
        const Point<dim> &p = points[k];
        // Work out position vector with respect to coil centre
        // and find the quadrant within the coil.
        const Point<dim> p0 (p[0] - coil_centre[0],
                             p[1] - coil_centre[1],
                             p[2] - coil_centre[2]);
                       
        const unsigned int coil_quadrant = get_quadrant(p0);
        // Work out position vector with respect to corner centre for this quadrant
        // and find the quadrant with respect to the corner centre.
        const Point<dim> p_corner (p[0] - corner_centres[coil_quadrant][0],
                                   p[1] - corner_centres[coil_quadrant][1],
                                   p[2] - corner_centres[coil_quadrant][2]);
        const unsigned int corner_quadrant = get_quadrant(p_corner);
        
        const double corner_angle = atan2(p_corner(1), p_corner(0));
        // Return the tangent to the coil given the quadrants (with respect to coil and the corner) and angle.
        const Point<dim> tangent_to_coil = get_tangent_by_quadrant(coil_quadrant,
                                                                   corner_quadrant,
                                                                   corner_angle);
        for (unsigned int d=0; d<dim; ++d)
        {
          values[k](d) = current_magnitude*tangent_to_coil[d];
          values[k](d+dim) = 0.0;
        }
      }  
    }
  }
  
  
  template<int dim>
  unsigned int TEAMBenchmark7<dim>::get_quadrant(const Point<dim> &p) const
  {
    // Returns the quadrant the given point lies within using atan2
    // NOTE, WE NUMBER STARTING FROM 0.
    // Quadrant 0 lies between -pi and -pi/2 (lower left),
    // then we proceed anti-clockwise.
    const double pi = numbers::PI;
    const double piby2 = numbers::PI/2.0;
    const double theta = atan2(p(1), p(0));
    if (-pi <= theta && theta < -piby2)
    {
      return 0;
    }
    else if (-piby2 <= theta && theta < 0)
    {
      return 1;
    }
    else if (0 <= theta && theta < piby2)
    {
      return 2;
    }
    else
    {
      return 3;
    }
  }
  
  template<int dim>
  Point<dim> TEAMBenchmark7<dim>::get_tangent_by_quadrant(const unsigned int &coil_quadrant,
                                                          const unsigned int &corner_quadrant,
                                                          const double &corner_angle) const
  {
    // Returns the tangent to the TEAMBenchmark7 coil (square with rounded corners).
    switch (coil_quadrant) {
      case 0: {
        switch (corner_quadrant) {
          case 0: {
            return Point<dim> (-sin(corner_angle), cos(corner_angle), 0.0);
          }
          case 1: {
            return Point<dim> (1.0, 0.0, 0.0);
          }
          // case 2 not possible
          case 3: {
            return Point<dim> (0.0, -1.0, 0.0);
          }
          default: {
            return Point<dim> (0.0, 0.0, 0.0);
          }
        }
      }
      case 1: {
        switch (corner_quadrant) {
          case 0: {
            return Point<dim> (1.0, 0.0, 0.0);
          }
          case 1: {
            return Point<dim> (-sin(corner_angle), cos(corner_angle), 0.0);
          }
          case 2: {
            return Point<dim> (0.0, 1.0, 0.0);
          }
          // case 3 not possible.
          default: {
            return Point<dim> (0.0, 0.0, 0.0);
          }
        }
      }
      case 2:{
        switch (corner_quadrant) {
          // case 0 not possible.
          case 1: {
            return Point<dim> (0.0, 1.0, 0.0);
          }
          case 2: {
            return Point<dim> (-sin(corner_angle), cos(corner_angle), 0.0);
          }
          case 3: {
            return Point<dim> (-1.0, 0.0, 0.0);
          }
          default: {
            return Point<dim> (0.0, 0.0, 0.0);
          }
        }
      }
      case 3:
      {
        switch (corner_quadrant) {
          case 0: {
            return Point<dim> (0.0, -1.0, 0.0);
          }           
          // case 1 not possible.
          case 2: {
            return Point<dim> (-1.0, 0.0, 0.0);
          }
          case 3: {
            return Point<dim> (-sin(corner_angle), cos(corner_angle), 0.0);
          }
          default: {
            return Point<dim> (0.0, 0.0, 0.0);
          }
        }
      }
      default: {
        return Point<dim> (0.0, 0.0, 0.0);
      }
    }
  }
  template class TEAMBenchmark7<3>;

}