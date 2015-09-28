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

#include <new_voltages.h>

using namespace dealii;

// This is a namespace containing the class Voltage.
// This class implements different methods of returning an array of calculated voltages for a 
// set of analytical or FE solutions on set of sensor locations.
namespace NewVoltages
{
  // VOLTAGE1 CLASS
  /* NOT IMPLEMENTED
  template<int dim>
  void Voltage1<dim>::calculateVoltages(const InverseSolver_Voltages::InternalData<dim> &inverse_data,
                                       const ForwardSolver::EddyCurrentData &fe_data,
                                       const std::vector<backgroundField::conductingSphereDipole<dim>> functions,
                                       const DoFHandler<dim> &dof_handler,
                                       std::vector<std::vector<std::vector<double>>> &calculated_voltages) const
  {
  }
  */
  // VOLTAGE2 CLASS
  // constructor:
  template<int dim>
  Voltage2<dim>::Voltage2 (const unsigned int &quad_order)
  :
  quad_order(quad_order)
  {
    // Setup standard coil centred at (0,0,0) with radius 1, direction straight up (0,0,1)
    // and with points mapped from a QGauss quad rule (0,1) to the disc (-pi,pi)
    // For the actual computation of the voltage these will be used through rotation/translation.
    
    // Gauss quad rule:
    QGauss<1> reference_quadrature(quad_order);
    n_quadrature_points = reference_quadrature.size();

    // Map quad points from 0, 1 to -pi, pi.
    const std::vector<Point<1>> temp_quad_points (reference_quadrature.get_points());
    
    // Gauss weights:
    reference_quad_weights.resize(n_quadrature_points);
    reference_quad_weights = reference_quadrature.get_weights();
    
    // Jacobian (from straight line to coil of radius 1).
    reference_jacobian = 2.0*numbers::PI;
    
    // Calculate quadrature angle and points
    // and tangent vectors to the disc at these points.:
    
    reference_quad_angles.reinit(n_quadrature_points);
    reference_quad_points.resize(n_quadrature_points);
    reference_tangent_vectors.resize(n_quadrature_points);
    
    for (unsigned int i = 0; i < n_quadrature_points; ++i)
    {
      // Gauss point angles:
      const double reference_quad_angle = numbers::PI*(2.0*temp_quad_points[i](0) - 1.0);
      
      // Points:
      reference_quad_points[i](0) = cos(reference_quad_angle);
      reference_quad_points[i](1) = sin(reference_quad_angle);
      reference_quad_points[i](2) = 0.0;
      
      // Tangents:
      reference_tangent_vectors[i][0] = -sin(reference_quad_angle);
      reference_tangent_vectors[i][1] =  cos(reference_quad_angle);
      reference_tangent_vectors[i][2] =  0.0;
    }
    // Store reference coil direction
    reference_coil_direction[0] = 0.0;
    reference_coil_direction[1] = 0.0;
    reference_coil_direction[2] = 1.0;
  }
  // private member function to return voltage for a given excitation/sensor combination
  // using the analytical function.
  template<int dim>
  std::complex<double> Voltage2<dim>::returnVoltage(const double &sensor_coil_radius,
                                                    const Point<dim> &sensor_coil_centre,
                                                    const Tensor<1,dim> &sensor_coil_direction,
                                                    const backgroundField::conductingSphereDipole<dim> &solution_function) const
  {
    // First need to work out the angle and axis of rotation
    Tensor<1, dim> normalised_coil_direction(sensor_coil_direction/sensor_coil_direction.norm());
    const double rotation_angle = acos(normalised_coil_direction*reference_coil_direction);
    
    // rotation axis:
    Tensor<1, dim> rotation_axis;
    cross_product(rotation_axis,
                  reference_coil_direction,
                  normalised_coil_direction);
    rotation_axis = rotation_axis/rotation_axis.norm ();
    
    // Now find the physical quadrature points and tangents at these points.
    std::vector< Point<dim> > coil_quad_points(n_quadrature_points);
    std::vector< Tensor<1,dim> > coil_tangent_vectors(n_quadrature_points);
    for (unsigned int q=0; q<n_quadrature_points; ++q)
    {
      Point<dim> scaled_ref_quad_point = sensor_coil_radius*reference_quad_points[q];
      Point<dim> temp_point(MyVectorTools::rotate_point(scaled_ref_quad_point,
                                                        rotation_axis,
                                                        rotation_angle));
      coil_quad_points[q] = temp_point + sensor_coil_centre;
      
      Tensor<1, dim> temp_tensor(MyVectorTools::rotate_tensor(reference_tangent_vectors[q],
                                                              rotation_axis,
                                                              rotation_angle));
      coil_tangent_vectors[q] = temp_tensor;
    }
    const double jacobian = sensor_coil_radius*reference_jacobian;
    
    // Now compute the integral:
    // First find the field values at all points:
    std::vector<Vector<double>> E_scattered_values(n_quadrature_points, Vector<double> (dim+dim));
    solution_function.scattered_field_value_list(coil_quad_points,
                                                 E_scattered_values,1);
    std::vector<double> integral_sum(2);
    integral_sum[0] = 0.0;
    integral_sum[1] = 0.0;
    for (unsigned int q=0; q<n_quadrature_points; ++q)
    {
      // Pull out real/imaginary parts of solution:
      std::vector<Tensor<1,dim>> solution_q (2);
      for (unsigned int c=0; c<2; ++c)
      {
        for (unsigned int d=0; d<dim; ++d)
        {
          solution_q[c][d] = E_scattered_values[q](d+c*dim);
        }
      }
      for (unsigned int c=0; c<2; ++c)
      {
        integral_sum[c] += solution_q[c]*coil_tangent_vectors[q]*reference_quad_weights[q];
      }
    }
    for (unsigned int c=0; c<2; ++c)
    {
      integral_sum[c] *= jacobian;
    }
    // convert to complex<double> & return
    std::complex<double> output(integral_sum[0], integral_sum[1]);
    return output; 
  }
  
  template<int dim>
  std::complex<double> Voltage2<dim>::returnVoltage(const double &sensor_coil_radius,
                                                    const Point<dim> &sensor_coil_centre,
                                                    const Tensor<1,dim> &sensor_coil_direction,
                                                    const Vector<double> &FEsolution,
                                                    const Mapping<dim> &mapping,
                                                    const DoFHandler<dim> &dof_handler) const
  {
    // First need to work out the angle and axis of rotation
    Tensor<1, dim> normalised_coil_direction(sensor_coil_direction/sensor_coil_direction.norm());
    const double rotation_angle = acos(normalised_coil_direction*reference_coil_direction);
   
    // rotation axis:
    Tensor<1, dim> rotation_axis;
    cross_product(rotation_axis,
                  reference_coil_direction,
                  normalised_coil_direction);
    rotation_axis = rotation_axis/rotation_axis.norm ();
    
    // Now find the physical quadrature points and tangents at these points.
    std::vector< Point<dim> > coil_quad_points(n_quadrature_points);
    std::vector< Tensor<1,dim> > coil_tangent_vectors(n_quadrature_points);
    for (unsigned int q=0; q<n_quadrature_points; ++q)
    {
      Point<dim> scaled_ref_quad_point = sensor_coil_radius*reference_quad_points[q];
      Point<dim> temp_point(MyVectorTools::rotate_point(scaled_ref_quad_point,
                                                        rotation_axis,
                                                        rotation_angle));
      coil_quad_points[q] = temp_point + sensor_coil_centre;
      
      Tensor<1, dim> temp_tensor(MyVectorTools::rotate_tensor(reference_tangent_vectors[q],
                                                              rotation_axis,
                                                              rotation_angle));
      coil_tangent_vectors[q] = temp_tensor;
    }
    const double jacobian = sensor_coil_radius*reference_jacobian;
    
    // Now compute the integral:
    std::vector<double> integral_sum(2);
    integral_sum[0] = 0.0;
    integral_sum[1] = 0.0;
    for (unsigned int q=0; q<n_quadrature_points; ++q)
    {
      // Recover solution values at the quadrature point:
      Vector<double> solution_at_quad_point(dof_handler.get_fe().n_components());
      VectorTools::point_value (mapping,
                                dof_handler,
                                FEsolution,
                                coil_quad_points[q],
                                solution_at_quad_point);
      // Pull out real/imaginary parts of solution:
      std::vector<Tensor<1,dim>> solution_q (2);
      for (unsigned int c=0; c<2; ++c)
      {
        for (unsigned int d=0; d<dim; ++d)
        {
        solution_q[c][d] = solution_at_quad_point(d+c*dim);
        }
      }
      for (unsigned int c=0; c<2; ++c)
      {
        integral_sum[c] += solution_q[c]*coil_tangent_vectors[q]*reference_quad_weights[q];
      }
    }
    for (unsigned int c=0; c<2; ++c)
    {
      integral_sum[c] *= jacobian;
    }
    // convert to complex<double> & return
    std::complex<double> output(integral_sum[0], integral_sum[1]);
    return output; 
  }
  
  template<int dim>
  std::complex<double> Voltage2<dim>::returnVoltage(const double &sensor_coil_radius,
                                                    const Point<dim> &sensor_coil_centre,
                                                    const Tensor<1,dim> &sensor_coil_direction,
                                                    const Vector<double> &FEsolution,
                                                    const DoFHandler<dim> &dof_handler) const
  {
    return returnVoltage(sensor_coil_radius,
                         sensor_coil_centre,
                         sensor_coil_direction,
                         FEsolution,
                         StaticMappingQ1<dim>::mapping,
                         dof_handler);
  }
  
  // Calculate voltage for analytical function
  // Note, there is no mapping required, but we provide a function for one anyway to
  // avoid mistakes when testing.
  template<int dim>
  void Voltage2<dim>::calculateVoltages(const InverseSolver_Voltages::InternalData<dim> &inverse_data,
                                        const ForwardSolver::EddyCurrentData &fe_data,
                                        const std::vector<backgroundField::conductingSphereDipole<dim>> functions,
                                        const DoFHandler<dim> &dof_handler,
                                        const Mapping<dim> &mapping,
                                        std::vector<std::vector<std::vector<double>>> &calculated_voltages) const
  {
    calculateVoltages(inverse_data,
                      fe_data,
                      functions,
                      dof_handler,
                      calculated_voltages);
  }  
  template<int dim>
  void Voltage2<dim>::calculateVoltages(const InverseSolver_Voltages::InternalData<dim> &inverse_data,
                                        const ForwardSolver::EddyCurrentData &fe_data,
                                        const std::vector<backgroundField::conductingSphereDipole<dim>> functions,
                                        const DoFHandler<dim> &dof_handler,
                                        std::vector<std::vector<std::vector<double>>> &calculated_voltages) const
  {
    Assert(functions.size() == inverse_data.number_of_excitations,
           ExcDimensionMismatch(functions.size(),
                                inverse_data.number_of_excitations));
    // resize the voltage vector:
    calculated_voltages.resize(inverse_data.number_of_excitations);
    for (unsigned int exciter=0; exciter<calculated_voltages.size(); ++exciter)
    {
      calculated_voltages[exciter].resize(inverse_data.number_of_measurements);
      for (unsigned int sensor=0; sensor<calculated_voltages[exciter].size(); ++sensor)
      {
        const std::complex<double> solution_voltage = returnVoltage(inverse_data.measurement_coil_radius[sensor],
                                                                    inverse_data.measurement_positions[sensor],
                                                                    inverse_data.measurement_directions[sensor],
                                                                    functions[exciter]);
        calculated_voltages[exciter][sensor].resize(2);
        calculated_voltages[exciter][sensor][0] = solution_voltage.real();
        calculated_voltages[exciter][sensor][1] = solution_voltage.imag();
      }
    }
  }
  
  // Calculate voltage for FE solution
  template<int dim>
  void Voltage2<dim>::calculateVoltages(const InverseSolver_Voltages::InternalData<dim> &inverse_data,
                                        const ForwardSolver::EddyCurrentData &fe_data,
                                        const std::vector<Vector<double>> &FEsolutions,
                                        const DoFHandler<dim> &dof_handler,
                                        const Mapping<dim> &mapping,
                                        std::vector<std::vector<std::vector<double>>> &calculated_voltages) const
  {
    Assert(FEsolutions.size() == inverse_data.number_of_excitations,
           ExcDimensionMismatch(FEsolutions.size(),
                                inverse_data.number_of_excitations));
    // resize the voltage vector:
    calculated_voltages.resize(inverse_data.number_of_excitations);
    for (unsigned int exciter=0; exciter<calculated_voltages.size(); ++exciter)
    {
      calculated_voltages[exciter].resize(inverse_data.number_of_measurements);
      for (unsigned int sensor=0; sensor<calculated_voltages[exciter].size(); ++sensor)
      {
        const std::complex<double> solution_voltage = returnVoltage(inverse_data.measurement_coil_radius[sensor],
                                                                    inverse_data.measurement_positions[sensor],
                                                                    inverse_data.measurement_directions[sensor],                                                                    
                                                                    FEsolutions[exciter],
                                                                    mapping,
                                                                    dof_handler);
        calculated_voltages[exciter][sensor].resize(2);
        calculated_voltages[exciter][sensor][0] = solution_voltage.real();
        calculated_voltages[exciter][sensor][1] = solution_voltage.imag();
      }
    }
  }
  // no mapping
  template<int dim>
  void Voltage2<dim>::calculateVoltages(const InverseSolver_Voltages::InternalData<dim> &inverse_data,
                                        const ForwardSolver::EddyCurrentData &fe_data,
                                        const std::vector<Vector<double>> &FEsolutions,
                                        const DoFHandler<dim> &dof_handler,
                                        std::vector<std::vector<std::vector<double>>> &calculated_voltages) const
  {
    Assert(FEsolutions.size() == inverse_data.number_of_excitations,
           ExcDimensionMismatch(FEsolutions.size(),
                                inverse_data.number_of_excitations));
    // resize the voltage vector:
    calculated_voltages.resize(inverse_data.number_of_excitations);
    for (unsigned int exciter=0; exciter<calculated_voltages.size(); ++exciter)
    {
      calculated_voltages[exciter].resize(inverse_data.number_of_measurements);
      for (unsigned int sensor=0; sensor<calculated_voltages[exciter].size(); ++sensor)
      {
        const std::complex<double> solution_voltage = returnVoltage(inverse_data.measurement_coil_radius[sensor],
                                                                    inverse_data.measurement_positions[sensor],
                                                                    inverse_data.measurement_directions[sensor],
                                                                    FEsolutions[exciter],
                                                                    dof_handler);
        calculated_voltages[exciter][sensor].resize(2);
        calculated_voltages[exciter][sensor][0] = solution_voltage.real();
        calculated_voltages[exciter][sensor][1] = solution_voltage.imag();
      }
    }
  }
  template class Voltage2<3>;
  // END VOLTAGE2.
  
  // CLASS VOLTAGE4
  template<int dim>
  Voltage4<dim>::Voltage4 (const unsigned int &quad_order)
  :
  quad_order(quad_order)
  {
  }
  template<int dim>
  void Voltage4<dim>::calculateVoltages(const InverseSolver_Voltages::InternalData<dim> &inverse_data,
                                        const ForwardSolver::EddyCurrentData &fe_data,
                                        const std::vector<backgroundField::conductingSphereDipole<dim>> functions,
                                        const DoFHandler<dim> &dof_handler,
                                        const Mapping<dim> &mapping,
                                        std::vector<std::vector<std::vector<double>>> &calculated_voltages) const
  {
    // Routine to caluculate the voltage difference using an alternative method to
    // a surface contour integral.
    // NOTE: We may calculate the voltage difference in measurement coil i, centred at x, through:
    //
    // \Delta V|_{i}(x) = m_{i}*H_{\Delta}(x)
    //
    // which comes from the below:
    // 
    // To calculate the voltage in coil i centred at the point x
    // for the excitation coil j, we use:
    
    // V|_{i} (x) = m_{i}*( H^{j}_{0}(x) + H_{\Delta}(x) )
    //
    // - H^{j}_{0} = D^{2}G(x,z)*m_{e} is the magnetic field for the exciter coil.
    // - m_{i} is the dipole moment of the measurement coil i.
    // - H_{\Delta}(x) is the magnetic field due to the presence of the conductor
    // - the conducting region is denoted \Omega_{c}
    //
    // H_{\Delta} (x) = -1/(4*pi)*\int_{\Omega_{c}} (x-y)/(|x-y|^{3})*J(y)dy
    //
    // here J(y) = sigma*E=-i*omega*sigma*A. (i.e. the eddy current).
    //
    // i.e. x-y is the radial vector from the sensor coil.
    //      Since the outer region lies in a region with sigma=0, outside \Omega_{c}.
    
    // Check that all the vector input make sense:
    // Let S=number of sensor coils
    //     E=number of excitation coils.
    //
    // we want:
    // sensor centres & all_sensor_directions to be of length S
    // fesolutions & all_boundary_functions to be of length E
    //
    // We resize the output vectors accordingly.
    
    // Check that the solutions given match with the inverse_data structure
    Assert(functions.size() == inverse_data.number_of_excitations,
           ExcDimensionMismatch(functions.size(),
                                inverse_data.number_of_excitations));

    // Setup a dummy fe (could also use one contained in the dof_handler).
//     const FiniteElement<dim> &fe = dof_handler.get_fe ();
    const FE_Q<dim> dummy_fe(1);
    QGauss<dim>  quadrature_formula(quad_order);
    const unsigned int n_q_points = quadrature_formula.size();
    FEValues<dim> fe_values(mapping, dummy_fe, quadrature_formula,
                            update_quadrature_points | update_JxW_values);

    std::vector<FEValuesExtractors::Vector> vec;
    vec.reserve(2);
    vec.push_back(FEValuesExtractors::Vector (0));
    vec.push_back(FEValuesExtractors::Vector (dim));
    
    std::vector<std::vector<std::vector<Tensor<1,dim>>>> all_hdelta(inverse_data.number_of_excitations);
    // set size and zero the integral storage.
    for (unsigned int exciter=0; exciter<all_hdelta.size(); ++exciter)
    {
      all_hdelta[exciter].resize(inverse_data.number_of_measurements);
      for (unsigned int sensor=0; sensor<all_hdelta[exciter].size(); ++sensor)
      {
        all_hdelta[exciter][sensor].resize(2);
        for (unsigned int c=0; c<all_hdelta[exciter][sensor].size(); ++c)
        {
          all_hdelta[exciter][sensor][c]=0.;
        }
      }
    }
    
    typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active();
    const typename DoFHandler<dim>::active_cell_iterator endc = dof_handler.end();
    for (; cell!=endc; ++cell)
    {
      // Check if in conducting cell
      if (fe_data.conducting_material[cell->material_id()])
      {
        const double current_sigma = fe_data.param_sigma[cell->material_id()];
        fe_values.reinit (cell);
        // Get the value of J for each excitation at each cell quad point
        // Note, J = sigma*E
        // if we have A, E=-i*omega*A
        // i.e. J = -i*omega*sigma*A.
        
        // Store all quadrature points:
        const std::vector<Point<dim>> all_quad_points = fe_values.get_quadrature_points();
        
        // Store the solution from the analytical function
        std::vector<std::vector<std::vector<Tensor<1,dim>>>> cell_solutions(inverse_data.number_of_excitations);
        for (unsigned int exciter=0; exciter<cell_solutions.size(); ++exciter)
        {
          cell_solutions[exciter].resize(n_q_points);
          // Calculate values at quad points for this exciter.
          std::vector<Vector<double>> E_values(n_q_points, Vector<double> (dim+dim));
          functions[exciter].vector_value_list(all_quad_points,
                                               E_values,
                                               1);
          for (unsigned int q=0; q<cell_solutions[exciter].size(); ++q)
          {
            cell_solutions[exciter][q].resize(2);
            for (unsigned int c=0; c<2; ++c)
            {
              for (unsigned int d=0; d<dim; ++d)
              {
                cell_solutions[exciter][q][c][d] = E_values[q](c*dim + d);
              }  
            }
          }
        }
        // Loop over all quad points & calculate the contribution to each hdelta
        for (unsigned int sensor=0; sensor<inverse_data.number_of_measurements; ++sensor)
        {
          for (unsigned int q=0; q<n_q_points; ++q)
          {
            // calculate (x-y)/(|x-y|^{3}):
            const Tensor<1,dim> shifted_point
            = inverse_data.measurement_positions[sensor] - all_quad_points[q];
            
            const double rad = inverse_data.measurement_positions[sensor].distance(all_quad_points[q]);
//             const Tensor<1,dim> scaled_vector = shifted_point;
            for (unsigned int exciter=0; exciter<inverse_data.number_of_excitations; ++exciter)
            {
              for (unsigned int c=0; c<2; ++c)
              {
                Tensor<1,dim> cross_product_result;
                cross_product(cross_product_result,
                              shifted_point,
                              cell_solutions[exciter][q][c]);
                
                all_hdelta[exciter][sensor][c] += current_sigma*cross_product_result*fe_values.JxW(q)/(rad*rad*rad);
              }
            }
          }
        }
      }
    }
    // Now calculate voltage difference
    std::vector<double> integral_factor(2);
    integral_factor[0] = -fe_data.param_omega*EquationData::constant_mu0/(4.0*numbers::PI);
    integral_factor[1] = fe_data.param_omega*EquationData::constant_mu0/(4.0*numbers::PI);
    std::vector<unsigned int> voltage_component(2);
    voltage_component[0] = 1;
    voltage_component[1] = 0;
    // resize and fill the output array:
    calculated_voltages.resize(inverse_data.number_of_excitations);
    for (unsigned int exciter=0; exciter<calculated_voltages.size(); ++exciter)
    {
      calculated_voltages[exciter].resize(inverse_data.number_of_measurements);
      for (unsigned int sensor=0; sensor<calculated_voltages[exciter].size(); ++sensor)
      {
        calculated_voltages[exciter][sensor].resize(2);
        const double dipole_factor = numbers::PI*inverse_data.measurement_coil_radius[sensor]*inverse_data.measurement_coil_radius[sensor];
        for (unsigned int c=0; c<2; ++c)
        {
          calculated_voltages[exciter][sensor][c] = integral_factor[c]*dipole_factor*inverse_data.measurement_directions[sensor]*all_hdelta[exciter][sensor][voltage_component[c]];
        }
      }
    }
  }
                          
  template<int dim>
  void Voltage4<dim>::calculateVoltages(const InverseSolver_Voltages::InternalData<dim> &inverse_data,
                                        const ForwardSolver::EddyCurrentData &fe_data,
                                        const std::vector<Vector<double>> &FEsolutions,
                                        const DoFHandler<dim> &dof_handler,
                                        const Mapping<dim> &mapping,
                                        std::vector<std::vector<std::vector<double>>> &calculated_voltages) const
  {
    // Routine to caluculate the voltage difference using an alternative method to
    // a surface contour integral.
    // NOTE: We may calculate the voltage difference in measurement coil i, centred at x, through:
    //
    // \Delta V|_{i}(x) = m_{i}*H_{\Delta}(x)
    //
    // which comes from the below:
    // 
    // To calculate the voltage in coil i centred at the point x
    // for the excitation coil j, we use:
    
    // V|_{i} (x) = m_{i}*( H^{j}_{0}(x) + H_{\Delta}(x) )
    //
    // - H^{j}_{0} = D^{2}G(x,z)*m_{e} is the magnetic field for the exciter coil.
    // - m_{i} is the dipole moment of the measurement coil i.
    // - H_{\Delta}(x) is the magnetic field due to the presence of the conductor
    // - the conducting region is denoted \Omega_{c}
    //
    // H_{\Delta} (x) = -1/(4*pi)*\int_{\Omega_{c}} (x-y)/(|x-y|^{3})*J(y)dy
    //
    // here J(y) = sigma*E=-i*omega*sigma*A. (i.e. the eddy current).
    //
    // i.e. x-y is the radial vector from the sensor coil.
    //      Since the outer region lies in a region with sigma=0, outside \Omega_{c}.
    
    // Check that all the vector input make sense:
    // Let S=number of sensor coils
    //     E=number of excitation coils.
    //
    // we want:
    // sensor centres & all_sensor_directions to be of length S
    // fesolutions & all_boundary_functions to be of length E
    //
    // We resize the output vectors accordingly.
    
    // Check that the solutions given match with the inverse_data structure
    Assert(FEsolutions.size() == inverse_data.number_of_excitations,
           ExcDimensionMismatch(FEsolutions.size(),
                                inverse_data.number_of_excitations));
    
    // Now calculate each Hdelta:
    // First setup the finite element:
    const FiniteElement<dim> &fe = dof_handler.get_fe ();
    const unsigned int element_quad_order = 2*fe.degree + 1;
    QGauss<dim>  quadrature_formula(element_quad_order);
    const unsigned int n_q_points = quadrature_formula.size();
    FEValues<dim> fe_values(mapping, fe, quadrature_formula,
                            update_values | update_quadrature_points | update_JxW_values); // gradients not needed.

    std::vector<FEValuesExtractors::Vector> vec;
    vec.reserve(2);
    vec.push_back(FEValuesExtractors::Vector (0));
    vec.push_back(FEValuesExtractors::Vector (dim));
    
    std::vector<std::vector<std::vector<Tensor<1,dim>>>> all_hdelta(inverse_data.number_of_excitations);
    // set size and zero the integral storage.
    for (unsigned int exciter=0; exciter<all_hdelta.size(); ++exciter)
    {
      all_hdelta[exciter].resize(inverse_data.number_of_measurements);
      for (unsigned int sensor=0; sensor<all_hdelta[exciter].size(); ++sensor)
      {
        all_hdelta[exciter][sensor].resize(2);
        for (unsigned int c=0; c<all_hdelta[exciter][sensor].size(); ++c)
        {
          all_hdelta[exciter][sensor][c]=0.;
        }
      }
    }
    
    typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active();
    const typename DoFHandler<dim>::active_cell_iterator endc = dof_handler.end();
    for (; cell!=endc; ++cell)
    {
      // Check if in conducting cell
      if (fe_data.conducting_material[cell->material_id()])
      {
        const double current_sigma = fe_data.param_sigma[cell->material_id()];
        fe_values.reinit (cell);
        // Get the value of J for each excitation at each cell quad point
        // Note, J = sigma*E
        // but we have A, E=-i*omega*A
        // so J = -i*omega*sigma*A.
        
        // We can consider the real and imaginary parts independently.
        // Do this differently to the function version as calculating the solution at
        // quad points is easier this way.
        std::vector<std::vector<std::vector<Tensor<1,dim>>>> cell_solutions(2);
        for (unsigned int c=0; c<cell_solutions.size(); ++c)
        {
          cell_solutions[c].resize(inverse_data.number_of_excitations);
          for (unsigned int exciter=0; exciter<cell_solutions[c].size(); ++exciter)
          {
            cell_solutions[c][exciter].resize(n_q_points);
            // Calculate real & imaginary parts of E 
            // TODO: Should we switch A im & re to get E here??.
            fe_values[vec[c]].get_function_values(FEsolutions[exciter],
                                                  cell_solutions[c][exciter]);
          }
        }
        // Store all quadrature points:
        const std::vector<Point<dim>> all_quad_points = fe_values.get_quadrature_points();
        
        // Loop over all quad points & calculate the contribution to each hdelta
        for (unsigned int sensor=0; sensor<inverse_data.number_of_measurements; ++sensor)
        {
          for (unsigned int q=0; q<n_q_points; ++q)
          {
            // calculate (x-y)/(|x-y|^{3}):
            const Tensor<1,dim> shifted_point
            = inverse_data.measurement_positions[sensor] - all_quad_points[q];
            
            const double rad = inverse_data.measurement_positions[sensor].distance(all_quad_points[q]);
            const Tensor<1,dim> scaled_vector = shifted_point/(rad*rad*rad);
            for (unsigned int exciter=0; exciter<inverse_data.number_of_excitations; ++exciter)
            {
              for (unsigned int c=0; c<2; ++c)
              {
                Tensor<1,dim> cross_product_result;
                cross_product(cross_product_result,
                              scaled_vector,
                              cell_solutions[c][exciter][q]);
                
                all_hdelta[exciter][sensor][c] += current_sigma*cross_product_result*fe_values.JxW(q);
              }
            }
          }
        }
      }
    }
    // Now calculate voltage difference
    std::vector<double> integral_factor(2);
    integral_factor[0] = -fe_data.param_omega*EquationData::constant_mu0/(4.0*numbers::PI);
    integral_factor[1] = fe_data.param_omega*EquationData::constant_mu0/(4.0*numbers::PI);
    std::vector<unsigned int> voltage_component(2);
    voltage_component[0] = 1;
    voltage_component[1] = 0;
    
    // resize and fill the output array:
    calculated_voltages.resize(inverse_data.number_of_excitations);
    for (unsigned int exciter=0; exciter<calculated_voltages.size(); ++exciter)
    {
      calculated_voltages[exciter].resize(inverse_data.number_of_measurements);
      for (unsigned int sensor=0; sensor<calculated_voltages[exciter].size(); ++sensor)
      {
        const double dipole_factor = numbers::PI*inverse_data.measurement_coil_radius[sensor]*inverse_data.measurement_coil_radius[sensor];
        calculated_voltages[exciter][sensor].resize(2);
        for (unsigned int c=0; c<2; ++c)
        {
          calculated_voltages[exciter][sensor][c] = integral_factor[c]*dipole_factor*inverse_data.measurement_directions[sensor]*all_hdelta[exciter][sensor][voltage_component[c]];
        }  
      }
    }
  }
  // No mapping versions
  template<int dim>
  void Voltage4<dim>::calculateVoltages(const InverseSolver_Voltages::InternalData<dim> &inverse_data,
                                        const ForwardSolver::EddyCurrentData &fe_data,
                                        const std::vector<backgroundField::conductingSphereDipole<dim>> functions,
                                        const DoFHandler<dim> &dof_handler,
                                        std::vector<std::vector<std::vector<double>>> &calculated_voltages) const
  {
    calculateVoltages(inverse_data,
                      fe_data,
                      functions,
                      dof_handler,
                      StaticMappingQ1<dim>::mapping,
                      calculated_voltages);
  }
  template<int dim>
  void Voltage4<dim>::calculateVoltages(const InverseSolver_Voltages::InternalData<dim> &inverse_data,
                                        const ForwardSolver::EddyCurrentData &fe_data,
                                        const std::vector<Vector<double>> &FEsolutions,
                                        const DoFHandler<dim> &dof_handler,
                                        std::vector<std::vector<std::vector<double>>> &calculated_voltages) const
  {
    calculateVoltages(inverse_data,
                      fe_data,
                      FEsolutions,
                      dof_handler,
                      StaticMappingQ1<dim>::mapping,
                      calculated_voltages);
                      
  }
  template class Voltage4<3>;
  // END VOLTAGE4.
}