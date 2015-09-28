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

#include <inversesolver_voltages.h>

using namespace dealii;

namespace InverseSolver_Voltages
{
  // TEMPORARY CLASS.
  // TODO: Move once this is definitely working.
  template<int dim>
  class Voltage
  {
    // Class to compute the voltage in a coil induced by some electric field.
    // TODO: Think the returned answer should be multiplied by i*omega, but need to check this.
  public:
    Voltage(const unsigned int &quad_order = 20);
    
    std::complex<double> calculateVoltage(const double &sensor_coil_radius,
                                          const Point<dim> &sensor_coil_centre,
                                          const Tensor<1,dim> &sensor_coil_direction,
                                          const backgroundField::conductingSphereDipole<dim> &solution_function) const;

  private:
    // Functions:
    Tensor<1,dim> rotate_coil(const Tensor<1, dim> &input_point,
                              const Tensor<1, dim> &axis,
                              const double &angle) const;
    
    
    // Data:
    const unsigned int quad_order;
    unsigned int n_quadrature_points;
    
    // Reference sensor coil data:
    std::vector<double> reference_quad_weights;    
    std::vector< Point<dim> > reference_quad_points;
    std::vector< Tensor<1,dim> > reference_tangent_vectors;
    Vector<double> reference_quad_angles; // probs not needed.
    double reference_jacobian;
    Tensor<1,3> reference_coil_direction;
  };
  
  template<int dim>
  Voltage<dim>::Voltage(const unsigned int &quad_order)
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
  // Members:
  // TODO: Remove this and make use of the function in MyVectorTools.
  template <int dim>
  Tensor<1,dim> Voltage<dim>::rotate_coil(const Tensor<1,dim> &input_tensor,
                                          const Tensor<1,dim> &axis,
                                          const double &angle) const
  {
    // Rotates a vector (point) about an axis by an angle using
    // Rodrigues rotation formula:
    // (rotating vector v about an axis k at angle th)
    //    v_rot = v*cos(th) + cross(k,v)*sin(th) + k*dot(k,v)*(1-cos(th)).
    
    if (abs(angle) < 1e-15)
    {
      return Tensor<1,dim> (input_tensor);
    }
    Tensor<1,dim> cross_prod;
    cross_product(cross_prod, axis, input_tensor);
    Tensor<1,dim> output_tensor(
      input_tensor*cos(angle)
      + cross_prod*sin(angle)
      + axis*(axis*input_tensor)*(1.0-cos(angle))
    );
    return output_tensor;
  }
  template<int dim>
  std::complex<double> Voltage<dim>::calculateVoltage(const double &sensor_coil_radius,
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
      Point<dim> temp_point(rotate_coil(reference_quad_points[q], rotation_axis, rotation_angle));
      coil_quad_points[q] = temp_point + sensor_coil_centre;
      
      Tensor<1, dim> temp_tensor(rotate_coil(reference_tangent_vectors[q], rotation_axis, rotation_angle));
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
  // END class Voltage
    
  // InverseSolver_Voltages class functions:  
  template<int dim, class DH>
  InverseSolver_Voltages<dim, DH>::InverseSolver_Voltages(const InternalData<dim> &data_in,
                                                          const DH &dof_handler,
                                                          const Mapping<dim> &mapping)
  :
  data(data_in)
  {
    // Store the measured voltages, which can be found from the functions in InternalData.
    assemble_rhs_data(measured_voltages,
                      data_in.measured_functions,
                      dof_handler,
                      mapping);
  }
  
  template<int dim, class DH>
  InverseSolver_Voltages<dim, DH>::InverseSolver_Voltages(const InternalData<dim> &data_in,
                                                          const std::vector<Vector<double>> &FEsolutions,
                                                          const ForwardSolver::EddyCurrentData &fe_data,
                                                          const DH &dof_handler,
                                                          const Mapping<dim> &mapping)
  :
  data(data_in)
  {
    // Store the measured voltages, which can be found from the FESolutions
    assemble_rhs_data(measured_voltages,
                      FEsolutions,
                      fe_data,
                      dof_handler,
                      mapping);
  }
  
  template<int dim, class DH>
  void InverseSolver_Voltages<dim, DH>::updateInternalData(const InternalData<dim> &data_in)
  {
    data = data_in;
  }
  
  template<int dim, class DH>
  void InverseSolver_Voltages<dim, DH>::assemble_rhs_data(std::vector<Vector<double>> &rhs_data,
                                                          const std::vector<backgroundField::conductingSphereDipole<dim>> &solution_functions,
                                                          const DH &dof_handler,
                                                          const Mapping<dim> &mapping)
  {
    // Returns the voltage measurement for a given set of solutions
    
    rhs_data.resize(2);
    rhs_data[0].reinit(data.number_of_combinations);
    rhs_data[1].reinit(data.number_of_combinations);
    
    // Calculate all voltages for the simulated solutions
    // then store the difference between this and the voltage for the measurement solutions (pre-computed by constructor).

    // needed? TODO: remove if not.
    //std::vector<std::complex<double>> simulated_voltages(data.number_of_combinations);
    // OLD WAY, remove later.
    Voltage<dim> voltage;
    unsigned int i_index = 0;
    for (unsigned int exciter=0; exciter < data.number_of_excitations; ++exciter)
    {
      for (unsigned int sensor=0; sensor < data.number_of_measurements; ++sensor)
      {
        if (exciter != sensor)
        {
          const std::complex<double> solution_voltage = voltage.calculateVoltage(data.measurement_coil_radius[sensor],
                                                                                 data.measurement_positions[sensor],
                                                                                 data.measurement_directions[sensor],
                                                                                 solution_functions[exciter]);
          
          rhs_data[0](i_index) = solution_voltage.real();
          rhs_data[1](i_index) = solution_voltage.imag();
          ++i_index;
        }
      }
    }
    // Use new version, requires the input include const ForwardSolver::EddyCurrentData &fe_data
    // For now this would break some older versions - will overload the function later maybe.    
//     NewVoltages::Voltage4<dim> voltage;
//     std::vector<std::vector<std::vector<double>>> simulated_voltages;
//     voltage.calculateVoltages(data,
//                               fe_data,
//                               sphere_functions,
//                               dof_handler,
//                               mapping,
//                               simulated_voltages);
//     unsigned int i_index = 0;
//     for (unsigned int exciter=0; exciter < data.number_of_excitations; ++exciter)
//     {
//       for (unsigned int sensor=0; sensor < data.number_of_measurements; ++sensor)
//       {
//         if (exciter != sensor)
//         {
//           rhs_data[0](i_index) = simulated_voltages[exciter][sensor][0];
//           rhs_data[1](i_index) = simulated_voltages[exciter][sensor][1];
//           ++i_index;
//         }
//       }
//     }
  }
  
  // Version where we supply the exact solution via an EddyCurrentFunction (i.e. no approximation used)
  template<int dim, class DH>
  void InverseSolver_Voltages<dim, DH>::assemble_sensitivity_matrix(const std::vector<backgroundField::conductingSphereDipole<dim>> &simulated_functions,
                                                               const DH &dof_handler,
                                                               const Mapping<dim> &mapping)
  {
    // Routine to assemble the sensitivity matrix for the MIT problem
    // This will be stored in the private FullMatrix, sensitivity_matrix.
    
    // This routine calculates the sensitivity matrix for the given
    // set of solutions and uses the DoFHandler to calculate the required integrals
    //
    // The entries are given by:
    // J_ij = \int_{vox_{j}} E^{e1} \cdot E^{e2} dV
    //
    // for e1 = 1, ... , data.number_of_excitations,
    // for e2 = 1, ... , data.number_of_measurements,
    // and e1 != e2
    // 
    // i = b + a*number_of_sensor_coils;
    
    // resize matrices:
    sensitivity_matrix.resize(2);
    for (unsigned int b=0; b<2; ++b)
    {
      sensitivity_matrix[b].reinit(data.number_of_combinations,
                                   data.n_voxels);
    }
    
    const QGauss<dim>  quadrature_formula(data.quad_order);
    const unsigned int n_q_points = quadrature_formula.size();    
    
//     const FE_Q<dim> dummy_fe(1);
    FEValues<dim> fe_values (mapping, dof_handler.get_fe(),
                             quadrature_formula,
                             update_quadrature_points  |  update_JxW_values);

    // Storage for computed solution on a cell at each quadrature point
    std::vector<std::vector<std::vector<Tensor<1,dim>>>>
    local_solutions(data.number_of_excitations,
                    std::vector<std::vector<Tensor<1,dim>>> (n_q_points,
                                                           std::vector<Tensor<1,dim>> (2) ));
    
    sensitivity_matrix[0] = 0.0;
    sensitivity_matrix[1] = 0.0;
    
    typename DH::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    
    for (; cell!=endc; ++cell)
    {
      // take note of which voxel we are in (via material_id)
      // If material_id is not in the recovery region, then skip.
      // The background region is hard-coded to be 0 in the setup region
      if (cell->material_id() > 0)
      {
        // Since 0 is the background region, take 1 from the material_id to get the voxel index
        // i.e. material_id 1 is voxel 0 & so on.
        // TODO: is there a better way to do this ??
        //
        // e.g. use a data.background_region_id and then adjust voxel index
        //      using vox_index = material_id() + (material_id() > background_region_id ? -1 : 0)
        //
        fe_values.reinit(cell);
        
        const unsigned int voxel_index = (unsigned int)cell->material_id() - 1;
        
        // within this cell, calculate solution values at all quadrature points:
        // For now we use the exact solution
        std::vector<Vector<double>> value_list(n_q_points, Vector<double> (dim+dim));
        for (unsigned int excitation=0; excitation < data.number_of_excitations; ++excitation)
        {
          simulated_functions[excitation].vector_value_list(fe_values.get_quadrature_points(),
                                                            value_list,
                                                            cell->material_id());
          for (unsigned int q=0; q<n_q_points; ++q)
          {
            for (unsigned int d=0; d<dim; ++d)
            {
              local_solutions[excitation][q][0][d] = value_list[q](d);
              local_solutions[excitation][q][1][d] = value_list[q](d+dim);
            }
          }
        }
        
        // NOTE: we ignore the (exciter==sensor) combination
        unsigned int i_index = 0;
        for (unsigned int exciter=0; exciter<data.number_of_excitations; ++exciter)
        {
          // Ignore case where e1=e2, so knock 1 off number of measurements.
          for (unsigned int sensor=0; sensor<data.number_of_measurements; ++sensor)
          {
            if (sensor != exciter)
            {
              double temp_re=0.0;
              double temp_im=0.0;
              for (unsigned int q=0; q<n_q_points; ++q)
              {
                temp_re += (
                  local_solutions[exciter][q][0]*local_solutions[sensor][q][0]
                  - local_solutions[exciter][q][1]*local_solutions[sensor][q][1]
                )*fe_values.JxW(q);
                
                temp_im += (
                  local_solutions[exciter][q][0]*local_solutions[sensor][q][1]
                  + local_solutions[exciter][q][1]*local_solutions[sensor][q][0]
                )*fe_values.JxW(q);
              }
              // We are solving the "A-based" formulation, where E = -i*omega*A,so we must
              // multiply each A by -i*omega to get the electric field.
              // Create a factor to multiply the sensitivity matrix by to get this
              // Note: this could go outside the loop involving voxels, but we may want to consider
              //       ways to use multiple frequencies in the future (not compatible in the current sensitivity though).
              double sensitivity_factor = 1.0;//-data.omega*data.omega;
              sensitivity_matrix[0](i_index, voxel_index) += sensitivity_factor*temp_re;
              sensitivity_matrix[1](i_index, voxel_index) += sensitivity_factor*temp_im;
              ++i_index;              
            }
          }
        }
      }
    }
  }
  
  template<int dim, class DH>
  void InverseSolver_Voltages<dim, DH>::assemble_sensitivity_rhs(const std::vector<backgroundField::conductingSphereDipole<dim>> &simulated_functions,
                                                                 const DH &dof_handler,
                                                                 const Mapping<dim> &mapping)
  {
    // Construct the RHS of the sensitivity system. This is:simulated_functions
    // The voltage difference between the measured and simulated voltages.
    // Later on it will be used as:
    //
    // -1/2*(J^{H}*V + J^{T}*V^{*}).
    //
    // in the gauss-newton iteration, where J is the sensitivity matrix calculated by assemble_sensitivity_matrix()
    // and V is the voltage difference between the image measured voltages and the
    // current simulated voltages.
    //
    // ^{*} denotes complex conjugate and ^{H} denotes hermitian (conjugate transpose).
    //
    // Care must be taken to match the exciter/sensor coil combinations of the voltage
    // to that which was used in the assemble_sensitivity_matrix() routine.    
    
    sensitivity_rhs.resize(2);
    sensitivity_rhs[0].reinit(data.number_of_combinations);
    sensitivity_rhs[1].reinit(data.number_of_combinations);
    
    // Call the update data function:
    std::vector<Vector<double>> simulated_voltages;
    assemble_rhs_data(simulated_voltages,
                      simulated_functions,
                      dof_handler,
                      mapping);
    
    // Sens rhs is just the measured result minus the simulated result.
    // Must also account for the fact that we have A and B (E=-i*omega*sigma, B=curl(A), H = (1/mu)*B
    // Since the rhs data is = int_{\partial\Omega} E^{i}\cdot(H_{0}^{j} \times n) d\Omega,
    // then we need to multiply the resulting integral by -i*omega*(1/mu)
    // TODO: Make sure the above is taken care of (IF IT NEEDS TO BE ??).
    unsigned int i_index = 0;
    for (unsigned int exciter=0; exciter < data.number_of_excitations; ++exciter)
    {
      for (unsigned int sensor=0; sensor < data.number_of_measurements; ++sensor)
      {
        if (exciter != sensor)
        {
          for (unsigned int c=0; c<2; ++c)
          {
            sensitivity_rhs[c](i_index) = simulated_voltages[c](i_index) - measured_voltages[c](i_index);
          }
          ++i_index;
        }
      }
    }
  }
  
  // Version where we supply a set of FE solutions.
  template<int dim, class DH>
  void InverseSolver_Voltages<dim, DH>::assemble_rhs_data(std::vector<Vector<double>> &rhs_data,
                                                          const std::vector<Vector<double>> &FEsolutions,
                                                          const ForwardSolver::EddyCurrentData &fe_data,
                                                          const DH &dof_handler,
                                                          const Mapping<dim> &mapping)
  {
    // Returns the voltage measurement for a given set of solutions    
    rhs_data.resize(2);
    rhs_data[0].reinit(data.number_of_combinations);
    rhs_data[1].reinit(data.number_of_combinations);
    
    // Calculate all voltages for the simulated solutions
    // then store the difference between this and the voltage for the measurement solutions (pre-computed by constructor).

    // needed? TODO: remove if not.
    std::vector<std::vector<std::vector<double>>> simulated_voltages;    
    NewVoltages::Voltage4<dim> voltage;
    voltage.calculateVoltages(data,
                              fe_data,
                              FEsolutions,
                              dof_handler,
                              mapping,
                              simulated_voltages);
    unsigned int i_index = 0;
    for (unsigned int exciter=0; exciter < data.number_of_excitations; ++exciter)
    {
      for (unsigned int sensor=0; sensor < data.number_of_measurements; ++sensor)
      {
        if (exciter != sensor)
        {
          rhs_data[0](i_index) = simulated_voltages[exciter][sensor][0];
          rhs_data[1](i_index) = simulated_voltages[exciter][sensor][1];
          ++i_index;
        }
      }
    }
  }
  
  template<int dim, class DH>
  void InverseSolver_Voltages<dim, DH>::assemble_sensitivity_matrix(/*const InternalData<dim> &inverse_data,
                                                                    const ForwardSolver::EddyCurrentData &fe_data,*/
                                                                    const std::vector<Vector<double>> &FEsolutions,
                                                                    const DH &dof_handler,
                                                                    const Mapping<dim> &mapping)
  {
    // Routine to assemble the sensitivity matrix for the MIT problem
    // This will be stored in the private FullMatrix, sensitivity_matrix.
    
    // This routine calculates the sensitivity matrix for the given
    // set of solutions and uses the DoFHandler to calculate the required integrals
    //
    // The entries are given by:
    // J_ij = \int_{vox_{j}} E^{e1} \cdot E^{e2} dV
    //
    // for e1 = 1, ... , data.number_of_excitations,
    // for e2 = 1, ... , data.number_of_measurements,
    // and e1 != e2
    // 
    // i = b + a*number_of_sensor_coils;
    
    // resize matrices:
    sensitivity_matrix.resize(2);
    for (unsigned int b=0; b<2; ++b)
    {
      sensitivity_matrix[b].reinit(data.number_of_combinations,
                                   data.n_voxels);
    }
    
    const unsigned int quad_order = 2*dof_handler.get_fe().degree + 1;
    const QGauss<dim>  quadrature_formula(quad_order);
    const unsigned int n_q_points = quadrature_formula.size();
    
    FEValues<dim> fe_values (mapping, dof_handler.get_fe(),
                             quadrature_formula,
                             update_values | update_quadrature_points | update_JxW_values);
    
    std::vector<FEValuesExtractors::Vector> vec;
    vec.reserve(2);
    vec.push_back(FEValuesExtractors::Vector (0));
    vec.push_back(FEValuesExtractors::Vector (dim));

    sensitivity_matrix[0] = 0.0;
    sensitivity_matrix[1] = 0.0;
    
    typename DH::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    
    for (; cell!=endc; ++cell)
    {
      // take note of which voxel we are in (via material_id)
      // If material_id is not in the recovery region, then skip.
      // The background region is hard-coded to be 0 in the setup region
      if (cell->material_id() > 0)
      {
        // Since 0 is the background region, take 1 from the material_id to get the voxel index
        // i.e. material_id 1 is voxel 0 & so on.
        //
        fe_values.reinit(cell);
        
        const unsigned int voxel_index = (unsigned int)cell->material_id() - 1;

        // within this cell, calculate solution values at all quadrature points:
        // Similarly to the voltage calculation (see voltage4), we store the real/imag
        // part in the first index.
        std::vector<std::vector<std::vector<Tensor<1,dim>>>> cell_solutions(2);
        for (unsigned int c=0; c<cell_solutions.size(); ++c)
        {
          cell_solutions[c].resize(data.number_of_excitations);
          for (unsigned int exciter=0; exciter < cell_solutions[c].size(); ++exciter)
          {
            cell_solutions[c][exciter].resize(n_q_points);
            fe_values[vec[c]].get_function_values(FEsolutions[exciter],
                                                  cell_solutions[c][exciter]);
          }
        }
        
        // NOTE: we ignore the (exciter==sensor) combination
        unsigned int i_index = 0;
        for (unsigned int exciter=0; exciter<data.number_of_excitations; ++exciter)
        {
          // Ignore case where e1=e2, so knock 1 off number of measurements.
          for (unsigned int sensor=0; sensor<data.number_of_measurements; ++sensor)
          {
            if (sensor != exciter)
            {
              double temp_re=0.0;
              double temp_im=0.0;
              for (unsigned int q=0; q<n_q_points; ++q)
              {
                temp_re += (
                  cell_solutions[0][exciter][q]*cell_solutions[0][sensor][q]
                  - cell_solutions[1][exciter][q]*cell_solutions[1][sensor][q]
                )*fe_values.JxW(q);
                
                temp_im += (
                  cell_solutions[0][exciter][q]*cell_solutions[1][sensor][q]
                  + cell_solutions[1][exciter][q]*cell_solutions[0][sensor][q]
                )*fe_values.JxW(q);
              }
              // We are solving the "A-based" formulation, where E = -i*omega*A,so we must
              // multiply each A by -i*omega to get the electric field.
              // Create a factor to multiply the sensitivity matrix by to get this
              // Note: this could go outside the loop involving voxels, but we may want to consider
              //       ways to use multiple frequencies in the future (not compatible in the current sensitivity though).
              // TODO !!!
              double sensitivity_factor = 1.0;//-data.omega*data.omega;
              sensitivity_matrix[0](i_index, voxel_index) += sensitivity_factor*temp_re;
              sensitivity_matrix[1](i_index, voxel_index) += sensitivity_factor*temp_im;
              ++i_index;              
            }
          }
        }
      }
    }
  }
  
  template<int dim, class DH>
  void InverseSolver_Voltages<dim, DH>::assemble_sensitivity_rhs(const std::vector<Vector<double>> &FEsolutions,
                                                                 const ForwardSolver::EddyCurrentData &fe_data,
                                                                 const DH &dof_handler,
                                                                 const Mapping<dim> &mapping)
  {
    // Construct the RHS of the sensitivity system. This is:simulated_functions
    // The voltage difference between the measured and simulated voltages.
    // Later on it will be used as:
    //
    // -1/2*(J^{H}*V + J^{T}*V^{*}).
    //
    // in the gauss-newton iteration, where J is the sensitivity matrix calculated by assemble_sensitivity_matrix()
    // and V is the voltage difference between the image measured voltages and the
    // current simulated voltages.
    //
    // ^{*} denotes complex conjugate and ^{H} denotes hermitian (conjugate transpose).
    //
    // Care must be taken to match the exciter/sensor coil combinations of the voltage
    // to that which was used in the assemble_sensitivity_matrix() routine.    
    
    sensitivity_rhs.resize(2);
    sensitivity_rhs[0].reinit(data.number_of_combinations);
    sensitivity_rhs[1].reinit(data.number_of_combinations);
    
    // Call the update data function:
    std::vector<Vector<double>> simulated_voltages;
    assemble_rhs_data(simulated_voltages,
                      FEsolutions,
                      fe_data,
                      dof_handler,
                      mapping);
    
    // Sens rhs is just the measured result minus the simulated result.
    // Must also account for the fact that we have A and B (E=-i*omega*sigma, B=curl(A), H = (1/mu)*B
    // Since the rhs data is = int_{\partial\Omega} E^{i}\cdot(H_{0}^{j} \times n) d\Omega,
    // then we need to multiply the resulting integral by -i*omega*(1/mu)
    // TODO: Make sure the above is taken care of (IF IT NEEDS TO BE ??).
    unsigned int i_index = 0;
    for (unsigned int exciter=0; exciter < data.number_of_excitations; ++exciter)
    {
      for (unsigned int sensor=0; sensor < data.number_of_measurements; ++sensor)
      {
        if (exciter != sensor)
        {
          for (unsigned int c=0; c<2; ++c)
          {
            sensitivity_rhs[c](i_index) = simulated_voltages[c](i_index) - measured_voltages[c](i_index);
          }
          ++i_index;
        }
      }
    }

  }
    
  template<int dim, class DH>
  void InverseSolver_Voltages<dim, DH>::assemble_regularisation(const DH &dof_handler)
  {
    // TODO: add regularisation matrix. For now, using identity.
    
  }
  
  template<int dim, class DH>
  void InverseSolver_Voltages<dim, DH>::gauss_newton_solve(const Vector<double> &last_solution,
                                                           Vector<double> &solution_out,
                                                           const unsigned int &iteration_count)
  {
    // TODO: Add output of the residual (voltage differences).
    // Could include the regularisation part of the residual as well (1/2*lambda^"norm(reg_matrix*dY)) etc.
    
    // Input is the last computed solution (deltaY, not Y), which is needed for regularisation
    // and the output is the computed solution.
    //
    // Gauss-Newton iterative inverse solver (DONE) with general regularisation (TODO)
    // Assumes we've already called:
    // - assemble_sensitivity_matrix
    // - assemble_sensitivity_rhs
    // - assemble_regularisation (TODO)
    //
    // Update term is:
    //
    // dY = (Re(J^{H}*J) + alpha^2*R^{T}*R)^{-1}*(0.5(J^{H}*X+J^{T}*X^{*}) + alpha^2*R^{T}*R*(SIGMA_N-SIGMA_0))
    //
    // where dY is the parameter update
    //       X is the sensitivity_rhs,
    //       SIGMA is the parameter distribution,
    //       R is the regularisation matrix,
    //       alpha is the regularisation parameter.
    //       J is the sensitivity matrix.
    // where 
    // Note for some matrix J = J_re + i*J_im:
    // say A = J^{H}*J = (J_re^{T} - i*J_im^{T})*(J_re + i*J_im)
    //                 = (J_re^{T}*J_re + J_im^{T}*J_im) + i*(J_im^{T}*J_re - J_re^{T}*J_im)
    //
    // i.e. A_re = J_re^{T}*J_re + J_im^{T}*J_im
    //      A_im = J_im^{T}*J_re - J_re^{T}*J_im
    //
    // However, here the matrix is Re(J^{H}*J), so we only need J_re^{T}*J_re + J_im^{T}*J_im.
    
    // increment iteration count: (can probably remove this safely)
    ++InverseProblemData::iteration_count;
    
    // For testing purpose (can read in and check in MATLAB)
    {
      // output the sens rhs
      std::stringstream rhs_filename_re;
      rhs_filename_re << "sens_rhs_re_step" << iteration_count << ".txt";
      std::stringstream rhs_filename_im;
      rhs_filename_im << "sens_rhs_im_step" << iteration_count << ".txt";
      
      std::ofstream rhs_file_re(rhs_filename_re.str());
      std::ofstream rhs_file_im(rhs_filename_im.str());
      rhs_file_re.precision(32);
      rhs_file_im.precision(32);
      for (unsigned int i=0; i<data.number_of_combinations; ++i)
      {
        rhs_file_re << sensitivity_rhs[0](i) << std::endl;
        rhs_file_im << sensitivity_rhs[1](i) << std::endl;
      }
      rhs_file_re.close();
      rhs_file_im.close();
      // output the sens matrix:
      std::stringstream matrix_filename_re;
      matrix_filename_re << "sens_matrix_re" << ".txt";
      std::stringstream matrix_filename_im;
      matrix_filename_im << "sens_matrix_im" << ".txt";
      
      std::ofstream matrix_file_re(matrix_filename_re.str());
      std::ofstream matrix_file_im(matrix_filename_im.str());
      
      matrix_file_re.precision(32);
      matrix_file_im.precision(32);
      for (unsigned int i=0; i<data.number_of_combinations; ++i)
      {
        
        for (unsigned int j=0; j<data.n_voxels; ++j)
        {
          matrix_file_re << sensitivity_matrix[0](i,j) << " ";
          matrix_file_im << sensitivity_matrix[1](i,j) << " ";
        }
        matrix_file_re << std::endl;
        matrix_file_im << std::endl;
      }
      matrix_file_re.close();
      matrix_file_im.close();
    }
    
    // Storage for the part which needs inverting
    FullMatrix<double> GN_matrix(data.n_voxels, data.n_voxels);
    // Options for computing the inverse are:
    // Use GMRES as the real system arising from the complex matrix is non-symmetric.
    // Alternative is to use a direct solver via FullMatrix::invert() or gauss_jordan()
    // TODO: Use gauss jordan for now, but should experiment with performance.
    
    // Construct the GN matrix:
    sensitivity_matrix[0].Tmmult(GN_matrix,
                                 sensitivity_matrix[0]); // copies over GN_matrix.
    sensitivity_matrix[1].Tmmult(GN_matrix,
                                 sensitivity_matrix[1],
                                 true); // adds result to GN_matrix.

    // For now we're using the identity for the regularisation matrix
    // so just add the regularisation parameter to the diagonal entries.
    // TODO: update to include a general regularisation matrix. For now just use identity.
    for (unsigned int i=0; i<data.n_voxels; ++i)
    {
      GN_matrix(i,i) += data.regularisation_parameter*data.regularisation_parameter;
    }
    
    // calculate the inverse: GN_matrix will now hold it.
    GN_matrix.gauss_jordan(); 
    
    // Now form the RHS, which the inverse is to be applied to:
    // TODO: Update for a general regularisation matrix and with
    //       a penalty term for large changes from the original (or last)
    //       solution.
    //       
    Vector<double> GN_rhs(data.n_voxels);
    sensitivity_matrix[0].Tvmult(GN_rhs,
                                 sensitivity_rhs[0]);
    sensitivity_matrix[0].Tvmult(GN_rhs,
                                 sensitivity_rhs[1],
                                 true);
    
    // Include the regularisation term which punishes large changes:
    for (unsigned int i=0; i<GN_rhs.size(); ++i)
    {
      GN_rhs(i) += data.regularisation_parameter*data.regularisation_parameter*last_solution(i);
    }
    
    // resize solution:
    solution_out.reinit(data.n_voxels);
    // Multiply RHS by the inverted matrix:    
    GN_matrix.vmult(solution_out, GN_rhs);
  }
  
  template<int dim, class DH>
  double InverseSolver_Voltages<dim, DH>::return_functional_norm() const
  {
    // Returns the norm of the current functional.
    // The current functional being the last computed sensitivity RHS.
    const double real_part = sensitivity_rhs[0].norm_sqr();
    const double imag_part = sensitivity_rhs[1].norm_sqr();
    
    return sqrt(real_part+imag_part);
  }
 
  template class InverseSolver_Voltages<3, DoFHandler<3>>;
}