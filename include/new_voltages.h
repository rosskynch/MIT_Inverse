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

#ifndef NEWVOLTAGES_H
#define NEWVOLTAGES_H

#include <deal.II/base/quadrature.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/fe/fe_nedelec.h>
#include <deal.II/fe/fe_q.h>

#include <all_data.h>
#include <backgroundfield.h>
#include <curlfunction.h>
#include <forwardsolver.h>
#include <inputtools.h>
#include <inversesolver_voltages.h>
#include <mydofrenumbering.h>
#include <mypreconditioner.h>
#include <myvectortools.h>
#include <outputtools.h>

#include <myfe_nedelec.h>

using namespace dealii;

// New namespace containing different methods for voltage calculation.

// Forward declare the InverseSolver_Voltages structure:
namespace InverseSolver_Voltages
{
  template<int dim>
  struct InternalData;
}

namespace NewVoltages
{
  /* Not implemented for now.
  template<int dim>
  class Voltage1
  {
    // Method 1 dv =- i omega * msensor . B^s
    // NOTE: Can't use the analytical solution from the sphere dipole here. (No curl soln at present, could add later).
  public:
    Voltage1;
    // Methods using a function solution:
    void calculateVoltage(const InverseSolver_Voltages::InternalData<dim> &inverse_data,
                          const ForwardSolver::EddyCurrentData &fe_data,
                          const std::vector<backgroundField::conductingSphereDipole<dim>> functions,
                          const DoFHandler<dim> &dof_handler,
                          std::vector<std::vector<std::vector<double>>> &calculated_voltages) const;

    // Methods using the FE solution:
    void calculateVoltage(const InverseSolver_Voltages::InternalData<dim> &inverse_data,
                          const ForwardSolver::EddyCurrentData &fe_data,
                          const std::vector<Vector<double>> &FEsolutions,
                          const DoFHandler<dim> &dof_handler,
                          std::vector<std::vector<std::vector<double>>> &calculated_voltages) const;
                          
  private:
  };*/
  
  template<int dim>
  class Voltage2
  {
    // Method 2 dv = oint E^s. dr = -i omega \oint A^s . dr.
  public:
    Voltage2(const unsigned int &quad_order = 20);
    
    // TODO: Join the mapping/non-mapping versions by switching the mapping to be the final argument
    //       and then can define it in the header as const Mapping<dim> &mapping = StaticMappingQ1<dim>::mapping
    //       No time to do it since it'll require changes to all voltage test codes.
    

    // Methods using a function solution:
    void calculateVoltages(const InverseSolver_Voltages::InternalData<dim> &inverse_data,
                           const ForwardSolver::EddyCurrentData &fe_data,
                           const std::vector<backgroundField::conductingSphereDipole<dim>> functions,
                           const DoFHandler<dim> &dof_handler,
                           const Mapping<dim> &mapping,
                           std::vector<std::vector<std::vector<double>>> &calculated_voltages) const;
    // no mapping
    void calculateVoltages(const InverseSolver_Voltages::InternalData<dim> &inverse_data,
                           const ForwardSolver::EddyCurrentData &fe_data,
                           const std::vector<backgroundField::conductingSphereDipole<dim>> functions,
                           const DoFHandler<dim> &dof_handler,
                           std::vector<std::vector<std::vector<double>>> &calculated_voltages) const;

    // Methods using the FE solution:
    void calculateVoltages(const InverseSolver_Voltages::InternalData<dim> &inverse_data,
                           const ForwardSolver::EddyCurrentData &fe_data,
                           const std::vector<Vector<double>> &FEsolutions,
                           const DoFHandler<dim> &dof_handler,
                           const Mapping<dim> &mapping,
                           std::vector<std::vector<std::vector<double>>> &calculated_voltages) const;
    // no mapping
    void calculateVoltages(const InverseSolver_Voltages::InternalData<dim> &inverse_data,
                           const ForwardSolver::EddyCurrentData &fe_data,
                           const std::vector<Vector<double>> &FEsolutions,
                           const DoFHandler<dim> &dof_handler,
                           std::vector<std::vector<std::vector<double>>> &calculated_voltages) const;
                          
  private:
    std::complex<double> returnVoltage(const double &sensor_coil_radius,
                                       const Point<dim> &sensor_coil_centre,
                                       const Tensor<1,dim> &sensor_coil_direction,
                                       const backgroundField::conductingSphereDipole<dim> &solution_function) const;
    // no mapping
    std::complex<double> returnVoltage(const double &sensor_coil_radius,
                                       const Point<dim> &sensor_coil_centre,
                                       const Tensor<1,dim> &sensor_coil_direction,
                                       const Vector<double> &FEsolution,
                                       const DoFHandler<dim> &dof_handler) const;
    // with mapping
    std::complex<double> returnVoltage(const double &sensor_coil_radius,
                                       const Point<dim> &sensor_coil_centre,
                                       const Tensor<1,dim> &sensor_coil_direction,
                                       const Vector<double> &FEsolution,
                                       const Mapping<dim> &mapping,
                                       const DoFHandler<dim> &dof_handler) const;
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
  
  /* Not implemented for now.
  template<int dim>
  class Voltage3
  {
    // Method 3 dv = - i omega oint A_I . dr
    // A_I (x) = mu/(4pi) int_Omega_c e^ik|x-x'| sigma E(x')/|x-x'| dx'
    //
    // B_I = curl A_I (x) = mu/(4 *pi ) int_Omega_c sigma e^ik |x-x'| (ik/|x-x'|^2 - 1/|x-x'|^3)(x-x') x E(x') dx'
    // as factor required for voltage to agree.
  public:
    Voltage3
    // Methods using a function solution:
    void calculateVoltage(const InverseSolver_Voltages::InternalData<dim> &inverse_data,
                          const ForwardSolver::EddyCurrentData &fe_data,
                          const std::vector<backgroundField::conductingSphereDipole<dim>> functions,
                          const DoFHandler<dim> &dof_handler,
                          std::vector<std::vector<std::vector<double>>> &calculated_voltages) const;

    // Methods using the FE solution:
    void calculateVoltage(const InverseSolver_Voltages::InternalData<dim> &inverse_data,
                          const ForwardSolver::EddyCurrentData &fe_data,
                          const std::vector<Vector<double>> &FEsolutions,
                          const DoFHandler<dim> &dof_handler,
                          std::vector<std::vector<std::vector<double>>> &calculated_voltages) const;
                          
  private:
  };*/
  
  template<int dim>
  class Voltage4
  {
    // Method 4 dv = - i omega msensor.B_I
    // A_I (x) = mu/(4pi) int_Omega_c sigma E(x')/|x-x'| dx'
    // perhaps should be 
    // A_I (x) = mu/(2) int_Omega_c sigma E(x')/|x-x'| dx'
    // as factor required for voltage to agree.
  public:
    Voltage4(const unsigned int &quad_order = 5);
    // Methods using a function solution:
    void calculateVoltages(const InverseSolver_Voltages::InternalData<dim> &inverse_data,
                           const ForwardSolver::EddyCurrentData &fe_data,
                           const std::vector<backgroundField::conductingSphereDipole<dim>> functions,
                           const DoFHandler<dim> &dof_handler,
                           const Mapping<dim> &mapping,
                           std::vector<std::vector<std::vector<double>>> &calculated_voltages) const;
    // no mapping
    void calculateVoltages(const InverseSolver_Voltages::InternalData<dim> &inverse_data,
                           const ForwardSolver::EddyCurrentData &fe_data,
                           const std::vector<backgroundField::conductingSphereDipole<dim>> functions,
                           const DoFHandler<dim> &dof_handler,
                           std::vector<std::vector<std::vector<double>>> &calculated_voltages) const;

    // Methods using the FE solution:
    void calculateVoltages(const InverseSolver_Voltages::InternalData<dim> &inverse_data,
                           const ForwardSolver::EddyCurrentData &fe_data,
                           const std::vector<Vector<double>> &FEsolutions,
                           const DoFHandler<dim> &dof_handler,
                           const Mapping<dim> &mapping,
                           std::vector<std::vector<std::vector<double>>> &calculated_voltages) const;
                           
    // no mapping
    void calculateVoltages(const InverseSolver_Voltages::InternalData<dim> &inverse_data,
                           const ForwardSolver::EddyCurrentData &fe_data,
                           const std::vector<Vector<double>> &FEsolutions,
                           const DoFHandler<dim> &dof_handler,
                           std::vector<std::vector<std::vector<double>>> &calculated_voltages) const;
                          
  private:
    const unsigned int quad_order;
  };

}
#endif