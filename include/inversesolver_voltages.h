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

#ifndef INVERSESOLVER_VOLTAGES_H
#define INVERSESOLVER_VOLTAGES_H

// deal.II includes:
#include <deal.II/base/quadrature.h>
#include <deal.II/base/point.h>
#include <deal.II/base/smartpointer.h>
#include <deal.II/base/tensor.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_nedelec.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/fe/mapping.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/hp/dof_handler.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>

// std includes:
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

// My includes:
#include <all_data.h>
#include <backgroundfield.h>
#include <curlfunction.h>
#include <forwardsolver.h>
#include <new_voltages.h>

using namespace dealii;

// For now we'll develop in a separate namespace to the original voltage-based inverse solver.
namespace InverseSolver_Voltages
{
  
  // Namespace for the inverse solver.
  // Will need to pass a vector of solution vectors
  // and a dof_handler associated with the FE used to
  // compute these solutions.
  // 
  // Note that the solutions will correspond to different
  // BCs but the underlying elements, grid and material
  // parameter distribution will be the same for all of them.
  
  template<int dim>
  struct InternalData {
    types::material_id recovery_region_id;
    types::material_id background_region_id=0;
    
    // Material/Problem Data:
    double omega;
    Vector<double> current_sigma;
    Vector<double> initial_sigma;
    
    // Inverse problem options:
    unsigned int max_GN_steps = 1;
    double GN_update_parameter = 1.;
    double regularisation_parameter = 1e-10;
    
    unsigned int quad_order = 8;
    
    unsigned int n_voxels;
    
    unsigned int number_of_excitations;
    unsigned int number_of_measurements;
    
    std::vector<double> measurement_coil_radius;
    std::vector<Point<dim>> measurement_positions;
    std::vector<Tensor<1,dim>> measurement_directions;
    
    unsigned int number_of_combinations;

//     std::vector<EddyCurrentFunction<dim>> boundary_functions;
    // temp fix, above causes issues:
    // If using functions for measured data:
    std::vector<backgroundField::conductingSphereDipole<dim>> measured_functions;
    
    bool use_solution_functions;
  };
  
  template <int dim, class DH>
  class InverseSolver_Voltages
  {
  public:
    InverseSolver_Voltages(const InternalData<dim> &data_in,
                           const DH &dof_handler,
                           const Mapping<dim> &mapping = StaticMappingQ1<dim>::mapping);
    
    InverseSolver_Voltages(const InternalData<dim> &data_in,
                           const std::vector<Vector<double>> &FEsolutions,
                           const ForwardSolver::EddyCurrentData &fe_data,
                           const DH &dof_handler,
                           const Mapping<dim> &mapping);
    
    // Replace the stored InternalData struct:
    void updateInternalData(const InternalData<dim> &data_in);
    
    void assemble_sensitivity_matrix(const std::vector<Vector<double>> &FEsolutions,
                                     const DH &dof_handler,
                                     const Mapping<dim> &mapping = StaticMappingQ1<dim>::mapping);

    void assemble_sensitivity_rhs(const std::vector<Vector<double>> &FEsolutions,
                                  const ForwardSolver::EddyCurrentData &fe_data,
                                  const DH &dof_handler,
                                  const Mapping<dim> &mapping = StaticMappingQ1<dim>::mapping);
    

// TODO: For the analytical version (i.e. via a function argument, we should
// replace std::vector<backgroundField::conductingSphereDipole<dim>>
// with std::vector<EddyCurrentFunction<dim>>

    // Using the exact solution provided by an EddyCurrentFunction<dim> object.
    void assemble_sensitivity_matrix(const std::vector<backgroundField::conductingSphereDipole<dim>> &simulated_functions,
                                     const DH &dof_handler,
                                     const Mapping<dim> &mapping = StaticMappingQ1<dim>::mapping);
    
    void assemble_sensitivity_rhs(const std::vector<backgroundField::conductingSphereDipole<dim>> &simulated_functions,
                                  const DH &dof_handler,
                                  const Mapping<dim> &mapping = StaticMappingQ1<dim>::mapping);
    
    void assemble_regularisation(const DH &dof_handler);
    
    void gauss_newton_solve(const Vector<double> &last_solution,
                            Vector<double> &solution_out,
                            const unsigned int &iteration_count = 0);
    
    double return_functional_norm() const;
    
  private:
    InternalData<dim> data;

    void assemble_rhs_data(std::vector<Vector<double>> &rhs_data,
                           const std::vector<backgroundField::conductingSphereDipole<dim>> &solution_functions,
                           const DH &dof_handler,
                           const Mapping<dim> &mapping = StaticMappingQ1<dim>::mapping);

    void assemble_rhs_data(std::vector<Vector<double>> &rhs_data,
                           const std::vector<Vector<double>> &FEsolutions,
                           const ForwardSolver::EddyCurrentData &fe_data,
                           const DH &dof_handler,
                           const Mapping<dim> &mapping = StaticMappingQ1<dim>::mapping);
   
    std::vector<FullMatrix<double>> regularisation_matrix;
    
    std::vector<FullMatrix<double>> sensitivity_matrix;
    
    std::vector<Vector<double>> sensitivity_rhs;
    std::vector<Vector<double>> measured_voltages;
  };
}
#endif