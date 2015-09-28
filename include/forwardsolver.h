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

#ifndef FORWARDSOLVER_H
#define FORWARDSOLVER_H

// deal.II includes:
#include <deal.II/base/quadrature.h>
#include <deal.II/base/point.h>
#include <deal.II/base/smartpointer.h>
#include <deal.II/base/tensor.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_nedelec.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/fe/mapping.h>

#include <deal.II/hp/dof_handler.h>

#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/compressed_sparsity_pattern.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/precondition_block.h>
#include <deal.II/lac/sparse_decomposition.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/vector_tools.h>

// std includes:

// My includes:
#include <all_data.h>
//#include <backgroundfield.h>
#include <curlfunction.h>
#include <mydofrenumbering.h>
#include <mypreconditioner.h>

using namespace dealii;

namespace ForwardSolver
{
  using namespace dealii;
  // TODO:
  // At the moment this is a general vector-wave equation solver. ?? It may be a good idea to rename it ???
  //
  // It can be made to be an eddy current solver with the correct choice of material parameters.
  //
  // Care must be taken with the choice of material parameters, boundary conditions and RHS (currently enforced to be zero)
  // in that they must all be chosen to be consistent with each other.
  //
  // i.e. if we use an A-based formulation:
  // curl(1/mu_r*(curl(A)) + kappa*mu_0*A = mu_0*J_s
  //
  // then the A and curl(A) for setting the BCs must be chosen to reflect this
  //
  // similarly, if solving for E:
  // curl(1/mu_r*(curl(E)) + kappa*mu_0*E = i*omega*mu_0*J_s
  //
  // then E and curl(E) must be set appropriately.
  //
  //
  // Also, in the case of neumann BCs, they must be applied using the EddyCurrentFunction<dim>
  // and then the contribution is n x mu_{r} x curl(F) = ... 
  // where curl(F) is defined through the derived EddyCurrentFunction<dim> passed to the solver.
  // note F is either A or E here.
  //
  // Finally, care in the choice of regularisation must be taken. Where kappa=0 then we should choose a small regularisation
  // to stop the system becoming singular, so that kappa is set to be kappa_re = reg*mu0, kappa_im = 0.
  // 
  
  
  // Internal data struct
  struct EddyCurrentData
  {
    // Data for use with the EddyCurrent class
    
    // Material Parameters:
    double param_omega = 1.0; // angular frequency, rads/sec.
    double param_regularisation = 1.0e-3;
    
    // Material parameters for non-conducting region:
    double param_epsilon_background = 0.0;
    double param_sigma_background = 0.0;
    double param_mur_background = 1.0;
    
    
    // Vectors holding equation parameters
    std::vector<double> param_mur;
    std::vector<double> param_sigma;
    std::vector<double> param_epsilon;
    
    // Geometry info:
    // Need to know which materials flag a conducting region
    // Can be stored via a number of materials and a flag.
    // where conducting_material.size = n_materials.
    unsigned int n_materials;
    std::vector<bool> conducting_material;
    
    // Factor for the RHS of the equation.
    double rhs_factor = 1.0;
    
    // Use neumann conditions
    bool neumann_flag = false;
    
    // Solver options
    bool direct_flag = false; // true = direct solver, false = iterative
    bool constrain_gradients = true; // flag to constrain gradient-based DoFs in the non-conducting region
    bool right_preconditioning = true; // true = right preconditioning, false = left preconditioning
    double solver_tolerance = 1e-7; // tolerance on the iterative GMRES solver
    
  };
  template <int dim, class DH>
  class EddyCurrent
  {
    // This class is for the solution of the eddy current approximation of the time-harmonic maxwell equations.
    //  
    // Input:
    // - a dof_handler which is attached to an finite element and triangulation with a set of material IDs and BC IDs.
    // - A list of material parameter values corresponding to these IDs.
    //  
    // Functions:
    // - compute constraints on the linear system according to the BCs and an input function for the BCs.
    // - compute the stiffness matrix associated with the PDE
    // - compute the RHS vector associated with the PDE
    // - solve the linear system.
    // 
    // Output:
    // - a solution vector for the coeffs of the FE solution of the PDE
  public:
    EddyCurrent (const EddyCurrentData &data_in,
                 DH &dof_handler,
                 const FiniteElement<dim> &fe);
    EddyCurrent (const EddyCurrentData &data_in,
                 const Mapping<dim,dim> &mapping_in,
                 DH &dof_handler,
                 const FiniteElement<dim> &fe);
    
    ~EddyCurrent ();
    
    // Updates the internal data.
    void update_eddy_data (const EddyCurrentData &data_in);
    
    void assemble_matrices (const DH &dof_handler);

    void assemble_rhs (const DH &dof_handler,
                       const EddyCurrentFunction<dim> &boundary_function);
    
    void initialise_solver ();
    void solve (Vector<double> &output_solution,
                unsigned int &n_iterations);
    
    
  private:
    void constructor_setup(DH &dof_handler);
    
    void compute_constraints (const DH &dof_handler,
                              const EddyCurrentFunction<dim> &boundary_function);
    
    // Data object
    EddyCurrentData data;
    
    // FE storage:
    const SmartPointer< const FiniteElement<dim> > fe; // TODO: check if ok, or should be protected??
    unsigned int p_order;
    unsigned int quad_order;
    
    // Mapping storage:
    SmartPointer< const Mapping<dim> > mapping;
    
    // Coefficients arising from materials parameters:
    // Kappa = Kappa_re + i*Kappa_im = -omega^2*epr + i*omega*sigma
    std::vector<std::vector<double>> param_kappa;
    
    // Block sparse storage:
    BlockSparsityPattern sparsity_pattern;
    BlockSparseMatrix<double> system_matrix;
    BlockVector<double> solution;
    BlockVector<double> system_rhs;
    
    // Dof ordering storage:
    unsigned int n_lowest_order_dofs;
    unsigned int n_higher_order_dofs;
    
    unsigned int n_gradient_real;
    unsigned int n_nongradient_real;
    unsigned int n_gradient_imag;
    unsigned int n_nongradient_imag;
    
    unsigned int n_higher_order_gradient_dofs;
    unsigned int n_higher_order_non_gradient_dofs;
    
    // Make-up of the re-ordering global dofs:
    // These are used to avoid confusion when accessing
    // different blocks in vector/array of length dof_handler.n_dofs().
    // each defines the end point of a particular block (i.e. where the next one begins).
    unsigned int end_lowest_order_dofs;
    unsigned int end_gradient_dofs_real;
    unsigned int end_nongradient_dofs_real;
    unsigned int end_gradient_dofs_imag;
    unsigned int end_nongradient_dofs_imag;
    
    // Preconditioner:
    BlockSparsityPattern preconditioner_sparsity_pattern;
    BlockSparseMatrix<double> system_preconditioner;
    // TODO: would this be better as SmartPointer<Preconditioner::EddyCurrentPreconditionerBase> preconditioner; ??
    Preconditioner::EddyCurrentPreconditionerBase* preconditioner;
    
    // Constraints:
    ConstraintMatrix constraints;
    
    // Linear Algebra Storage:
    bool initialised = false;
    SparseDirectUMFPACK direct_solve;
    
  // protected: // removed, used to contain smart pointer for fe.
  };  
}
#endif
