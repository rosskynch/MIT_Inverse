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

#include <deal.II/base/quadrature.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/fe/fe_nedelec.h>

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
#include <new_voltages.h>

#include <myfe_nedelec.h>

using namespace dealii;

namespace InverseSolver
{
  
  template <int dim>
  class InverseSolver
  {
  public:
    InverseSolver (const unsigned int poly_order,
                       const unsigned int mapping_order = 2);
    ~InverseSolver ();
    void run(std::string &output_filename,
             const unsigned int href_in = 0);
  private:
    Triangulation<dim> tria;
    FESystem<dim> fe;
    DoFHandler<dim> dof_handler;

    const unsigned int poly_order;
    const unsigned int mapping_order=2;
    
    void setup_fe_data();
    void setup_inverse_data();
    void create_mesh();
    void update_material_parameters(const Vector<double> &delta_sigma);
    
    // Solver data:
    ForwardSolver::EddyCurrentData eddy_data_exact;
    ForwardSolver::EddyCurrentData eddy_data_approx;
    // Function data:
    std::vector<backgroundField::conductingSphereDipoleData<dim>> dipole_data;
    
    // Data for the problem:
    // Geometry data:
    const double sphere_radius = 0.01;
    const double boundary_radius = 0.07;
    
    //mesh option:
    unsigned int href;
    const unsigned int free_space_id = 0;
    const unsigned int sphere_id = 1;
    const unsigned int sphere_manifold_id = 100;
    
    const double coil_radius = 0.01;//0.025;
    const double sigma = 1.4;
    const double mur = 1.0;
    const double mu = mur*EquationData::constant_mu0;
    const double omega = 2.0*numbers::PI*1.0e6;
    const double h0 = 0.08; // radial distance of coil centre from sphere centre.
    const double theta0 = atan(coil_radius/h0); // angle between radial line from centre of sphere to centre of coil and radial line to the edge of the coil.
    const double r0 = coil_radius/sin(theta0); // distance from sphere centre to outer edge of coil
    const double I0 = 1.0;///EquationData::constant_mu0; // Current.
    
    // Inverse data:
    const double GN_regularisation_parameter = 1e-10;
    const unsigned int max_GN_steps = 10;
    const double GN_update_parameter = 1.;
    unsigned int n_voxels;
    InverseSolver_Voltages::InternalData<dim> inverse_data;
    const unsigned int coils_per_axis = 5; // avoid numbers which put coils on the other cartesian axes.
    const unsigned int n_rotation_axes = 3;
    const unsigned int number_of_excitations = coils_per_axis*(n_rotation_axes);
    const unsigned int number_of_measurements = coils_per_axis*(n_rotation_axes);
    const double sigma_initial = 1.0;
    const double mur_initial = mur; // not used at the moment.
    
    // constraints set on the update to sigma.
    const double sigma_max = 2.0*sigma;
    const double sigma_min = 0.5*sigma_initial;
  };
  
  template <int dim>
  InverseSolver<dim>::InverseSolver(const unsigned int poly_order,
                                            const unsigned int mapping_order)
  :
  fe (MyFE_Nedelec<dim>(poly_order), 2),
  dof_handler (tria),
  poly_order(poly_order),
  mapping_order(mapping_order)
  {
  }
  
  template <int dim>
  InverseSolver<dim>::~InverseSolver ()
  {
    dof_handler.clear ();  
  }
  
  template <int dim>
  void InverseSolver<dim>::setup_fe_data()
  {
    dipole_data.resize(number_of_excitations);
    
    // Have set this up to try to avoid putting coils in the same location

    // Rotation about x:
    const double angle_inc = 2.0*numbers::PI/(coils_per_axis);
    for (unsigned int i=0; i<coils_per_axis; ++i)
    {
      const double angle = i*angle_inc;
      const Tensor<1,dim> coil_direction({0.0, sin(angle), cos(angle)});
      dipole_data[i].coil_direction = coil_direction;
      // Populate the data struct for the analytical solution 
      // for a conducting sphere in a dipole field.
      dipole_data[i].sphere_radius = sphere_radius;
      dipole_data[i].coil_radius = coil_radius;
      dipole_data[i].sigma = sigma;
      dipole_data[i].mu = mu;
      dipole_data[i].theta0 = theta0;
      dipole_data[i].h0 = h0;
      dipole_data[i].r0 = r0;
      dipole_data[i].I0 = I0;      
      dipole_data[i].omega = omega;
      
    }
    // Rotation about y:
    const unsigned int y_offset = coils_per_axis;
    for (unsigned int i=0; i<coils_per_axis; ++i)
    {
      const double angle = i*angle_inc;
      const Tensor<1,dim> coil_direction({cos(angle), 0.0, sin(angle)});
      dipole_data[y_offset+i].coil_direction = coil_direction;
      // Populate the data struct for the analytical solution 
      // for a conducting sphere in a dipole field.
      dipole_data[y_offset+i].sphere_radius = sphere_radius;
      dipole_data[y_offset+i].coil_radius = coil_radius;
      dipole_data[y_offset+i].sigma = sigma;
      dipole_data[y_offset+i].mu = mu;
      dipole_data[y_offset+i].theta0 = theta0;
      dipole_data[y_offset+i].h0 = h0;
      dipole_data[y_offset+i].r0 = r0;
      dipole_data[y_offset+i].I0 = I0;      
      dipole_data[y_offset+i].omega = omega;
    }
    // Rotation about z:
    const unsigned int z_offset = 2*coils_per_axis;
    for (unsigned int i=0; i<coils_per_axis; ++i)
    {
      const double angle = i*angle_inc;
      const Tensor<1,dim> coil_direction({sin(angle), cos(angle), 0.0});
      dipole_data[z_offset+i].coil_direction = coil_direction;
      // Populate the data struct for the analytical solution 
      // for a conducting sphere in a dipole field.
      dipole_data[z_offset+i].sphere_radius = sphere_radius;
      dipole_data[z_offset+i].coil_radius = coil_radius;
      dipole_data[z_offset+i].sigma = sigma;
      dipole_data[z_offset+i].mu = mu;
      dipole_data[z_offset+i].theta0 = theta0;
      dipole_data[z_offset+i].h0 = h0;
      dipole_data[z_offset+i].r0 = r0;
      dipole_data[z_offset+i].I0 = I0;      
      dipole_data[z_offset+i].omega = omega;
    }
    
    // Populate the data struct for the forward solver
    // Note: not using an input file for this.
    eddy_data_exact.param_mur_background = 1.0;
    eddy_data_exact.param_sigma_background = 0.0;
    eddy_data_exact.param_epsilon_background = 0.0;
    
    eddy_data_exact.param_regularisation = 1e-6;
    
    // material data:
    // Note first material is assumed to be free space.
    eddy_data_exact.n_materials = n_voxels+1;
    eddy_data_exact.conducting_material.resize(eddy_data_exact.n_materials);
    eddy_data_exact.conducting_material[0] = false;

    eddy_data_exact.param_mur.resize(eddy_data_exact.n_materials);
    eddy_data_exact.param_mur[0] = mur;    
    
    eddy_data_exact.param_sigma.resize(eddy_data_exact.n_materials);
    eddy_data_exact.param_sigma[0] = 0.0;    
    
    eddy_data_exact.param_epsilon.resize(eddy_data_exact.n_materials);
    eddy_data_exact.param_epsilon[0] = 0.0;
    
    // Handle all active voxels together, 0 is free space.
    for (unsigned int v=1;v<n_voxels+1; ++v)
    {
      eddy_data_exact.conducting_material[v] = true;
      eddy_data_exact.param_mur[v] = mur;
      eddy_data_exact.param_sigma[v] = sigma;
      eddy_data_exact.param_epsilon[v] = 0.0;
    }
    
    eddy_data_exact.param_omega = omega;
    
    eddy_data_exact.rhs_factor = 0.0; // No source term for this problem.
    
    eddy_data_exact.neumann_flag = false; // Can only perform dirichlet for this one.
    
    // solver data:
    eddy_data_exact.direct_flag = false;
    eddy_data_exact.constrain_gradients = true;
    eddy_data_exact.right_preconditioning = true;
    eddy_data_exact.solver_tolerance = 1.0e-7;
    
    // Extra copy to shared data to ensure compatibility with output
    // TODO: update output tools to handle this.
    EquationData::param_mur.reinit(eddy_data_exact.n_materials);
    EquationData::param_sigma.reinit(eddy_data_exact.n_materials);
    EquationData::param_epsilon.reinit(eddy_data_exact.n_materials);
    for (unsigned int i=0; i<eddy_data_exact.n_materials; ++i)
    {
      EquationData::param_mur(i) = eddy_data_exact.param_mur[i];
      EquationData::param_sigma(i) = eddy_data_exact.param_sigma[i];
      EquationData::param_epsilon(i) = eddy_data_exact.param_epsilon[i];
    }
    // Now set up approx eddy current data (used for the inverse solver updates)
    // Just copy over the same parts and set the approx sigma.
    eddy_data_approx = eddy_data_exact;
    eddy_data_approx.param_sigma[0] = 0.0;
    for (unsigned int v=1;v<n_voxels+1; ++v)
    {
      eddy_data_approx.param_sigma[v] = sigma_initial;
    }
  }

  template <int dim>
  void InverseSolver<dim>::setup_inverse_data()
  {
    const unsigned int quad_order = 8;

    // n_voxels should'be been set in create_mesh
    inverse_data.n_voxels = n_voxels;
    
    inverse_data.number_of_excitations = number_of_excitations;
    inverse_data.number_of_measurements = number_of_measurements;
    
    // For now, we're matching exciter/sensor locations, so we ignore the same location.
    inverse_data.number_of_combinations = inverse_data.number_of_excitations*(inverse_data.number_of_measurements-1);
    
    inverse_data.use_solution_functions = false;
    
    inverse_data.quad_order = quad_order;
    
//     inverse_data.measured_functions.resize(number_of_measurements);
    inverse_data.measurement_coil_radius.resize(number_of_measurements);
    inverse_data.measurement_positions.resize(number_of_measurements);
    inverse_data.measurement_directions.resize(number_of_measurements);

    for (unsigned int i=0; i<number_of_measurements; ++i)
    {
      // direction of coil from centre of sphere:
      // Simply take this info from the forward solver's boundary function data:
      inverse_data.measurement_directions[i] = dipole_data[i].coil_direction;
      inverse_data.measurement_positions[i] = Point<dim> (h0*dipole_data[i].coil_direction);
      inverse_data.measurement_coil_radius[i] = coil_radius;
    }
    inverse_data.omega = omega;
    
    // MAY NEED TO CHANGE DEPENDING ON MATERIAL PARAMETERS.
    inverse_data.max_GN_steps = max_GN_steps;
    inverse_data.GN_update_parameter = GN_update_parameter;
    inverse_data.regularisation_parameter = GN_regularisation_parameter;
  }
  
  template <int dim>
  void InverseSolver<dim>::create_mesh()
  {
    typename Triangulation<dim>::cell_iterator cell,endc;
    
    static const SphericalManifold<dim> sph_boundary;
    const double ball_radius = sphere_radius;
    // Make the central shell account for 1/2 of the shell part of the mesh:
    // This will be part of the recovery area, leaving the outer shell to be background.
    const double middle_radius = 0.5*(boundary_radius-sphere_radius) + sphere_radius;
    const double outer_radius = boundary_radius;
    {
      Triangulation<dim> whole_ball;
      GridGenerator::hyper_ball(whole_ball,
                                Point<dim> (0.0,0.0,0.0),
                                ball_radius);
      cell = whole_ball.begin();
      endc = whole_ball.end();
      // Inner ball is material id 1.
      for (;cell!=endc; ++cell)
      {
        cell->set_material_id(sphere_id);
      }
      
      Triangulation<dim> middle_shell;
      GridGenerator::hyper_shell(middle_shell,
                                 Point<dim> (0.0,0.0,0.0),
                                 ball_radius,
                                 //middle_radius,
                                 outer_radius,
                                 6);
      cell = middle_shell.begin();
      endc = middle_shell.end();
      // Middle shell is material id is 0 (non-conducting).
      for (;cell!=endc; ++cell)
      {
        cell->set_material_id(free_space_id);
      }
      GridGenerator::merge_triangulations(whole_ball,
                                          middle_shell,
                                          tria);

      // Remove outer shell for now - no need.
      /*
      Triangulation<dim> middle_ball;
      GridGenerator::merge_triangulations(whole_ball,
                                          middle_shell,
                                          middle_ball);
      Triangulation<dim> outer_shell;
      GridGenerator::hyper_shell(outer_shell,
                                 Point<dim> (0.0,0.0,0.0),
                                 middle_radius,
                                 outer_radius,
                                 6);
      cell = outer_shell.begin();
      endc = outer_shell.end();
      // Outer shell is boundary id 0
      // NOTE: boundary id 0 is ignored in the inverse problem - it is 
      //       background by default and will not change.
      for (;cell!=endc; ++cell)
      {
        cell->set_material_id(free_space_id);
      }
      GridGenerator::merge_triangulations(middle_ball,
                                          outer_shell,
                                          tria);*/
    }
    // Now mark the central ball and the outer shell's boundaries as spherical manifolds:
    cell = tria.begin ();
    endc = tria.end ();
    for (; cell!=endc; ++cell)
    {
      cell->set_all_manifold_ids(numbers::invalid_manifold_id);
    }
    tria.set_all_manifold_ids_on_boundary(sphere_manifold_id);
    cell = tria.begin();
    endc = tria.end();
    for (; cell!=endc; ++cell)
    {
      // First if cell lies on boundary of the inner sphere, then flag as spherical (manifold 100).
      // We could mark them all, but with refinement the 0 point causes problems.
      if (cell->material_id() == sphere_id)
      {
        for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
        {          
          if (cell->neighbor(face)->material_id() != sphere_id)
          {
            cell->face(face)->set_all_manifold_ids(sphere_manifold_id);
            cell->neighbor(face)->set_all_manifold_ids(sphere_manifold_id);
          }
        }
      }
      else if (cell->material_id() == free_space_id)
      {
        for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
        {
          const unsigned int check = cell->neighbor_index(face);
          if (check != -1)
          {
            if (cell->neighbor(face)->material_id() == sphere_id)
            {
              cell->face(face)->set_all_manifold_ids(sphere_manifold_id);
            }
          }
        }
      }
      // Mark all as spherical outside the central sphere.
//       else if (cell->material_id() == free_space_id )
//       {
//         cell->set_all_manifold_ids(sphere_manifold_id);
//       }
    }
    tria.set_manifold (sphere_manifold_id, sph_boundary);
    
    // Number the voxels inside the sphere:
    typename Triangulation<dim>::cell_iterator active_cell = tria.begin_active ();
    n_voxels=0;
    for (; active_cell!=endc; ++active_cell)
    {
      if (active_cell->material_id() == sphere_id)
      {
        active_cell->set_material_id(n_voxels+1);
        ++n_voxels;
      }
    }
    
    // global refinement if required:      
    tria.refine_global(href);
    
  }
  template <int dim>
  void InverseSolver<dim>::update_material_parameters(const Vector<double> &delta_sigma)
  {
    // Add the update to the voxels:
    for (unsigned int v=1;v<n_voxels+1; ++v)
    {
      double new_sigma = eddy_data_approx.param_sigma[v] + GN_update_parameter*delta_sigma(v-1);
      if (new_sigma > sigma_max)
      {
        new_sigma = sigma_max;
      }
      else if (new_sigma < sigma_min)
      {
        new_sigma=sigma_min;
      }
      eddy_data_approx.param_sigma[v] = new_sigma;
    }
    
    // As in the setup routine, need an extra copy to shared data to ensure compatibility with output
    for (unsigned int i=0; i<eddy_data_exact.n_materials; ++i)
    {
      EquationData::param_sigma(i) = eddy_data_approx.param_sigma[i];
    }
  }
  
    
  template <int dim>
  void InverseSolver<dim>::run(std::string &output_filename,
                                   const unsigned int href_in)
  {
    href = href_in;
    create_mesh();
    setup_fe_data();
    setup_inverse_data();
    
    deallog << "Number of active cells:       "
    << tria.n_active_cells()
    << std::endl;
    
    const MappingQ<dim> mapping(mapping_order, (mapping_order>1 ? true : false));

    dof_handler.distribute_dofs (fe);
    // First we calculate the "exact solutions" for the correct value of sigma.
    // Storage for all solutions:
    std::vector<Vector<double>> all_solutions_exact(inverse_data.number_of_excitations);
    deallog << "INVERSE EXACT SOLUTION" << std::endl;
    {

      ForwardSolver::EddyCurrent<dim, DoFHandler<dim>> eddy_exact(eddy_data_exact,
                                                                  mapping,
                                                                  dof_handler,
                                                                  fe);
      deallog << "Number of degrees of freedom: "
      << dof_handler.n_dofs()
      << std::endl;
      
      // assemble the matrix for the eddy current problem:
      deallog << "Assembling System Matrix...." << std::endl;
      eddy_exact.assemble_matrices(dof_handler);
      deallog << "Matrix Assembly complete. " << std::endl;
      
      // initialise the linear solver - precomputes any inverses for the preconditioner, etc:
      deallog << "Initialising Solver..." << std::endl;
      eddy_exact.initialise_solver();
      deallog << "Solver initialisation complete. " << std::endl;
      
      for (unsigned int exciter=0; exciter<inverse_data.number_of_excitations; ++exciter)
      {
        // Setup the boundary condition function.
        backgroundField::SphericalDipole<dim> boundary_conditions(dipole_data[exciter]);
        
        // assemble rhs
        deallog << "Assembling System RHS...." << std::endl;
        eddy_exact.assemble_rhs(dof_handler,
                                boundary_conditions);
        deallog << "Matrix RHS complete. " << std::endl;
        
        deallog << "Solving... " << std::endl;
        // solve system & storage in the vector of solutions:
        unsigned int n_gmres_iterations;
        eddy_exact.solve(all_solutions_exact[exciter],
                         n_gmres_iterations);
        
        deallog << "Computed solution. " << std::endl;
        
        double l2err_wholeDomain;
        double l2err_conductor;
        double hcurlerr_wholeDomain;
        double hcurlerr_conductor;
        
        MyVectorTools::calcErrorMeasures(mapping,
                                         dof_handler,
                                         all_solutions_exact[exciter],
                                         boundary_conditions,
                                         l2err_wholeDomain,
                                         l2err_conductor,
                                         hcurlerr_wholeDomain,
                                         hcurlerr_conductor);
        // Can't output the hcurl error at the moment - there's no implementation of curlA for the solution.
        deallog << "Excitation " << exciter << ":" << std::endl;
        deallog << "L2 Errors | Whole Domain: " << l2err_wholeDomain << "   Conductor Only:" << l2err_conductor << std::endl;
        //     deallog << "HCurl Error | Whole Domain: " << hcurlerr_wholeDomain << "   Conductor Only:" << hcurlerr_conductor << std::endl;
          // Short version:
//           std::cout << tria.n_active_cells()
//           << " " << dof_handler.n_dofs()
//           << " " << n_gmres_iterations
//           << " " << l2err_wholeDomain
//           << " " << l2err_conductor << std::endl;
          // removed.
          //     << " " << hcurlerr_wholeDomain
          //     << " " << hcurlerr_conductor << std::endl;
      }
      // only plot one of the excitations
      // pick number 3.
      {
        const unsigned int plot_excitation = 3;
        IO_Data::n_subdivisions = 3;
        std::ostringstream tmp;
        tmp << output_filename << "_true";
        backgroundField::SphericalDipole<dim> boundary_conditions(dipole_data[plot_excitation]);
        OutputTools::output_to_vtk<dim, DoFHandler<dim>>(mapping,
                                                         dof_handler,
                                                         all_solutions_exact[plot_excitation],
                                                         tmp.str(),
                                                         boundary_conditions);
      }
      
      // Calculate voltages from the solution.
      std::vector<std::vector<std::vector<double>>> simulated_voltages;
      
      NewVoltages::Voltage4<dim> voltage;
      voltage.calculateVoltages(inverse_data,
                                eddy_data_exact,
                                all_solutions_exact,
                                dof_handler,
                                mapping,
                                simulated_voltages);
      for (unsigned int exciter=0; exciter<inverse_data.number_of_excitations; ++exciter)
      {
        std::stringstream voltage_filename;
        voltage_filename << output_filename << "voltages_p"
        << poly_order << "_true" << exciter << ".out";
        std::ofstream voltage_file(voltage_filename.str());
        for (unsigned int sensor=0; sensor<inverse_data.number_of_measurements; ++sensor)
        {
          voltage_file << simulated_voltages[exciter][sensor][0] << " "
          << simulated_voltages[exciter][sensor][1] << std::endl;
        }
        voltage_file.close();
      }
    }
    // Begin Gauss-Newton loop
    Vector<double> last_solution(inverse_data.n_voxels);
    Vector<double> current_solution(inverse_data.n_voxels);
    last_solution = 0.0;
    current_solution = 0.0;

    // Initialise the inverse solver.
    InverseSolver_Voltages::InverseSolver_Voltages<dim, DoFHandler<dim,dim>>
    inverse(inverse_data,
            all_solutions_exact,
            eddy_data_exact,
            dof_handler,
            mapping);
    // Open log file:
    std::stringstream gn_log_filename;
    gn_log_filename << output_filename << "_p" << poly_order 
    << "_GN.log";
    std::ofstream GN_logfile(gn_log_filename.str());
    for (unsigned int gn_step=0; gn_step<max_GN_steps; ++gn_step)
    {
      GN_logfile << "GN-step " << gn_step << std::endl;
      // Now calculate the FE solutions for the current sigma value:
      // Storage for all solutions:
      std::vector<Vector<double>> all_solutions_approx(inverse_data.number_of_excitations);
      deallog << "INVERSE APPROX SOLUTION" << std::endl;
      {
        ForwardSolver::EddyCurrent<dim, DoFHandler<dim>> eddy_approx(eddy_data_approx,
                                                                     mapping,
                                                                     dof_handler,
                                                                     fe);
        deallog << "Number of degrees of freedom: "
        << dof_handler.n_dofs()
        << std::endl;
        
        // assemble the matrix for the eddy current problem:
        deallog << "Assembling System Matrix...." << std::endl;
        eddy_approx.assemble_matrices(dof_handler);
        deallog << "Matrix Assembly complete. " << std::endl;
        
        // initialise the linear solver - precomputes any inverses for the preconditioner, etc:
        deallog << "Initialising Solver..." << std::endl;
        eddy_approx.initialise_solver();
        deallog << "Solver initialisation complete. " << std::endl;
        
        for (unsigned int exciter=0; exciter<inverse_data.number_of_excitations; ++exciter)
        {
          // Setup the boundary condition function.
          backgroundField::SphericalDipole<dim> boundary_conditions(dipole_data[exciter]);
          
          // assemble rhs
          deallog << "Assembling System RHS...." << std::endl;
          eddy_approx.assemble_rhs(dof_handler,
                                   boundary_conditions);
          deallog << "Matrix RHS complete. " << std::endl;
          
          deallog << "Solving... " << std::endl;
          // solve system & storage in the vector of solutions:
          unsigned int n_gmres_iterations;
          eddy_approx.solve(all_solutions_approx[exciter],
                            n_gmres_iterations);
          
          deallog << "Computed solution. " << std::endl;
          
          double l2err_wholeDomain;
          double l2err_conductor;
          double hcurlerr_wholeDomain;
          double hcurlerr_conductor;

          MyVectorTools::calcErrorMeasures(mapping,
                                           dof_handler,
                                           all_solutions_approx[exciter],
                                           boundary_conditions,
                                           l2err_wholeDomain,
                                           l2err_conductor,
                                           hcurlerr_wholeDomain,
                                           hcurlerr_conductor);
          // Can't output the hcurl error at the moment - there's no implementation of curlA for the solution.
          deallog << "Excitation " << exciter << ":" << std::endl;
          deallog << "L2 Errors | Whole Domain: " << l2err_wholeDomain << "   Conductor Only:" << l2err_conductor << std::endl;
          //     deallog << "HCurl Error | Whole Domain: " << hcurlerr_wholeDomain << "   Conductor Only:" << hcurlerr_conductor << std::endl;
          // Short version:
//           std::cout << tria.n_active_cells()
//           << " " << dof_handler.n_dofs()
//           << " " << n_gmres_iterations
//           << " " << l2err_wholeDomain
//           << " " << l2err_conductor << std::endl;
          // removed.
          //     << " " << hcurlerr_wholeDomain
          //     << " " << hcurlerr_conductor << std::endl;
          
        }

        
        // Calculate voltages from the solution.
        std::vector<std::vector<std::vector<double>>> simulated_voltages;
        
        NewVoltages::Voltage4<dim> voltage;
        voltage.calculateVoltages(inverse_data,
                                  eddy_data_approx,
                                  all_solutions_approx,
                                  dof_handler,
                                  mapping,
                                  simulated_voltages);
        for (unsigned int exciter=0; exciter<inverse_data.number_of_excitations; ++exciter)
        {
          std::stringstream voltage_filename;
          voltage_filename << output_filename << "_voltages_p" << poly_order 
          << "_approx" << exciter << "_step" << gn_step << ".out";
          std::ofstream voltage_file(voltage_filename.str());
          for (unsigned int sensor=0; sensor<inverse_data.number_of_measurements; ++sensor)
          {
            voltage_file << simulated_voltages[exciter][sensor][0] << " "
            << simulated_voltages[exciter][sensor][1] << std::endl;
          }
          voltage_file.close();
        }
      }
      
      
      // Calculate the sensitivity system:
      inverse.assemble_sensitivity_rhs(all_solutions_approx,
                                       eddy_data_approx,
                                       dof_handler,
                                       mapping);
      
      inverse.assemble_sensitivity_matrix(all_solutions_approx,
                                          dof_handler,
                                          mapping);
      
      GN_logfile << "Current functional norm: " << inverse.return_functional_norm() << std::endl;
      last_solution=current_solution;
      inverse.gauss_newton_solve(last_solution,
                                 current_solution,
                                 gn_step);
      GN_logfile << "G-N update:" << std::endl;
      for (unsigned int i=0; i<inverse_data.n_voxels; ++i)
      {
        GN_logfile << current_solution(i) << std::endl;
      }

      update_material_parameters(current_solution);
      // Plot out the final solution:
      // only plot one of the excitations
      // pick number 3.
      {
        const unsigned int plot_excitation = 3;
        IO_Data::n_subdivisions = 3;
        std::ostringstream tmp;
        tmp << output_filename << "_step" <<gn_step;
        backgroundField::SphericalDipole<dim> boundary_conditions(dipole_data[plot_excitation]);
        OutputTools::output_to_vtk<dim, DoFHandler<dim>>(mapping,
                                                         dof_handler,
                                                         all_solutions_approx[plot_excitation],
                                                         tmp.str(),
                                                         boundary_conditions);
      }
    }
  }
}

int main (int argc, char* argv[])
{
  using namespace dealii;
  
  const int dim = 3;
  // Set default input:
  unsigned int p_order = 0;
  unsigned int href = 0;
  unsigned int mapping_order = 2;
  std::string output_filename = "sphere";
  
  // Allow for input from command line:
  if (argc > 0)
  {
    for (int i=1;i<argc;i++)
    {
      if (i+1 != argc)
      {
        std::string input = argv[i];
        if (input == "-p")
        {
          std::stringstream strValue;
          strValue << argv[i+1];
          strValue >> p_order;
        }
        if (input == "-m")
        {
          std::stringstream strValue;
          strValue << argv[i+1];
          strValue >> mapping_order;
          if (mapping_order < 1)
          {
            std::cout << "ERROR: mapping order must be > 0" << std::endl;
            return 1;
          }
        }
        if (input == "-o")
        {
          output_filename = argv[i+1];
        }
        if (input == "-h") // h refinement
        {
          std::stringstream strValue;
          strValue << argv[i+1];
          strValue >> href;
        }
      }
    }
  }
  
  // Only output to logfile, not console:
  deallog.depth_console(0);
  std::ostringstream deallog_filename;
  deallog_filename << output_filename << "_p" << p_order << ".deallog";
  std::ofstream deallog_file(deallog_filename.str());
  deallog.attach(deallog_file);
  
  InverseSolver::InverseSolver<dim> eddy_voltages(p_order,
                                                          mapping_order);
  eddy_voltages.run(output_filename,
                    href);
  
  deallog_file.close();
  return 0;
}