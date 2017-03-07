An MIT Inverse Solver
=====================

This code is provided to solve the magnetic induction tomography (MIT) inverse problem where the goal to recover the conductivity of a target object using measurements taken some distance away from the object in the form of voltage differences.
This is achieved through a regularised Gauss-Newton iterative scheme.

The forward solver driving the iterative scheme solves the eddy current approximation of the time-harmonic Maxwell equations. This makes use of H(curl)-conforming hp-finite element methods at arbitrary polynomial order. 
A fully benchmarked version of this code can be found at https://github.com/rosskynch/MIT_Forward. Note that there may be some minor alterations between the two forward solvers in these two repositories, but they are largely the same.

The deal.II library, found at http://www.dealii.org is required for this code to run.

Assuming deal.II 8.3 is installed and configured properly, then the code should run successfully. However, we recommend that the deal.II development branch dated July 6th (SHA hash 79583e56) is used to ensure total compatibility.

A library for computing complex bessel functions is also required, see here: https://github.com/valandil/complex_bessel

Usage:
--------

The solver is setup to run a test problem of a conducting sphere with low conductivity using excitation/sensor coils modelled as dipoles.

To run the sphere test code:

    $ cd inverse
    $ mkdir build
    $ cd build
    $ cmake ../
    $ make
    $ ./inverse -p 2 -m 2 -h 0

This will run the solver on a very coarse mesh (13 elements in total, with a second-order mapping), with polynomial order 2 for the hp-FEM solver. There is no output to the screen, and it may take a while to run (use make release to run an optimized version), but progressed can be seen in
the .deallog and .log files which are written in the run directory. When finished the code produces a number of output files which can be investigated using software such as MATLAB or python scripts.
Further, .vtk files will be produced which may be visualised using software such as PARAVIEW.

There are further command line options which can be seen in the source of the main file.

The intention is to provide a working prototype of the inverse solver, which can be demonstrated quickly, so the number of recovery pixels for this problem is fixed at 7. This can be easily changed inside the setup routines in the main file.

It is wise to consider the memory requirements of the code when considering fine meshes (increase in h) and higher polynomial orders (increase in p).

Work derived from this software:
--------
In addition to the terms of the GPL v3 license (see below), we kindly ask that any work using or derived from this code cites the following papers:

[1] R.M. Kynch, P.D. Ledger. Resolving the sign conflict problem for hp–hexahedral Nédélec elements with application to eddy current problems. Computers & Structures 181 (2017) 41-54. ISSN 0045-7949, http://dx.doi.org/10.1016/j.compstruc.2016.05.021

[2] R.M. Kynch, P.D. Ledger. TBC

[3] P.D. Ledger, S. Zaglmayr, hp-Finite element simulation of three-diemensional eddy current problems on multiply connected domains. Computer Methods in Applied Mechanics and Engineering 199 (2010) 3386-3401.

[4] J. Schoeberl and S. Zaglmayr,  High order Nedelec elements with local complete exact sequence properties Int. J. Comput. Math. Electr. Electron. Engrg (COMPEL) 24 (2005) 374-384.


License:
--------

The code in this repository is provided under the GPL v3 license, please see the file ./LICENSE for details.
