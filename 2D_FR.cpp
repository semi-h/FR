// 2D Flux Reconstruction Code - 22.11.2018
// Semih Akkurt
// git - master/viscous

//g++ -O2 -o run_FR 1D_FR.cpp -I/home/semih/lapack-3.8.0/LAPACKE/include -I/home/semih/lapack-3.8.0/CBLAS/include -L/home/semih/lapack-3.8.0 -llapacke -llapack -lcblas -lrefblas -lgfortran

// set initial condition
// evaluate derivatives of correction function at gauss legendre points and store
// compute F_D left and right lagrange fit evaluation constants


// loop
// using known solution, construct discontinious flux function
// call flux function for each cell boundary using reconstructed solution
// update solution points


#include <iostream>
#include <fstream>
#include <math.h>
#include <cmath>
#include <iomanip> 


#include <lapacke.h>
#include <lapacke_utils.h>
#include <cblas.h>

struct essential {
  double dt, jacob, j_x, j_y, mu, Pr, beta, tau, ndim, tol;
  int nvar, porder, nnode, nelem, nelem_x, nelem_y, columnL, maxIte, nRK, nse, nfe, sub_ite, writeOut;
} ;

const double gammaVal=1.4;

void set_solnPoints(int p, double **soln_coords, double **weights);
void set_lagrangeDerivs(int p, double *soln_coords, double **derivs);
void set_lagrangeInterCoeffs(int p, double *soln_coords, double **lagrInterL, double **lagrInterR);
void set_correctionDerivs(int p, double *soln_coords, double **hL, double **hR);


void roe_flux(double *q_L, double *q_R, double n_x, double n_y, double *fluxI, double *q_hat);
void calc_matrix_A(double *q_hat, double n_x, double n_y, double *matrix_A);
void nodeViscFlux(double *U_vals, double *pd_U_x, double *pd_U_y, double mu, double Pr, double *fv_node, double *gv_node);

void F_flux_jacob( essential *params, int i_elem, double *jacob, int i_start, int jCN, double *u_vals, int i_u );
void G_flux_jacob( essential *params, int i_elem, double *jacob, int i_start, int jCN, double *u_vals, int i_u );
void correct_Flux( essential *params, int i_elem, double *jacob, int i_start, int jCN, double *u_vals, int i_u, double n_x, double n_y );

void writeSolution(essential *params, double *u, int iter);

// NEW SET OF FUNCTIONS COMPUTABLE WITH NAVIER-STOKES;

void deriveGiven( essential *params, 
                  double *pd_X_ksi, double *pd_Y_eta, double *X, double *Y, double *fixed_f,
                  double *lagrDerivs, double *hL, double *hR );

void extrapToFace( essential *params,
                   double *X, double *Y, double *face,
                   double *lagrInterL, double *lagrInterR );
void getInterFlux( essential* params,
                   double *f_face, double *u_face, double *q_x_face, double *q_y_face, double *u_hat_face );
void getFlux( essential* params, 
              double *u, double *q_x, double *q_y, double *f, double *g );
void getCommonU(essential *params, double *u_face);



int main()
{
  double ql[4] = {1, 1.5, 0.5, 2.5};
  double qr[4] = {1, 0.5, 1.5, 2.5};
  double q_hat[4] = {0};
  double fluxx[4];
  roe_flux(ql, qr, 1.0, 0.0, fluxx, q_hat);
std::cout << fluxx[0] << " " << fluxx[1] << " " << fluxx[2] << " " << fluxx[3] << "\n";
  double L = 10;
  essential params;
  double L_x, L_y;
  //int nelem_x, nelem_y, nnode;
  //double J_x, J_y, Jacob;

  L_x = 20;
  L_y = 20;
  params.nelem_x = 10;
  params.nelem_y = 10;
  params.nnode = (params.nelem_x+1)*(params.nelem_y+1);
  params.j_x = L_x/params.nelem_x/2.0;
  params.j_y = L_y/params.nelem_y/2.0;
  params.jacob = params.j_x*params.j_y;

  params.beta = 0.5;
  params.tau = 0.1;
  params.mu = 0.417;
  params.Pr = 0.72;

  params.ndim   = 2;
  params.nvar   = 4;
  params.porder = 3; // 0 || 1 || 2 || 3
  params.nse    = pow(params.porder+1,2);
  params.nfe    = 4*(params.porder+1); //only for quad
  params.dt     = 0.2;
  params.nelem  = params.nelem_x*params.nelem_y;
  params.maxIte = 25;//3125
  params.writeOut = 100000;
  params.nRK = 6; //1 || 4
  params.columnL = params.nvar*params.nelem;
  //params.jacob  = L/params.nelem/2;
  params.sub_ite = 100;
  params.tol = -8.0;

  bool withMagicA = true;
  bool analyticFJ = false;

  //create mesh!
  //no need for a mesh
  // just arrange flux pairs so that its periodic on sides
  // to the right, the normal is always [1 0]; to the top, it is [0 1]


  double *soln_coords, *weights;
  set_solnPoints(params.porder, &soln_coords, &weights);

  double *lagrDerivs;
  set_lagrangeDerivs(params.porder, soln_coords, &lagrDerivs);

  double *lagrInterL, *lagrInterR;
  set_lagrangeInterCoeffs(params.porder, soln_coords, &lagrInterL, &lagrInterR);

  double *hL, *hR;
  set_correctionDerivs(params.porder, soln_coords, &hL, &hR);


  // solution arrays, array of structures
  double *u, *f, *g, *u_face, *f_face;
  u = new double [params.nse*params.columnL];
  f = new double [params.nse*params.columnL];
  g = new double [params.nse*params.columnL];
  u_face = new double [params.nfe*params.columnL];
  f_face = new double [params.nfe*params.columnL];

  // perturbed_u array
  double *perturbed_u;
  perturbed_u = new double [params.nse*params.columnL];

  //visc only arrays;
  double *q_x, *q_y, *q_x_face, *q_y_face;
  q_x = new double [params.nse*params.columnL];
  q_y = new double [params.nse*params.columnL];
  q_x_face = new double [params.nfe*params.columnL];
  q_y_face = new double [params.nfe*params.columnL];

  for ( int i = 0; i < params.nse*params.columnL; i++ )
  { u[i] = 0; f[i] = 0; g[i] = 0; q_x[i] = 0; q_y[i] = 0; }
  for ( int i = 0; i < params.nfe*params.columnL; i++ )
  { u_face[i] = 0; f_face[i] = 0; q_x_face[i] = 0; q_y_face[i] = 0; }

  double *u_hat_face;
  u_hat_face = new double [params.nfe*params.columnL];

  // element wise implicit matrix
  double *magicA;
  magicA = new double [params.nvar*params.nse*params.nvar*params.nse];
  double *Fblock;
  Fblock = new double [params.nvar*(params.porder+1)*params.nvar];
  double *Gblock;
  Gblock = new double [params.nvar*(params.porder+1)*params.nvar];
  double *rhs;
  rhs = new double [params.nvar*params.nse];

  double *d_magicA;
  d_magicA = new double [params.nvar*params.nse*params.nvar*params.nse];


  // element wise interface jacobians;
  double *F_I_LR, *G_I_BT;
  F_I_LR = new double [params.nvar*params.nvar*2];
  G_I_BT = new double [params.nvar*params.nvar*2];

  // 2N storage RK
  double *old_u;
  old_u = new double [params.nse*params.columnL];
  double *u_curr;
  u_curr = new double [params.nse*params.columnL];

  //butcher table
  double *butcher_a;
  butcher_a = new double [params.nRK*params.nRK];
  double *butcher_b, *butcher_c;
  butcher_b = new double [params.nRK];
  butcher_c = new double [params.nRK];
  // dirk storage
  double **ks;
  ks = new double*[params.nRK];
  for ( int i = 0; i < params.nRK; i++ )
    ks[i] = new double [params.nse*params.columnL];

  std::cout << "ndim: " << params.ndim << "\n";
  std::cout << "polynomial order is: " << params.porder << "\n";
  std::cout << "number of solution points per element: " << params.nse << "\n";
  std::cout << "number of face points per element: " << params.nfe << "\n";
  std::cout << "nvar: " << params.nvar << "\n";
  std::cout << "max ite: " << params.maxIte << "\n";
  std::cout << "nelem: " << params.nelem << "\n";
  std::cout << "columnL: " << params.columnL << "\n";
  std::cout << "dx: " << L_x/params.nelem_x << "\n";
  std::cout << "dy: " << L_y/params.nelem_y << "\n";
  std::cout << "Lenght: " << L << "\n";
  std::cout << "jacob: " << params.jacob << "\n";
  std::cout << "jacob x: " << params.j_x << "\n";
  std::cout << "jacob y: " << params.j_y << "\n";
  std::cout << "dt: " << params.dt << "\n";
  std::cout << "end time: " << params.maxIte*params.dt << "\n";
  std::cout << "RK stage: " << params.nRK << std::endl;

  double *alpha;
  alpha = new double [params.nRK];


  for ( int i = 0; i < params.nRK*params.nRK; i++ ) butcher_a[i] = 0;

  if ( params.nRK == 1 )
  {
    alpha[0] = 1.0;

    butcher_a[0] = 1;
    butcher_b[0] = 1;
  }
  else if ( params.nRK == 4 )
  {
    alpha[0] = 0.25; alpha[1] = 1.0/3.0; alpha[2] = 0.5; alpha[3] = 1.0;
    std::cout << "alpha: ";
    for ( int i = 0; i < params.nRK; i++ ) std::cout << alpha[i] << " ";
    std::cout << std::endl;

    butcher_a[4] = 0.5; butcher_a[9] = 0.5; butcher_a[14] = 1;
    butcher_b[0] = 1.0/6.0; butcher_b[1] = 1.0/3.0; butcher_b[2] = 1.0/3.0; butcher_b[3] = 1.0/6.0;
  }
  else if ( params.nRK == 3 )
  {
    double pi = 4.0*atan(1.0), param = 2.0*cos(pi/18.0)/sqrt(3);
    std::cout << "alpha param: " << param << "\n";

    butcher_a[0] = (1+param)*0.5;
    butcher_a[3] =-param*0.5;     butcher_a[4] = (1+param)*0.5;
    butcher_a[6] = 1+param;       butcher_a[7] =-(1+2*param);   butcher_a[8] = (1+param)*0.5;

    butcher_b[0] = 1/(6*param*param); butcher_b[1] = 1-1/(3*param*param); butcher_b[2] = 1/(6*param*param);

    //butcher_c[0] = (1+param)*0.5; butcher_c[1] = 0.5; butcher_c[2] = (1-param)*0.5;

  }
  else if ( params.nRK == 6 )
  {
    //1st row
    butcher_a[0] = 0.25;
    //2nd row
    butcher_a[6] = 0.25; butcher_a[7] = 0.25;
    //3rd row
    butcher_a[12] = 8611.0/62500.0; butcher_a[13] =-1743.0/31250.0; butcher_a[14] = 0.25;
    //4th row
    butcher_a[18] = 5012029.0/34652500.0; butcher_a[19] = -654441.0/2922500.0; butcher_a[20] = 174375.0/388108.0;
      butcher_a[21] = 0.25;
    //5th row
    butcher_a[24] = 15267082809.0/155376265600.0; butcher_a[25] =-71443401.0/120774400.0; butcher_a[26] = 730878875.0/902184768.0;
      butcher_a[27] = 2285395.0/8070912.0; butcher_a[28] = 0.25;
    //6th row
    butcher_a[30] = 82889.0/524892.0; butcher_a[31] = 0; butcher_a[32] = 15625.0/83664.0;
      butcher_a[33] = 69875.0/102672.0; butcher_a[34] =-2260.0/8211.0; butcher_a[35] = 0.25;

    butcher_b[0] = 82889.0/524892.0; butcher_b[1] = 0; butcher_b[2] = 15625.0/83664.0;
      butcher_b[3] = 69875.0/102672.0; butcher_b[4] =-2260.0/8211.0; butcher_b[5] = 0.25;
  }


  std::cout << "butcher table;\n";
  std::cout << "butcher a matrix;\n";
  for ( int i = 0; i < params.nRK; i++ )
  {
    for ( int j = 0; j < params.nRK; j++ )
    {
      std::cout << butcher_a[i*params.nRK+j] << " ";
    }
    std::cout << "\n";
  }
  std::cout << "butcher b values;\n";
  for ( int i = 0; i < params.nRK; i++ ) std::cout << butcher_b[i] << " ";
  std::cout << "\n";


  // Initialization
  double rho_0 = 1, u_0 = 0, p_0 = 1;
  double rho_1 = 0.125, u_1 = 0, p_1 = 0.1;

  if ( params.nvar == 3 )
  {
  double q[3], flux[3];
  std::cout << "left side\n";
  q[0] = rho_0; q[1] = rho_0*u_0; q[2] = p_0/(gammaVal-1) + 0.5*rho_0*pow(u_0,2);
  for ( int i = 0; i < params.nelem/2; i++ )
  {
    for ( int j = 0; j < params.porder+1; j++ ) 
    { //rowN*columnL + columnN
      //for ( int k = 0; k < nvar; k++ ) u[j][i+k*nelem] = q[k];
      for ( int k = 0; k < params.nvar; k++ )
      {
        u[j*params.columnL + i+k*params.nelem] = q[k];
        old_u[j*params.columnL + i+k*params.nelem] = 0;
      }
    }
  }
  std::cout << "right side\n";
  q[0] = rho_1; q[1] = rho_1*u_1; q[2] = p_1/(gammaVal-1) + 0.5*rho_1*pow(u_1,2);
  for ( int i = params.nelem/2; i < params.nelem; i++ )
  {
    for ( int j = 0; j < params.porder+1; j++ ) 
    {
      for ( int k = 0; k < params.nvar; k++ )
      {
        u[j*params.columnL + i+k*params.nelem] = q[k];
        old_u[j*params.columnL + i+k*params.nelem] = 0;
      }
    }
  }
  }
  else if ( params.nvar == 1 )
  {
    //gaussian bump
    std::cout << "Advection solver\n";
    double x, sigma=0.2, pi=3.141592653589793, mu=0;
    std::cout << "Gaussian bump with sigma = " << sigma << " and mu = " << mu << "\n";
    for ( int i = 0; i < params.nelem; i++ )
    {
      //x = i*L/nelem-5;
      for ( int j = 0; j < params.porder+1; j++ )
      {
      x = i*L/params.nelem-5 + soln_coords[j]*params.jacob;
      u[j*params.columnL + i] = 1/(sigma*sqrt(2*pi))*exp(-0.5*pow((x-mu)/sigma,2));
      old_u[j*params.columnL + i] = 0;
      }
    }
  }
  else if ( params.nvar == 4 )
  {
    if ( true )
    {
      std::cout << "Euler vortex\n";
      double S = 13.5;    // Strength
      double M = 0.4;     // Mach number
      double R = 1.5;     // Radius
      double pi = 3.141592653589793;
      double f, rho, u_vel, v_vel, p, x, y;
      double x_L =-10, x_R = 10, y_B =-10, y_T = 10;
      double dx = (x_R-x_L)/params.nelem_x, dy = (y_T-y_B)/params.nelem_y;
      for ( int i = 0; i < params.nelem_x; i++ )
      {
        for ( int j = 0; j < params.nelem_y; j++ )
        {
          for ( int ii = 0; ii < params.porder+1; ii++ )
          {
            for ( int jj = 0; jj < params.porder+1; jj++ )
            {
              x = x_L + i*dx + dx/2.0 + soln_coords[ii]*params.j_x;
              y = y_B + j*dy + dy/2.0 + soln_coords[jj]*params.j_y;
              f = ((1 - x*x - y*y)/(2*R*R));
              rho = pow(1 - S*S*M*M*(gammaVal - 1)*exp(2*f)/(8*pi*pi), 1/(gammaVal - 1));
              u_vel = 1 + S*y*exp(f)/(2*pi*R);
              v_vel = 1 - S*x*exp(f)/(2*pi*R);
              p = 1/(gammaVal*M*M)*pow(1 - S*S*M*M*(gammaVal - 1)*exp(2*f)/(8*pi*pi), gammaVal/(gammaVal - 1));
              int row_loc = (jj*(params.porder+1)+ii)*params.columnL;
              u[row_loc+(j*params.nelem_x+i)] = rho;
              u[row_loc+(j*params.nelem_x+i)+params.nelem] = rho*u_vel;
              u[row_loc+(j*params.nelem_x+i)+2*params.nelem] = rho*v_vel;
              u[row_loc+(j*params.nelem_x+i)+3*params.nelem] = p/(gammaVal-1.0)
                                                + 0.5*rho*(u_vel*u_vel+v_vel*v_vel);
            }
          }
        }
      }
    }
    else if ( false ) // constant q_inf
    {
      double rho_inf = 1, u_inf = 1, v_inf = 0, p_inf = 1;
      std::cout << "const speed, u=" << u_inf << ", v=" << v_inf << "\n";
      for ( int i = 0; i < params.nelem; i++ )
      {
        for ( int j = 0; j < params.nse; j++ )
        {
          u[j*params.columnL+i+0*params.nelem] = rho_inf;
          u[j*params.columnL+i+1*params.nelem] = rho_inf*u_inf;
          u[j*params.columnL+i+2*params.nelem] = rho_inf*v_inf;
          u[j*params.columnL+i+3*params.nelem] = p_inf/(gammaVal-1.0)
                                               + 0.5*rho_inf*(u_inf*u_inf+v_inf*v_inf);
        }
      }
      for ( int i = 0; i < params.nelem; i++ )
      {
        if ( i >= 0 && i < 1250 )
        {
        for ( int j = 0; j < params.nse; j++ )
        {
          u[j*params.columnL+i+0*params.nelem] = rho_inf;
          u[j*params.columnL+i+1*params.nelem] = 0;
          u[j*params.columnL+i+2*params.nelem] = 0;
          u[j*params.columnL+i+3*params.nelem] = p_inf/(gammaVal-1.0);
        }
        }
      }
    }
    else if ( false ) // couette flow
    {
      double p_c = 100000;
      double T_w = 300;
      double c_p = 1005;
      double u_w = 69.445;  // these values vorrespond to a Mach=0.2, Re=200;

      double rho, u_vel, v_vel, p, x, y, phi;
      double x_L =-10, x_R = 10, y_B =-10, y_T = 10;
      double dx = (x_R-x_L)/params.nelem_x, dy = (y_T-y_B)/params.nelem_y;
      for ( int i = 0; i < params.nelem_x; i++ )
      {
        for ( int j = 0; j < params.nelem_y; j++ )
        {
          for ( int ii = 0; ii < params.porder+1; ii++ )
          {
            for ( int jj = 0; jj < params.porder+1; jj++ )
            {
              x = x_L + i*dx + dx/2.0 + soln_coords[ii]*params.j_x;
              y = y_B + j*dy + dy/2.0 + soln_coords[jj]*params.j_y;
              phi = (y-y_B)/(y_T-y_B);
              u_vel = phi*u_w;
              //if ( j < 25 ) u_vel = 0;
              //else u_vel = u_w;
              v_vel = 0;
              p = p_c;
              rho = gammaVal/(gammaVal-1)*(2.0*p)/(2*c_p*T_w + params.Pr*u_w*u_w*phi*(1.0-phi));
              int row_loc = (jj*(params.porder+1)+ii)*params.columnL;
              u[row_loc+(j*params.nelem_x+i)] = rho;
              u[row_loc+(j*params.nelem_x+i)+params.nelem] = rho*u_vel;
              u[row_loc+(j*params.nelem_x+i)+2*params.nelem] = rho*v_vel;
              u[row_loc+(j*params.nelem_x+i)+3*params.nelem] = p/(gammaVal-1.0)
                                                + 0.5*rho*(u_vel*u_vel+v_vel*v_vel);
            }
          }
        }
      }

    }
    else // gaussian bump in 2D
    {
      double x, y, sigma=10, pi=3.141592653589793, mu=0;
      std::cout << "Gaussian bump with sigma = " << sigma << " and mu = " << mu << "\n";
      double x_L =-100, x_R = 100, y_B =-100, y_T = 100;
      double dx = (x_R-x_L)/params.nelem_x, dy = (y_T-y_B)/params.nelem_y;
      for ( int i = 0; i < params.nelem_x; i++ )
      {
        for ( int j = 0; j < params.nelem_y; j++ )
        {
          for ( int ii = 0; ii < params.porder+1; ii++ )
          {
            for ( int jj = 0; jj < params.porder+1; jj++ )
            {
              x = x_L + i*dx + dx/2.0 + soln_coords[ii]*params.j_x;
              y = y_B + j*dy + dy/2.0 + soln_coords[jj]*params.j_y;

              int row_loc = (jj*(params.porder+1)+ii)*params.columnL;
              u[row_loc+(j*params.nelem_x+i)] = 1/(sigma*sqrt(2*pi))*exp(-0.5*pow((x-mu)/sigma,2));
              u[row_loc+(j*params.nelem_x+i)+params.nelem] = 1/(sigma*sqrt(2*pi))*exp(-0.5*pow((y-mu)/sigma,2));
              u[row_loc+(j*params.nelem_x+i)+2*params.nelem] = 1/(sigma*sqrt(2*pi))*exp(-0.5*pow(((x+y)*0.5-mu)/sigma,2));
              u[row_loc+(j*params.nelem_x+i)+3*params.nelem] = 1/(sigma*sqrt(2*pi))*exp(-0.5*pow(((x-y)*0.5-mu)/sigma,2));
            }
          }
        }
      }


    }
  }

  double CONSERVATION = 0;
  for ( int j = 0; j < params.nelem; j++ )
  {
    for ( int k = 0; k < params.porder+1; k++ )
    {
      for ( int l = 0; l < params.porder+1; l++ )
      {
        CONSERVATION += u[(k*(params.porder+1)+l)*params.columnL+j]*weights[k]*weights[l]*params.jacob;
      }
    }
  }

  std::cout << "conversation(rho): " << CONSERVATION << "\n";

  int *loc_u = new int[params.nvar];
  double *dummyU = new double[params.nvar*params.nse];

  double ress, conservation;
  double res[4] = {0};

  std::cout << "main loop begins\nite: \n";
  //std::ofstream fface;
  //fface.open("f_face");
  //std::ofstream uvals;
  //uvals.open("u_vals");
  std::ofstream fmagicA;
  fmagicA.open("magicA");
  // MAIN LOOP
  for ( int ite = 0; ite < params.maxIte; ite++ )
  {
    std::cout << " ite: " << ite << "\n";
    for ( int i = 0; i < params.nse*params.columnL; i++ ) old_u[i] = u[i];
    for ( int i_RK = 0; i_RK < params.nRK; i_RK++ ) {
      for ( int i = 0; i < params.nse*params.columnL; i++ ) ks[i_RK][i] = 0;
    }
    for ( int i_RK = 0; i_RK < params.nRK; i_RK++ )
    {

      // DIRK
      // initial guess for intermediate negDiv, ks
      for ( int j = 0; j < params.nse*params.columnL; j++ )
      {
        u[j] = old_u[j];
        for ( int k = 0; k < i_RK+1; k++ )
        {
          u[j] += butcher_a[i_RK*params.nRK+k]*ks[k][j]*params.dt;
        }
      }

ress = 10; res[0] = 10; res[1] = 10; res[2] = 10; res[3] = 10;
for ( int sub_ite = 0; sub_ite < params.sub_ite; sub_ite++ )
{
      if ( ress < params.tol ) continue;
      ress = 0; // if residual small enough exit sub ite loop
      res[0] = 0; res[1] = 0; res[2] = 0; res[3] = 0;
      std::cout << "    subIte" << std::setw(4) << sub_ite;
      //std::cout << "ite: " << ite << " ";


      // we need u_face first
      extrapToFace(&params, u, u, u_face, lagrInterL, lagrInterR);
      // then corrected u_face with the common u_I
      getCommonU(&params, u_face);
      // know we can get the q_x and q_y
      deriveGiven(&params, q_x, q_y, u, u, u_face, lagrDerivs, hL, hR);
      // get u_face back
      extrapToFace(&params, u, u, u_face, lagrInterL, lagrInterR);
      // get q_x_face adn q_y_face
      extrapToFace(&params, q_x, q_x, q_x_face, lagrInterL, lagrInterR);
      extrapToFace(&params, q_y, q_y, q_y_face, lagrInterL, lagrInterR);
      // get flux
      getFlux(&params, u, q_x, q_y, f, g);
      // get f_face
      extrapToFace(&params, f, g, f_face, lagrInterL, lagrInterR);
      // get interface flux (both inviscid and viscous within the same func)
      getInterFlux(&params, f_face, u_face, q_x_face, q_y_face, u_hat_face);
      // get the final result
           for ( int j = 0; j < params.nse*params.columnL; j++ ) u_curr[j] = u[j];
      deriveGiven(&params, u, u, f, g, f_face, lagrDerivs, hL, hR);

double resM = 0;
if ( !withMagicA ) {
      //explicit update
      for ( int j = 0; j < params.nse*params.columnL; j++ )
      {
        //u[j] = old_u[j] - alpha[i_RK]*params.dt*u[j]; //// for low storage RK
        ks[i_RK][j] =-u[j];
        ress += u[j]*u[j];
      }
      ress = log10(sqrt(ress));
}
else{
      // assign neg div to k
      //for ( int j = 0; j < params.nse*params.columnL; j++ ) ks[i_RK][j] = u[j];

      // change here for the element wise implicit


      // for each element, create a local matrix, and update the solution by solving resulting linear system
      for ( int i_elem = 0; i_elem < params.nelem; i_elem++ )
      { //std::cout << i_elem << "\n";
if (analyticFJ) {
        // analytic jacobian calls;
        // create the matrix using flux function for each element
        for ( int j = 0; j < params.nvar*params.nse*params.nvar*params.nse; j++ ) magicA[j] = 0;
        for ( int j = 0; j < params.porder+1; j++ )
        {

          for ( int k = 0; k < params.porder+1; k++ )
          {
            F_flux_jacob( &params, i_elem, Fblock, k*params.nvar, 
                          params.nvar*(params.porder+1), old_u, j*(params.porder+1)+k );//u_curr!!!!!!!!!!

            G_flux_jacob( &params, i_elem, Gblock, k*params.nvar, 
                          params.nvar*(params.porder+1), old_u, k*(params.porder+1)+j );//u_curr!!!!!!!!!!
          }


          //calc_matrix_A(double *q_hat, double n_x, double n_y, double *matrix_A)

          // Evaluate interface flux jacobians;
          // store in F_I_LR G_I_BT
          //F_flux_jacob( essential *params, int i_elem, double *jacob, int i_start, int jCN, double *u_vals, int i_u );
          // F_L, F_R, G_B, G_T respectively;
          // F_L;
          int face_i;
          face_i = (params.porder+1)*(params.porder+1)-1-j; // left
          F_flux_jacob( &params, i_elem, F_I_LR, 0, params.nvar*2, u_face, face_i );
          correct_Flux( &params, i_elem, F_I_LR, 0, params.nvar*2, u_hat_face, face_i, 1, 0 );
          // F_R;
          face_i = params.porder+1+j; // right
          F_flux_jacob( &params, i_elem, F_I_LR, 4, params.nvar*2, u_face, face_i );
          correct_Flux( &params, i_elem, F_I_LR, 4, params.nvar*2, u_hat_face, face_i, 1, 0 );
          // G_B
          face_i = j; // bottom
          G_flux_jacob( &params, i_elem, G_I_BT, 0, params.nvar*2, u_face, face_i );
          correct_Flux( &params, i_elem, G_I_BT, 0, params.nvar*2, u_hat_face, face_i, 0, 1 );
          // G_T
          face_i = 3*(params.porder+1)-1-j; // top
          G_flux_jacob( &params, i_elem, G_I_BT, 4, params.nvar*2, u_face, face_i );
          correct_Flux( &params, i_elem, G_I_BT, 4, params.nvar*2, u_hat_face, face_i, 0, 1 );

          // interface flux jacobs are ready, now add these into local jacob


          for ( int k = 0; k < params.porder+1; k++ )
          {
            //multiply F/Gblock with correct coeffs and add it to the magicA
            int i_magicA;
            // add Fblock to magicA
            for ( int l = 0; l < params.porder+1; l++ )
            {
              i_magicA = (j*(params.porder+1)+k)*params.nvar*params.nvar*params.nse
                       + j*(params.porder+1)*params.nvar + l*params.nvar;
              for ( int ii = 0; ii < params.nvar; ii++ )
              {
                for ( int jj = 0; jj < params.nvar; jj++ )
                {
                  magicA[i_magicA+ii*params.nse*params.nvar+jj] 
                    += Fblock[ii*params.nvar*(params.porder+1)+l*params.nvar+jj]
                      *( lagrDerivs[k*(params.porder+1)+l]
                       - lagrInterL[l]*hL[k]
                       - lagrInterR[l]*hR[k] )
                     + F_I_LR[ii*params.nvar*2+0*params.nvar+jj]*lagrInterL[l]*hL[k]*0.5  //left
                     + F_I_LR[ii*params.nvar*2+1*params.nvar+jj]*lagrInterR[l]*hR[k]*0.5; //right
                }
              }
            }
            // add Gblock to magicA
            for ( int l = 0; l < params.porder+1; l++ )
            {
              i_magicA = (k*(params.porder+1)+j)*params.nvar*params.nvar*params.nse
                       + j*params.nvar + l*params.nvar*(params.porder+1);
              for ( int ii = 0; ii < params.nvar; ii++ )
              {
                for ( int jj = 0; jj < params.nvar; jj++ )
                {
                  magicA[i_magicA+ii*params.nse*params.nvar+jj] 
                    += Gblock[ii*params.nvar*(params.porder+1)+l*params.nvar+jj]
                     *( lagrDerivs[k*(params.porder+1)+l]
                      - lagrInterL[l]*hL[k]
                      - lagrInterR[l]*hR[k] )
                     + G_I_BT[ii*params.nvar*2+0*params.nvar+jj]*lagrInterL[l]*hL[k]*0.5  //left
                     + G_I_BT[ii*params.nvar*2+1*params.nvar+jj]*lagrInterR[l]*hR[k]*0.5; //right
                }
              }
            }
          }
        }

//for ( int j = 0; j < params.nvar*params.nse*params.nvar*params.nse; j++ ) d_magicA[j] = magicA[j];

} //analytic FJ
else { // numeric jacob
// UP TO THIS POINT ITS JUST magicA

// replace it with the finite differencing the residual method
// this is still within the element

// perturb the state values of all solution points within the cell one by one 
// and synthesize the flux jacobian matrix in this way column by column




// [sol_i*params.columnL+var_i*params.nelem+i_elem]
        for ( int j = 0; j < params.nvar*params.nse*params.nvar*params.nse; j++ ) magicA[j] = 0;
        for ( int sol_i = 0; sol_i < params.nse; sol_i++ )
        {
          for ( int var_i = 0; var_i < params.nvar; var_i++ )
          {
            double eps = 0.000001;
            for ( int j = 0; j < params.nse*params.columnL; j++ ) perturbed_u[j] = u_curr[j];
            perturbed_u[sol_i*params.columnL+var_i*params.nelem+i_elem] += eps;
            // we need u_face first
            extrapToFace(&params, perturbed_u, perturbed_u, u_face, lagrInterL, lagrInterR);
            // then corrected u_face with the common u_I
            getCommonU(&params, u_face);
            // know we can get the q_x and q_y
            deriveGiven(&params, q_x, q_y, perturbed_u, perturbed_u, u_face, lagrDerivs, hL, hR);
            // get u_face back
            extrapToFace(&params, perturbed_u, perturbed_u, u_face, lagrInterL, lagrInterR);
            // get q_x_face adn q_y_face
            extrapToFace(&params, q_x, q_x, q_x_face, lagrInterL, lagrInterR);
            extrapToFace(&params, q_y, q_y, q_y_face, lagrInterL, lagrInterR);
            // get flux
            getFlux(&params, perturbed_u, q_x, q_y, f, g);
            // get f_face
            extrapToFace(&params, f, g, f_face, lagrInterL, lagrInterR);
            // get interface flux (both inviscid and viscous within the same func)
            getInterFlux(&params, f_face, u_face, q_x_face, q_y_face, u_hat_face);
            // get the final result
            deriveGiven(&params, perturbed_u, perturbed_u, f, g, f_face, lagrDerivs, hL, hR);

            for ( int j = 0; j < params.nse; j++ )
            {
              for ( int k = 0; k < params.nvar; k++ )
              {
                int row_loc, col_loc;
                row_loc = j*params.nvar+k;
                col_loc = sol_i*params.nvar+var_i;
                magicA[row_loc*params.nvar*params.nse + col_loc] 
                                    = ( perturbed_u[j*params.columnL+k*params.nelem+i_elem]
                                      - u[j*params.columnL+k*params.nelem+i_elem] )/eps;
              }
            }
          }
        }
}//numeric J

//for ( int j = 0; j < params.nvar*params.nse*params.nvar*params.nse; j++ ) resM += pow(magicA[j]-d_magicA[j],2);

        // DIRK
        for ( int j = 0; j < params.nvar*params.nse*params.nvar*params.nse; j++ )
        {
          magicA[j] *= params.dt*butcher_a[i_RK*params.nRK+i_RK];
        }

        for ( int j = 0; j < params.nvar*params.nse; j++ )
        {
          magicA[j*params.nvar*params.nse+j] += 1;//.0/params.dt;
        }
/*
////////
        // TEMP- write magicA into file
        fmagicA << i_elem << "\n";
        for ( int j = 0; j < params.nse*params.nvar*params.nse*params.nvar; j++ )
        {
          int val;
          if ( magicA[j] != 0 ) val = 8;
          else val = 0;
          if ( magicA[j] != magicA[j] ) val = 666666;
          fmagicA << std::scientific << magicA[j] << " "; if ( (j+1)%(params.nse*params.nvar) == 0 ) fmagicA << "\n";
          if ( (j+1)%(params.nse*params.nvar*params.nvar) == 0 ) fmagicA << "\n";
          //fmagicA << val << " "; if ( (j+1)%(params.nse*params.nvar) == 0 ) fmagicA << "\n";
        }
        fmagicA << "\n\n";
/////////
*/


        // rhs; u is assigned as rhs temporarily after update func call 
        for ( int j = 0; j < params.nvar; j++ )
        {
          for ( int k = 0; k < params.nse; k++ )
          {
            //rhs[k*params.nvar+j] =-u[k*params.columnL+i_elem+j*params.nelem];
            //                     - 1/params.dt
            //                      *( u_curr[k*params.columnL+i_elem+j*params.nelem]
            //                       - old_u[k*params.columnL+i_elem+j*params.nelem] );

            //DIRK;
            rhs[k*params.nvar+j] =-ks[i_RK][k*params.columnL+i_elem+j*params.nelem]
                                 - u[k*params.columnL+i_elem+j*params.nelem];
                                 //+ 1/params.dt
                                 // *( u_curr[k*params.columnL+i_elem+j*params.nelem]
                                 //  - old_u[k*params.columnL+i_elem+j*params.nelem] );

            //if ( j == 0 ) 
            //ress += rhs[k*params.nvar+j]*rhs[k*params.nvar+j];
            res[j] += rhs[k*params.nvar+j]*rhs[k*params.nvar+j];
          }
        }
        // solve the system

        int N = params.nse*params.nvar;
        int *IPIV = new int[N+1];
        int INFO;

        //LAPACKE_dgesv (int matrix_layout, lapack_int n, lapack_int nrhs, double *a, lapack_int lda, lapack_int *ipiv, double *b, lapack_int ldb)
        INFO = LAPACKE_dgesv(LAPACK_ROW_MAJOR,N,1,magicA,N,IPIV,rhs,1);
        //if ( INFO != 0 ) {std::cout << "dgesv, elem# " << i_elem << " info: " << INFO << "\n\n";
        //for ( int j = 0; j<params.nvar*params.nse; j++ ) if(rhs[j]!=rhs[j]) std::cout << rhs[j] << " ";
        //std::cout << "\n";}

        delete[] IPIV;

        for ( int j = 0; j < params.nvar; j++ )
        {
          for ( int k = 0; k < params.nse; k++ )
          {
            //u[k*params.columnL+i_elem+j*params.nelem] = old_u[k*params.columnL+i_elem+j*params.nelem] 
            //                                          + alpha[i_RK]*rhs[k*params.nvar+j];

            //DIRK;  // if i_RK == 0; it is the first explicit iteration, the matrix should be identity
                     // which makes the solution equal to rhs given above;

            //DIRK
            ks[i_RK][k*params.columnL+i_elem+j*params.nelem] += rhs[k*params.nvar+j];

            u[k*params.columnL+i_elem+j*params.nelem] = old_u[k*params.columnL+i_elem+j*params.nelem];
            for ( int l = 0; l < i_RK+1; l++ ) //from 0 to the current rk stage n
            {
              u[k*params.columnL+i_elem+j*params.nelem]
                += params.dt*butcher_a[i_RK*params.nRK+l]
                  *ks[l][k*params.columnL+i_elem+j*params.nelem];
            }

          }
        }

      }//for each element local implicit update
      ress = log10(sqrt(res[0]+res[1]+res[2]+res[3]));
}


      conservation = 0;
      for ( int j = 0; j < params.nelem; j++ )
      {
        for ( int k = 0; k < params.porder+1; k++ )
        {
          for ( int l = 0; l < params.porder+1; l++ )
          {
            //conservation += u[(k*(params.porder+1)+l)*params.columnL+j]
            //               *weights[k]*weights[l]*params.jacob;
            conservation += ( old_u[(k*(params.porder+1)+l)*params.columnL+j]
                            + params.dt*ks[i_RK][(k*(params.porder+1)+l)*params.columnL+j])
                           *weights[k]*weights[l]*params.jacob;
          }
        }
      }



      for ( int j = 0; j < params.nvar; j++ )
        std::cout << std::fixed << std::setprecision(3) << " " << std::setw(6) << log10(sqrt(res[j]));
      std::cout << " rT=" << std::setw(6) << ress
                << "  conservation:" << std::scientific << std::setw(10)
                << conservation-CONSERVATION << "\n";
                // << " " << std::fixed << std::setprecision(3) <<  log10(sqrt(resM)) << "\n";



/*
std::cout << "\n";
std::cout << old_u[0*params.columnL+1*params.nelem+0]/old_u[0*params.columnL+0*params.nelem+0];
std::cout << "\n";
std::cout << old_u[0*params.columnL+2*params.nelem+0]/old_u[0*params.columnL+0*params.nelem+0];

      double eps = 0.000001;
      old_u[0*params.columnL+0*params.nelem+0] += eps;
      superFunc(&params, old_u, f, g, u_face, f_face, lagrInterL, lagrInterR);
      computeFlux(&params, u_face, f_face, u_hat_face);
      update(&params, old_u, f, g, f_face, lagrDerivs, hL, hR);
      std::cout << "\nJACOBIAN\n";
      for ( int j = 0; j < params.nse; j++ )
      {
        for ( int k = 0; k < params.nvar; k++ )
        {
          std::cout << std::scientific << (-old_u[j*params.columnL+k*params.nelem+0]
                       + u[j*params.columnL+k*params.nelem+0] )/eps << "\n";
        }
        std::cout << "\n";
      }
*/

      // NaN CHECK
      double errr=0.0;
      for ( int j = 0; j < params.nse*params.columnL; j++ ) errr += u[j];
      if ( errr != errr ) 
      {
        std::cout << "\nthere is a nan somewhere\n";
        for ( int j = 0; j < params.nse*params.columnL; j++ ) u[j] = u_curr[j];
        goto jumptowrite;
      }


/*
      fface << ite << "\n";
      for ( int j = 0; j < params.nfe*params.columnL; j++ )
      {
        fface << f_face[j] << " "; if ( (j+1)%params.columnL == 0 ) fface << "\n";
      }
      fface << "\n";

      for ( int j = 0; j < params.nse*params.columnL; j++ )
      {
        uvals << u[j] << " "; if ( (j+1)%params.columnL == 0 ) uvals << "\n";
      }
      uvals << "\n";
*/
} // sub_ite

    } // R-K loop


    // DIRK
    
    // butcher table implementation
    // update solution here using all ks stages;
    //explicit update
    for ( int j = 0; j < params.nse*params.columnL; j++ )
    {
      u[j] = old_u[j];
      for ( int k = 0; k < params.nRK; k++ )
      {
        u[j] += butcher_b[k]*ks[k][j]*params.dt;
      }
    }


    if ( ite%params.writeOut == 0 ) writeSolution(&params, u, ite);
  } // time iteration

  jumptowrite: writeSolution(&params, u, params.maxIte);


/*
  //error for gaussian bump
  double error = 0;
  double x, sigma=0.2, pi=3.141592653589793, mu=0;
  std::cout << "Gaussian bump with sigma = " << sigma << " and mu = " << mu << "\n";
  for ( int i = 0; i < params.nelem; i++ )
  {
    //x = i*L/params.nelem-5;
    for ( int j = 0; j < params.porder+1; j++ )
    {
    x = i*L/params.nelem-5 + soln_coords[j]*params.jacob;
    //u[j*params.columnL + i] = 1/(sigma*sqrt(2*pi))*exp(-0.5*pow((x-mu)/sigma,2));
    error += pow(u[j*params.columnL + i]-1/(sigma*sqrt(2*pi))*exp(-0.5*pow((x-mu)/sigma,2)) , 2);
    }
  }

  std::cout << "error " << log10(sqrt(error)/sqrt(params.nelem*(params.porder+1))) << "\n";
  std::cout << log10(L/params.nelem) << "\n";
*/

/*
  jumptowrite: std::cout << "\njumped to write the last state\n";
  std::ofstream solution;
  solution.open("solution.csv");
  double x, y, x_L =-10, x_R = 10, y_B =-10, y_T = 10;
  double dx = (x_R-x_L)/params.nelem_x, dy = (y_T-y_B)/params.nelem_y;
  double avg_u[4];
  int i_elem;
  for ( int i = 0; i < params.nelem_x; i++ )
  {
    for ( int j = 0; j < params.nelem_y; j++ )
    {
      i_elem = j*params.nelem_x+i;
      x = x_L + i*dx + dx/2.0;
      y = y_B + j*dy + dy/2.0;
      for ( int k = 0; k < params.nvar; k++ ) avg_u[k] = 0;
      for ( int k = 0; k < params.nse; k++ )
      {
        for ( int l = 0; l < params.nvar; l++ )
        {
          avg_u[l] += u[k*params.columnL+i_elem+l*params.nelem];
        }
      }
      solution << x << ", " << y;
      for ( int k = 0; k < params.nvar; k++ ) solution << ", "  << avg_u[k]/params.nse;
      solution << "\n";
    }
  }
*/

  double errr=0.0;
  for ( int i = 0; i < params.columnL*params.nse; i++ ) errr += u[i];
  if ( errr != errr ) std::cout << "there is a nan somewhere\n";

  return 0;
}




// NEW SET OF FUNCTIONS COMPUTABLE WITH NAVIER-STOKES;

void deriveGiven( essential *params, 
                  double *pd_X_ksi, double *pd_Y_eta, double *X, double *Y, double *fixed_f,
                  double *lagrDerivs, double *hL, double *hR )
{

  double *dummy_x = new double[params->nse];
  double *dummy_y = new double[params->nse];

  int ji_row, indx_L, indx_R, indx_B, indx_T;

  for ( int i_elem = 0; i_elem < params->columnL; i_elem++ )
  {
    //rowsT
    for ( int j = 0; j < params->nse; j++ ) {dummy_x[j] = 0; dummy_y[j] = 0;}
    for ( int j = 0; j < params->porder+1; j++ )
    {
      indx_L = (4*(params->porder+1)-1-j)*params->columnL+i_elem;
      indx_R = (params->porder+1+j)*params->columnL+i_elem;
      for ( int i = 0; i < params->porder+1; i++ )
      {
        ji_row = j*(params->porder+1)+i;
        indx_B = i*params->columnL+i_elem;
        indx_T = (3*(params->porder+1)-1-i)*params->columnL+i_elem;
        for ( int k = 0; k < params->porder+1; k++ )
        {
          //jk_row = j*(params->porder+1);
          dummy_x[ji_row] += X[(j*(params->porder+1)+k)*params->columnL+i_elem]
                          *lagrDerivs[i*(params->porder+1)+k];
          dummy_y[ji_row] += Y[(k*(params->porder+1)+i)*params->columnL+i_elem]
                          *lagrDerivs[j*(params->porder+1)+k];
        }
        dummy_x[ji_row] += fixed_f[indx_L]*hL[i] + fixed_f[indx_R]*hR[i];
        dummy_y[ji_row] += fixed_f[indx_B]*hL[j] + fixed_f[indx_T]*hR[j];
      }
    }
    for ( int j = 0; j < params->nse; j++ )
    {
      pd_X_ksi[j*params->columnL+i_elem] = 0;
      pd_Y_eta[j*params->columnL+i_elem] = 0;
    }
    for ( int j = 0; j < params->nse; j++ )
    {
      pd_X_ksi[j*params->columnL+i_elem] += dummy_x[j]/params->jacob;
      pd_Y_eta[j*params->columnL+i_elem] += dummy_y[j]/params->jacob;
    }
  }

  delete[] dummy_x, dummy_y;

  return;
}


void extrapToFace( essential *params,
                   double *X, double *Y, double *face,
                   double *lagrInterL, double *lagrInterR )
{

  int *loc_q = new int[params->nvar];

  int indx_L, indx_R, indx_B, indx_T, indx_elem;

  for ( int i_elem = 0; i_elem < params->nelem; i_elem++ )
  {
    for ( int j = 0; j < params->nvar; j++ ) loc_q[j] = i_elem+j*params->nelem; //column pos
    for ( int j = 0; j < params->nvar; j++ )
    {
      for ( int k = 0; k < params->nfe; k++ )
      {
        face[params->columnL*k + loc_q[j]] = 0;
      }
    }
    for ( int j = 0; j < params->porder+1; j++ )
    {
      for ( int k = 0; k < params->nvar; k++ )
      {
        indx_L = (4*(params->porder+1)-1-j)*params->columnL+loc_q[k];
        indx_R = (params->porder+1+j)*params->columnL+loc_q[k];
        for ( int i = 0; i < params->porder+1; i++ )
        {
          indx_elem = (j*(params->porder+1)+i)*params->columnL+loc_q[k];
          indx_B = i*params->columnL+loc_q[k];
          indx_T = (3*(params->porder+1)-1-i)*params->columnL+loc_q[k];
          face[indx_L] += lagrInterL[i]*X[indx_elem];
          face[indx_R] += lagrInterR[i]*X[indx_elem];
          face[indx_B] += lagrInterL[j]*Y[indx_elem];
          face[indx_T] += lagrInterR[j]*Y[indx_elem];
        }
      }

    }
  }

  delete[] loc_q;
  return;
}



void getInterFlux( essential* params,
                   double *f_face, double *u_face, double *q_x_face, double *q_y_face, double *u_hat_face )
{

  //int loc_q[nvar];
  //double u_L[nvar], u_R[nvar], f_I[nvar];
  int *loc_q = new int[params->nvar];
  double *u_L = new double[params->nvar];
  double *u_R = new double[params->nvar];
  double *f_I = new double[params->nvar];
  double *fv_I = new double[params->nvar];
  double *u_hat = new double[params->nvar];

  double *fv_L = new double[params->nvar];
  double *gv_L = new double[params->nvar];
  double *fv_R = new double[params->nvar];
  double *gv_R = new double[params->nvar];
  double *pd_U_x_L = new double[params->nvar];
  double *pd_U_y_L = new double[params->nvar];
  double *pd_U_x_R = new double[params->nvar];
  double *pd_U_y_R = new double[params->nvar];

  int *pairL = new int[params->porder+1];
  int *pairR = new int[params->porder+1];

  int next_elem, indx_L, indx_R;
  //double n_x, n_y;
  //roe_flux(u_R, u_L, n_x, n_y, f_I);

  // get all the interface fluxes at the interior faces
  for ( int i_x = 0; i_x < params->nelem_x; i_x++ )
  {
    for ( int i_y = 0; i_y < params->nelem_y; i_y++ )
    {
      int i_elem = i_y*params->nelem_x+i_x;
      for ( int i = 0; i < params->nvar; i++ ) loc_q[i] = i_elem+i*params->nelem;
      //------------------------------------------------------------------------
      // left and right
      if ( i_x != params->nelem_x-1 ) // interior zone
      {
        next_elem = 1;
        for ( int i = 0; i < params->porder+1; i++ ) pairL[i] = params->porder+1+i;
        for ( int i = 0; i < params->porder+1; i++ ) pairR[i] = 4*(params->porder+1)-1-i;
        for ( int i = 0; i < params->porder+1; i++ )
        {
          for ( int j = 0; j < params->nvar; j++ )
          {
            u_L[j] = u_face[pairL[i]*params->columnL + loc_q[j]]; // right of the present element
            u_R[j] = u_face[pairR[i]*params->columnL + loc_q[j]+next_elem]; //left of the next element
            //f_I[j] = u_L[j];//linear advection

            pd_U_x_L[j] = q_x_face[pairL[i]*params->columnL + loc_q[j]];
            pd_U_x_R[j] = q_x_face[pairR[i]*params->columnL + loc_q[j]+next_elem];
            pd_U_y_L[j] = q_y_face[pairL[i]*params->columnL + loc_q[j]];
            pd_U_y_R[j] = q_y_face[pairR[i]*params->columnL + loc_q[j]+next_elem];
          }
          roe_flux(u_L, u_R, 1.0, 0.0, f_I, u_hat);
          //left side
          nodeViscFlux(u_L, pd_U_x_L, pd_U_y_L, params->mu, params->Pr, fv_L, gv_L); // use only fv_L
          //right side
          nodeViscFlux(u_R, pd_U_x_R, pd_U_y_R, params->mu, params->Pr, fv_R, gv_R); // use only fv_R

          for ( int j = 0; j < params->nvar; j++ )
            fv_I[j] = 0.5*(fv_L[j]+fv_R[j]) + params->tau*(u_L[j]-u_R[j]) + params->beta*(fv_L[j]-fv_R[j]);

          //overwrite f_face
          for ( int j = 0; j < params->nvar; j++ )
          {
            indx_L = pairL[i]*params->columnL + loc_q[j];
            indx_R = pairR[i]*params->columnL + loc_q[j]+next_elem;
            f_face[indx_L] = f_I[j] - fv_I[j] - f_face[indx_L];
            f_face[indx_R] = f_I[j] - fv_I[j] - f_face[indx_R];
            u_hat_face[indx_L] = u_hat[j];
            u_hat_face[indx_R] =-u_hat[j];
          }
        }
      }
      else // right wall
      {
        // leave this to BC
        next_elem = 1-params->nelem_x; //if at the end
      }
      //------------------------------------------------------------------------
      // bottom and top
      if ( i_y != params->nelem_y-1 ) // interior zone
      {
        next_elem = params->nelem_x;
        for ( int i = 0; i < params->porder+1; i++ ) pairL[i] = 3*(params->porder+1)-1-i;
        for ( int i = 0; i < params->porder+1; i++ ) pairR[i] = i;
        for ( int i = 0; i < params->porder+1; i++ )
        {
          for ( int j = 0; j < params->nvar; j++ )
          {
            u_L[j] = u_face[pairL[i]*params->columnL + loc_q[j]]; // top of the present element
            u_R[j] = u_face[pairR[i]*params->columnL + loc_q[j]+next_elem]; //bottom of the next element
            //f_I[j] = u_L[j];//linear advection

            pd_U_x_L[j] = q_x_face[pairL[i]*params->columnL + loc_q[j]];
            pd_U_x_R[j] = q_x_face[pairR[i]*params->columnL + loc_q[j]+next_elem];
            pd_U_y_L[j] = q_y_face[pairL[i]*params->columnL + loc_q[j]];
            pd_U_y_R[j] = q_y_face[pairR[i]*params->columnL + loc_q[j]+next_elem];
          }
          roe_flux(u_L, u_R, 0.0, 1.0, f_I, u_hat);
          //left side
          nodeViscFlux(u_L, pd_U_x_L, pd_U_y_L, params->mu, params->Pr, fv_L, gv_L); // use only gv_L
          //right side
          nodeViscFlux(u_R, pd_U_x_R, pd_U_y_R, params->mu, params->Pr, fv_R, gv_R); // use only gv_R

          for ( int j = 0; j < params->nvar; j++ )
            fv_I[j] = 0.5*(gv_L[j]+gv_R[j]) + params->tau*(u_L[j]-u_R[j]) + params->beta*(gv_L[j]-gv_R[j]);

          //overwrite f_face
          for ( int j = 0; j < params->nvar; j++ )
          {
            indx_L = pairL[i]*params->columnL + loc_q[j];
            indx_R = pairR[i]*params->columnL + loc_q[j]+next_elem;
            f_face[indx_L] = f_I[j] - fv_I[j] - f_face[indx_L];
            f_face[indx_R] = f_I[j] - fv_I[j] - f_face[indx_R];
            u_hat_face[indx_L] = u_hat[j];
            u_hat_face[indx_R] =-u_hat[j];
          }
        }
      }
      else // top wall
      {
        // leave this to BC
        next_elem = (1-params->nelem_y)*params->nelem_x; //if at the end
      }
    }
  }


  // BC time
  // left wall

  // right wall

if ( true ) // periodic left and right
{
  int i_x = params->nelem_x-1; // right wall
  for ( int i_y = 0; i_y < params->nelem_y; i_y++ )
  {
    int i_elem = i_y*params->nelem_x+i_x;
    for ( int i = 0; i < params->nvar; i++ ) loc_q[i] = i_elem+i*params->nelem;
    next_elem = 1-params->nelem_x;
    for ( int i = 0; i < params->porder+1; i++ ) pairL[i] = params->porder+1+i;
    for ( int i = 0; i < params->porder+1; i++ ) pairR[i] = 4*(params->porder+1)-1-i;
    for ( int i = 0; i < params->porder+1; i++ )
    {
      for ( int j = 0; j < params->nvar; j++ )
      {
        u_L[j] = u_face[pairL[i]*params->columnL + loc_q[j]]; // right of the present element
        u_R[j] = u_face[pairR[i]*params->columnL + loc_q[j]+next_elem]; //left of the next element
        //f_I[j] = u_L[j];//linear advection

        pd_U_x_L[j] = q_x_face[pairL[i]*params->columnL + loc_q[j]];
        pd_U_x_R[j] = q_x_face[pairR[i]*params->columnL + loc_q[j]+next_elem];
        pd_U_y_L[j] = q_y_face[pairL[i]*params->columnL + loc_q[j]];
        pd_U_y_R[j] = q_y_face[pairR[i]*params->columnL + loc_q[j]+next_elem];
      }
      roe_flux(u_L, u_R, 1.0, 0.0, f_I, u_hat);
      //left side
      nodeViscFlux(u_L, pd_U_x_L, pd_U_y_L, params->mu, params->Pr, fv_L, gv_L); // use only fv_L
      //right side
      nodeViscFlux(u_R, pd_U_x_R, pd_U_y_R, params->mu, params->Pr, fv_R, gv_R); // use only fv_R

      for ( int j = 0; j < params->nvar; j++ )
        fv_I[j] = 0.5*(fv_L[j]+fv_R[j]) + params->tau*(u_L[j]-u_R[j]) + params->beta*(fv_L[j]-fv_R[j]);

      //overwrite f_face
      for ( int j = 0; j < params->nvar; j++ )
      {
        indx_L = pairL[i]*params->columnL + loc_q[j];
        indx_R = pairR[i]*params->columnL + loc_q[j]+next_elem;
        f_face[indx_L] = f_I[j] - fv_I[j] - f_face[indx_L];
        f_face[indx_R] = f_I[j] - fv_I[j] - f_face[indx_R];
        u_hat_face[indx_L] = u_hat[j];
        u_hat_face[indx_R] =-u_hat[j];
      }
    }
  }
}

if ( true ) // periodic top and bottom
{
  for ( int i_x = 0; i_x < params->nelem_x; i_x++ )
  {
    int i_y = params->nelem_y-1; //top wall
    int i_elem = i_y*params->nelem_x+i_x;
    for ( int i = 0; i < params->nvar; i++ ) loc_q[i] = i_elem+i*params->nelem;
    next_elem = (1-params->nelem_y)*params->nelem_x;
    for ( int i = 0; i < params->porder+1; i++ ) pairL[i] = 3*(params->porder+1)-1-i;
    for ( int i = 0; i < params->porder+1; i++ ) pairR[i] = i;
    for ( int i = 0; i < params->porder+1; i++ )
    {
      for ( int j = 0; j < params->nvar; j++ )
      {
        u_L[j] = u_face[pairL[i]*params->columnL + loc_q[j]]; // top of the present element
        u_R[j] = u_face[pairR[i]*params->columnL + loc_q[j]+next_elem]; //bottom of the next element
        //f_I[j] = u_L[j];//linear advection

        pd_U_x_L[j] = q_x_face[pairL[i]*params->columnL + loc_q[j]];
        pd_U_x_R[j] = q_x_face[pairR[i]*params->columnL + loc_q[j]+next_elem];
        pd_U_y_L[j] = q_y_face[pairL[i]*params->columnL + loc_q[j]];
        pd_U_y_R[j] = q_y_face[pairR[i]*params->columnL + loc_q[j]+next_elem];
      }
      roe_flux(u_L, u_R, 0.0, 1.0, f_I, u_hat);
      //left side
      nodeViscFlux(u_L, pd_U_x_L, pd_U_y_L, params->mu, params->Pr, fv_L, gv_L); // use only gv_L
      //right side
      nodeViscFlux(u_R, pd_U_x_R, pd_U_y_R, params->mu, params->Pr, fv_R, gv_R); // use only gv_R

      for ( int j = 0; j < params->nvar; j++ )
        fv_I[j] = 0.5*(gv_L[j]+gv_R[j]) + params->tau*(u_L[j]-u_R[j]) + params->beta*(gv_L[j]-gv_R[j]);

      //overwrite f_face
      for ( int j = 0; j < params->nvar; j++ )
      {
        indx_L = pairL[i]*params->columnL + loc_q[j];
        indx_R = pairR[i]*params->columnL + loc_q[j]+next_elem;
        f_face[indx_L] = f_I[j] - fv_I[j] - f_face[indx_L];
        f_face[indx_R] = f_I[j] - fv_I[j] - f_face[indx_R];
        u_hat_face[indx_L] = u_hat[j];
        u_hat_face[indx_R] =-u_hat[j];
      }
    }
  }
}


if ( false ) { //couette
  for ( int i_x = 0; i_x < params->nelem_x; i_x++ )
  {
    // bottom wall // no slip with u_w = 0, v_w = 0
    int i_y = 0; //bottom wall
    int i_elem = i_y*params->nelem_x+i_x;
    for ( int i = 0; i < params->nvar; i++ ) loc_q[i] = i_elem+i*params->nelem;
    for ( int i = 0; i < params->porder+1; i++ ) pairR[i] = i;
    for ( int i = 0; i < params->porder+1; i++ )
    {
      for ( int j = 0; j < params->nvar; j++ )
      {
        u_R[j] = u_face[pairR[i]*params->columnL + loc_q[j]]; //bottom of the present element
        pd_U_x_R[j] = q_x_face[pairR[i]*params->columnL + loc_q[j]];
        pd_U_x_L[j] = pd_U_x_R[j];
        pd_U_y_R[j] = q_y_face[pairR[i]*params->columnL + loc_q[j]];
        pd_U_y_L[j] = pd_U_y_R[j];
      }
      // set u_L for inviscid BC
      double u_w = 0.0;
      double v_w = 0.0;
      double RT = 86142.8571429;//1.0;
      u_L[0] = u_R[0];
      u_L[1] = u_R[0]*2.0*u_w - u_R[1];
      u_L[2] = u_R[0]*2.0*v_w - u_R[2];
      u_L[3] = u_R[0]*( RT/(gammaVal-1.0) + ( pow(2.0*u_w-u_R[1]/u_R[0],2)
                                            + pow(2.0*v_w-u_R[2]/u_R[0],2) )*0.5 );
      roe_flux(u_L, u_R, 0.0, 1.0, f_I, u_hat);

      // set u_L for viscous BC
      u_L[0] = u_R[0];
      u_L[1] = u_R[0]*u_w;
      u_L[2] = u_R[0]*v_w;
      u_L[3] = u_R[0]*(RT/(gammaVal-1.0) + (u_w*u_w + v_w*v_w)*0.5);
      //left side
      nodeViscFlux(u_L, pd_U_x_L, pd_U_y_L, params->mu, params->Pr, fv_L, gv_L); // use only gv_L
      //right side
      nodeViscFlux(u_R, pd_U_x_R, pd_U_y_R, params->mu, params->Pr, fv_R, gv_R); // use only gv_R

      for ( int j = 0; j < params->nvar; j++ )
        fv_I[j] = 0.5*(gv_L[j]+gv_R[j]) + params->tau*(u_L[j]-u_R[j]) + params->beta*(gv_L[j]-gv_R[j]);

      //overwrite f_face
      for ( int j = 0; j < params->nvar; j++ )
      {
        indx_R = pairR[i]*params->columnL + loc_q[j];
        f_face[indx_R] = f_I[j] - fv_I[j] - f_face[indx_R];
        u_hat_face[indx_R] =-u_hat[j];
      }
    }

    // top wall // no slip with u_w = 1, v_w = 0
    i_y = params->nelem_y-1; //top wall
    i_elem = i_y*params->nelem_x+i_x;
    for ( int i = 0; i < params->nvar; i++ ) loc_q[i] = i_elem+i*params->nelem;
    for ( int i = 0; i < params->porder+1; i++ ) pairL[i] = 3*(params->porder+1)-1-i;
    for ( int i = 0; i < params->porder+1; i++ )
    {
      for ( int j = 0; j < params->nvar; j++ )
      {
        u_L[j] = u_face[pairL[i]*params->columnL + loc_q[j]]; // top of the present element
        pd_U_x_L[j] = q_x_face[pairL[i]*params->columnL + loc_q[j]];
        pd_U_x_R[j] = pd_U_x_L[j];
        pd_U_y_L[j] = q_y_face[pairL[i]*params->columnL + loc_q[j]];
        pd_U_y_R[j] = pd_U_y_L[j];
      }
      // set u_R for inviscid BC
      double u_w = 69.445;//1.0;
      double v_w = 0.0;
      double RT = 86142.8571429;//1.0;
      u_R[0] = u_L[0];
      u_R[1] = u_L[0]*2.0*u_w - u_L[1];
      u_R[2] = u_L[0]*2.0*v_w - u_L[2];
      u_R[3] = u_L[0]*( RT/(gammaVal-1.0) + ( pow(2.0*u_w-u_L[1]/u_L[0],2)
                                            + pow(2.0*v_w-u_L[2]/u_L[0],2) )*0.5 );
      roe_flux(u_L, u_R, 0.0, 1.0, f_I, u_hat);

      // set u_R for viscous BC
      u_R[0] = u_L[0];
      u_R[1] = u_L[0]*u_w;
      u_R[2] = u_L[0]*v_w;
      u_R[3] = u_L[0]*(RT/(gammaVal-1.0) + (u_w*u_w + v_w*v_w)*0.5);
      //left side
      nodeViscFlux(u_L, pd_U_x_L, pd_U_y_L, params->mu, params->Pr, fv_L, gv_L); // use only gv_L
      //right side
      nodeViscFlux(u_R, pd_U_x_R, pd_U_y_R, params->mu, params->Pr, fv_R, gv_R); // use only gv_R

      for ( int j = 0; j < params->nvar; j++ )
        fv_I[j] = 0.5*(gv_L[j]+gv_R[j]) + params->tau*(u_L[j]-u_R[j]) + params->beta*(gv_L[j]-gv_R[j]);

      //overwrite f_face
      for ( int j = 0; j < params->nvar; j++ )
      {
        indx_L = pairL[i]*params->columnL + loc_q[j];
        f_face[indx_L] = f_I[j] - fv_I[j] - f_face[indx_L];
        u_hat_face[indx_L] = u_hat[j];
      }
    }
  }
} // if couette


  delete[] loc_q, u_L, u_R, f_I, u_hat, pairL, pairR;
  delete[] fv_L, gv_L, fv_L, gv_L, pd_U_x_L, pd_U_y_L, pd_U_x_R, pd_U_y_R;
  return;
}


void getFlux( essential* params, 
              double *u, double *q_x, double *q_y, double *f, double *g )
{
  // compute flux on the solution nodes

  //int loc_q[nvar];
  int *loc_q = new int[params->nvar];

  double *fv_node = new double[params->nvar];
  double *gv_node = new double[params->nvar];
  double *U_vals = new double[params->nvar];
  double *pd_U_x = new double[params->nvar];
  double *pd_U_y = new double[params->nvar];
  //double vel_u, vel_v, p;

  int indx_L, indx_R, indx_B, indx_T, indx_elem;

  for ( int i_elem = 0; i_elem < params->nelem; i_elem++ )
  {
    for ( int j = 0; j < params->nvar; j++ ) loc_q[j] = i_elem+j*params->nelem; //column pos
    //rows
    for ( int j = 0; j < params->porder+1; j++ )
    {
      for ( int i = 0; i < params->porder+1; i++ )
      {
        //index of the first data in the row; +loc_q[k] gives the correct pos
        indx_elem = (j*(params->porder+1)+i)*params->columnL;

        f[indx_elem+loc_q[0]] = u[indx_elem+loc_q[1]];
        g[indx_elem+loc_q[0]] = u[indx_elem+loc_q[2]];
        double rho = u[indx_elem+loc_q[0]];
        double vel_u = u[indx_elem+loc_q[1]]/rho;
        double vel_v = u[indx_elem+loc_q[2]]/rho;
        double E = u[indx_elem+loc_q[3]];
        double p = (gammaVal-1)*(E-0.5*rho*(vel_u*vel_u+vel_v*vel_v));
        f[indx_elem+loc_q[1]] = rho*vel_u*vel_u + p;
        g[indx_elem+loc_q[1]] = rho*vel_u*vel_v;
        f[indx_elem+loc_q[2]] = rho*vel_u*vel_v;//rho*vel_v*vel_u;
        g[indx_elem+loc_q[2]] = rho*vel_v*vel_v + p;
        f[indx_elem+loc_q[3]] = (E+p)*vel_u;
        g[indx_elem+loc_q[3]] = (E+p)*vel_v;

        for ( int k = 0; k < params->nvar; k++ )
        {
          U_vals[k] = u[indx_elem+loc_q[k]];
          pd_U_x[k] = q_x[indx_elem+loc_q[k]];
          pd_U_y[k] = q_y[indx_elem+loc_q[k]];
        }
        nodeViscFlux(U_vals, pd_U_x, pd_U_y, params->mu, params->Pr, fv_node, gv_node);
        for ( int k = 0; k < params->nvar; k++ )
        {
          f[indx_elem+loc_q[k]] -= fv_node[k];
          g[indx_elem+loc_q[k]] -= gv_node[k];
        }

        /*
        for ( int k = 0; k < params->nvar; k++ )
        {
          f[indx_elem+loc_q[k]] = u[indx_elem+loc_q[k]];
          g[indx_elem+loc_q[k]] = u[indx_elem+loc_q[k]];
        }
        */
      }
    }
  }

  delete[] loc_q, fv_node, gv_node, U_vals, pd_U_x, pd_U_y;
  return;
}

void getCommonU(essential *params, double *u_face)
{

  //int loc_q[nvar];
  //double u_L[nvar], u_R[nvar], f_I[nvar];
  int *loc_q = new int[params->nvar];
  double *u_L = new double[params->nvar];
  double *u_R = new double[params->nvar];
  double *u_I = new double[params->nvar];

  int *pairL = new int[params->porder+1];
  int *pairR = new int[params->porder+1];

  int next_elem, indx_L, indx_R;
  //double n_x, n_y;
  //roe_flux(u_R, u_L, n_x, n_y, f_I);

  for ( int i_x = 0; i_x < params->nelem_x; i_x++ )
  {
    for ( int i_y = 0; i_y < params->nelem_y; i_y++ )
    {
      int i_elem = i_y*params->nelem_x+i_x;
      for ( int i = 0; i < params->nvar; i++ ) loc_q[i] = i_elem+i*params->nelem;
      // left and right
      if ( i_x != params->nelem_x-1 )
      {
        next_elem = 1;
        for ( int i = 0; i < params->porder+1; i++ ) pairL[i] = params->porder+1+i;
        for ( int i = 0; i < params->porder+1; i++ ) pairR[i] = 4*(params->porder+1)-1-i;
        for ( int i = 0; i < params->porder+1; i++ )
        {
          for ( int j = 0; j < params->nvar; j++ )
          {
            u_L[j] = u_face[pairL[i]*params->columnL + loc_q[j]]; // right of the present element
            u_R[j] = u_face[pairR[i]*params->columnL + loc_q[j]+next_elem]; //left of the next element
            u_I[j] = (u_L[j] + u_R[j])*0.5 - params->beta*(u_L[j]-u_R[j]);
          }
          //overwrite f_face
          for ( int j = 0; j < params->nvar; j++ )
          {
            indx_L = pairL[i]*params->columnL + loc_q[j];
            indx_R = pairR[i]*params->columnL + loc_q[j]+next_elem;
            u_face[indx_L] = u_I[j] - u_face[indx_L];
            u_face[indx_R] = u_I[j] - u_face[indx_R];
          }
        }
      }
      else // 
      {
        // leave this to BC
        next_elem = 1-params->nelem_x; //if at the end
      }
      // bottom and top
      if ( i_y != params->nelem_y-1 )
      {
        next_elem = params->nelem_x;
        for ( int i = 0; i < params->porder+1; i++ ) pairL[i] = 3*(params->porder+1)-1-i;
        for ( int i = 0; i < params->porder+1; i++ ) pairR[i] = i;
        for ( int i = 0; i < params->porder+1; i++ )
        {
          for ( int j = 0; j < params->nvar; j++ )
          {
            u_L[j] = u_face[pairL[i]*params->columnL + loc_q[j]]; // top of the present element
            u_R[j] = u_face[pairR[i]*params->columnL + loc_q[j]+next_elem]; //bottom of the next element
            u_I[j] = (u_L[j] + u_R[j])*0.5 - params->beta*(u_L[j]-u_R[j]);
          }
          //overwrite f_face
          for ( int j = 0; j < params->nvar; j++ )
          {
            indx_L = pairL[i]*params->columnL + loc_q[j];
            indx_R = pairR[i]*params->columnL + loc_q[j]+next_elem;
            u_face[indx_L] = u_I[j] - u_face[indx_L];
            u_face[indx_R] = u_I[j] - u_face[indx_R];
          }
        }
      }
      else // leave this to BC
      {
        next_elem = (1-params->nelem_y)*params->nelem_x; //if at the end
      }
    }
  }

if ( true ) {
  // periodic left and right
  int i_x = params->nelem_x-1;
  for ( int i_y = 0; i_y < params->nelem_y; i_y++ )
  {
    int i_elem = i_y*params->nelem_x+i_x;
    for ( int i = 0; i < params->nvar; i++ ) loc_q[i] = i_elem+i*params->nelem;
    // left and right
    next_elem = 1-params->nelem_x; //if at the end
    for ( int i = 0; i < params->porder+1; i++ ) pairL[i] = params->porder+1+i;
    for ( int i = 0; i < params->porder+1; i++ ) pairR[i] = 4*(params->porder+1)-1-i;
    for ( int i = 0; i < params->porder+1; i++ )
    {
      for ( int j = 0; j < params->nvar; j++ )
      {
        u_L[j] = u_face[pairL[i]*params->columnL + loc_q[j]]; // right of the present element
        u_R[j] = u_face[pairR[i]*params->columnL + loc_q[j]+next_elem]; //left of the next element
        u_I[j] = (u_L[j] + u_R[j])*0.5 - params->beta*(u_L[j]-u_R[j]);
      }
      //overwrite f_face
      for ( int j = 0; j < params->nvar; j++ )
      {
        indx_L = pairL[i]*params->columnL + loc_q[j];
        indx_R = pairR[i]*params->columnL + loc_q[j]+next_elem;
        u_face[indx_L] = u_I[j] - u_face[indx_L];
        u_face[indx_R] = u_I[j] - u_face[indx_R];
      }
    }
  }
}

if ( true ) {
  // periodic bottom and top
  for ( int i_x = 0; i_x < params->nelem_x; i_x++ )
  {
    int i_y = params->nelem_y-1;
    int i_elem = i_y*params->nelem_x+i_x;
    for ( int i = 0; i < params->nvar; i++ ) loc_q[i] = i_elem+i*params->nelem;
    // bottom and top
    next_elem = (1-params->nelem_y)*params->nelem_x; //if at the end
    for ( int i = 0; i < params->porder+1; i++ ) pairL[i] = 3*(params->porder+1)-1-i;
    for ( int i = 0; i < params->porder+1; i++ ) pairR[i] = i;
    for ( int i = 0; i < params->porder+1; i++ )
    {
      for ( int j = 0; j < params->nvar; j++ )
      {
        u_L[j] = u_face[pairL[i]*params->columnL + loc_q[j]]; // top of the present element
        u_R[j] = u_face[pairR[i]*params->columnL + loc_q[j]+next_elem]; //bottom of the next element
        u_I[j] = (u_L[j] + u_R[j])*0.5 - params->beta*(u_L[j]-u_R[j]);
      }
      //overwrite f_face
      for ( int j = 0; j < params->nvar; j++ )
      {
        indx_L = pairL[i]*params->columnL + loc_q[j];
        indx_R = pairR[i]*params->columnL + loc_q[j]+next_elem;
        u_face[indx_L] = u_I[j] - u_face[indx_L];
        u_face[indx_R] = u_I[j] - u_face[indx_R];
      }
    }
  }
}

if ( false ) { //couette
  for ( int i_x = 0; i_x < params->nelem_x; i_x++ )
  {
    // bottom wall
    int i_y = 0;
    int i_elem = i_y*params->nelem_x+i_x;
    for ( int i = 0; i < params->nvar; i++ ) loc_q[i] = i_elem+i*params->nelem;
    for ( int i = 0; i < params->porder+1; i++ ) pairR[i] = i;
    for ( int i = 0; i < params->porder+1; i++ )
    {
      for ( int j = 0; j < params->nvar; j++ )
      {
        u_R[j] = u_face[pairR[i]*params->columnL + loc_q[j]]; //bottom of the present element
      }

      // set u_L with ldg no slip wall, u_w=0, v_w=0;
      double u_w = 0.0;
      double v_w = 0.0;
      double RT = 86142.8571429;//1.0;
      u_L[0] = u_R[0];
      u_L[1] = u_R[0]*u_w;
      u_L[2] = u_R[0]*v_w;
      u_L[3] = u_R[0]*(RT/(gammaVal-1.0) + (u_w*u_w + v_w*v_w)*0.5);

      for ( int j = 0; j < params->nvar; j++ )
      {
        u_I[j] = (u_L[j] + u_R[j])*0.5 - params->beta*(u_L[j]-u_R[j]);
      }
      //overwrite f_face
      for ( int j = 0; j < params->nvar; j++ )
      {
        indx_R = pairR[i]*params->columnL + loc_q[j];
        u_face[indx_R] = u_I[j] - u_face[indx_R];
      }
    }

    // top wall
    i_y = params->nelem_y-1;
    i_elem = i_y*params->nelem_x+i_x;
    for ( int i = 0; i < params->nvar; i++ ) loc_q[i] = i_elem+i*params->nelem;
    for ( int i = 0; i < params->porder+1; i++ ) pairL[i] = 3*(params->porder+1)-1-i;
    for ( int i = 0; i < params->porder+1; i++ )
    {
      for ( int j = 0; j < params->nvar; j++ )
      {
        u_L[j] = u_face[pairL[i]*params->columnL + loc_q[j]]; // top of the present element
      }
      // set u_R wrt ldg no slip wall, u_w=1, v_w=0;
      double u_w = 69.445;//1.0;
      double v_w = 0.0;
      double RT = 86142.8571429;//1.0;
      u_R[0] = u_L[0];
      u_R[1] = u_L[0]*u_w;
      u_R[2] = u_L[0]*v_w;
      u_R[3] = u_L[0]*(RT/(gammaVal-1.0) + (u_w*u_w + v_w*v_w)*0.5);

      for ( int j = 0; j < params->nvar; j++ )
      {
        u_I[j] = (u_L[j] + u_R[j])*0.5 - params->beta*(u_L[j]-u_R[j]);
      }
      //overwrite f_face
      for ( int j = 0; j < params->nvar; j++ )
      {
        indx_L = pairL[i]*params->columnL + loc_q[j];
        u_face[indx_L] = u_I[j] - u_face[indx_L];
      }
    }
  }
} //couette


  delete[] loc_q, u_L, u_R, u_I, pairL, pairR;
  return;
}




void roe_flux(double *q_L, double *q_R, double n_x, double n_y, double *fluxI, double *q_hat)
{

  double rho_L, u_L, v_L, E_L, p_L, c_L, H_L;
  rho_L = q_L[0];
  u_L = q_L[1]/q_L[0];
  v_L = q_L[2]/q_L[0];
  E_L = q_L[3];
  p_L = (gammaVal-1)*(E_L-0.5*rho_L*(u_L*u_L+v_L*v_L));
  c_L = sqrt(gammaVal*p_L/rho_L);
  H_L = (p_L+E_L)/rho_L;

  double rho_R, u_R, v_R, E_R, p_R, c_R, H_R;
  rho_R = q_R[0];
  u_R = q_R[1]/q_R[0];
  v_R = q_R[2]/q_R[0];
  E_R = q_R[3];
  p_R = (gammaVal-1)*(E_R-0.5*rho_R*(u_R*u_R+v_R*v_R));
  c_R = sqrt(gammaVal*p_R/rho_R);
  H_R = (p_R+E_R)/rho_R;

  double rsl, rsr, rho_hat, u_hat, v_hat, H_hat, c_hat, E_hat, p_hat;
  rsl = sqrt(rho_L);
  rsr = sqrt(rho_R);
  rho_hat = rsl*rsr;
  u_hat = (rsl*u_L+rsr*u_R)/(rsl+rsr);
  v_hat = (rsl*v_L+rsr*v_R)/(rsl+rsr);
  H_hat = (rsl*H_L+rsr*H_R)/(rsl+rsr);
  c_hat = sqrt((gammaVal-1)*(H_hat-0.5*(u_hat*u_hat+v_hat*v_hat)));
  p_hat = rho_hat*c_hat*c_hat/gammaVal;
  E_hat = rho_hat*H_hat-p_hat;

  q_hat[0] = rho_hat;
  q_hat[1] = rho_hat*u_hat;
  q_hat[2] = rho_hat*v_hat;
  q_hat[3] = E_hat;

  double lambda[4] = { u_hat*n_x+v_hat*n_y-c_hat,
                       u_hat*n_x+v_hat*n_y,
                       u_hat*n_x+v_hat*n_y+c_hat,
                       u_hat*n_x+v_hat*n_y };
  for ( int i = 0; i < 4; i++ ) lambda[i] = std::abs(lambda[i]);

  double lam_L[4] = { u_L*n_x+v_L*n_y-c_L,
                      u_L*n_x+v_L*n_y,
                      u_L*n_x+v_L*n_y+c_L,
                      u_L*n_x+v_L*n_y };
  double lam_R[4] = { u_R*n_x+v_R*n_y-c_R,
                      u_R*n_x+v_R*n_y,
                      u_R*n_x+v_R*n_y+c_R,
                      u_R*n_x+v_R*n_y };

  double eps = 0.01;
  for ( int i = 0; i < 4; i++ ) eps = std::max(std::abs(lam_R[i]-lam_L[i]),eps);
  for ( int i = 0; i < 4; i++ ) if ( std::abs(lambda[i]) <= 2*eps ) lambda[i] = lambda[i]*lambda[i]/(4*eps)+eps;

  double r_eig[4][4] = { {1.0, 1.0, 1.0, 0.0},
                         {u_hat-c_hat*n_x, u_hat, u_hat+c_hat*n_x,-n_y},
                         {v_hat-c_hat*n_y, v_hat, v_hat+c_hat*n_y, n_x},
                         { H_hat-(u_hat*n_x+v_hat*n_y)*c_hat, 0.5*(pow(u_hat,2)+pow(v_hat,2)), 
                           H_hat+(u_hat*n_x+v_hat*n_y)*c_hat, -u_hat*n_y+v_hat*n_x } };

  double w0[4] = { ( p_R-p_L - rho_hat*c_hat*( (u_R*n_x+v_R*n_y) 
                                             - (u_L*n_x+v_L*n_y) ) )/(2.0*c_hat*c_hat),
                  -(p_R-p_L)/(c_hat*c_hat) + rho_R-rho_L,
                   ( p_R-p_L + rho_hat*c_hat*( (u_R*n_x+v_R*n_y)
                                             - (u_L*n_x+v_L*n_y) ) )/(2.0*c_hat*c_hat),
                   rho_hat*((-u_R*n_y+v_R*n_x)+(u_L*n_y-v_L*n_x)) };

  double diss[4];
  for ( int i = 0; i < 4; i++ )
  {
    diss[i] = 0;
    for ( int j = 0; j < 4; j++ )
    {
      diss[i] += r_eig[i][j]*lambda[j]*w0[j];
    }
  }

  double flux_L[4], flux_R[4];
  flux_L[0] = q_L[1]*n_x + q_L[2]*n_y;
  flux_L[1] = (q_L[1]*n_x + q_L[2]*n_y)*q_L[1]/q_L[0] + p_L*n_x;
  flux_L[2] = (q_L[1]*n_x + q_L[2]*n_y)*q_L[2]/q_L[0] + p_L*n_y;
  flux_L[3] = (q_L[1]*n_x + q_L[2]*n_y)*H_L;
  flux_R[0] = q_R[1]*n_x + q_R[2]*n_y;
  flux_R[1] = (q_R[1]*n_x + q_R[2]*n_y)*q_R[1]/q_R[0] + p_R*n_x;
  flux_R[2] = (q_R[1]*n_x + q_R[2]*n_y)*q_R[2]/q_R[0] + p_R*n_y;
  flux_R[3] = (q_R[1]*n_x + q_R[2]*n_y)*H_R;

  //for ( int i = 0; i < 4; i++ ) fluxI[i] = 0.5*(flux_L[i] + flux_R[i] - diss[i]);

  //switch to rusanov
  double L_max = std::max(std::abs(lam_L[0]),std::abs(lam_L[2]));
  double R_max = std::max(std::abs(lam_R[0]),std::abs(lam_R[2]));
  double rus_eig = std::max(L_max,R_max);
  for ( int i = 0; i < 4; i++ ) fluxI[i] = 0.5*(flux_L[i] + flux_R[i] - rus_eig*(q_R[i]-q_L[i]));

  q_hat[0] = rus_eig;
  q_hat[1] = rus_eig;
  q_hat[2] = rus_eig;
  q_hat[3] = rus_eig;

  return;
}

//nodeViscFlux(U_vals, pd_U_x, pd_U_y, fv_node, gv_node);
void nodeViscFlux(double *U_vals, double *pd_U_x, double *pd_U_y, double mu, double Pr, double *fv_node, double *gv_node)
{

  double inv_rho, vel_u, vel_v, E;
  double pd_u_x, pd_u_y, pd_v_x, pd_v_y, pd_e_x, pd_e_y;
  double tau_xx, tau_xy, tau_yy, Q_x, Q_y;

  inv_rho = 1.0/U_vals[0];
  vel_u = U_vals[1]*inv_rho;
  vel_v = U_vals[2]*inv_rho;
  E = U_vals[3]*inv_rho;

  pd_u_x = inv_rho*(pd_U_x[1]-vel_u*pd_U_x[0]);
  pd_u_y = inv_rho*(pd_U_y[1]-vel_u*pd_U_y[0]);
  pd_v_x = inv_rho*(pd_U_x[2]-vel_v*pd_U_x[0]);
  pd_v_y = inv_rho*(pd_U_y[2]-vel_v*pd_U_y[0]);
  pd_e_x = inv_rho*( (pd_U_x[3]-E*pd_U_x[0])
                   - vel_u*(pd_U_x[1]-vel_u*pd_U_x[0])
                   - vel_v*(pd_U_x[2]-vel_v*pd_U_x[0]) );
  pd_e_y = inv_rho*( (pd_U_y[3]-E*pd_U_y[0])
                   - vel_u*(pd_U_y[1]-vel_u*pd_U_y[0])
                   - vel_v*(pd_U_y[2]-vel_v*pd_U_y[0]) );

  tau_xx = mu*(4.0/3.0*pd_u_x-2.0/3.0*pd_v_y);
  tau_yy = mu*(4.0/3.0*pd_v_y-2.0/3.0*pd_u_x);
  tau_xy = mu*(pd_u_y+pd_v_x);
  Q_x = -gammaVal*mu/Pr*pd_e_x;
  Q_y = -gammaVal*mu/Pr*pd_e_y;

  fv_node[0] = 0;
  fv_node[1] = tau_xx;
  fv_node[2] = tau_xy;
  fv_node[3] = vel_u*tau_xx+vel_v*tau_xy-Q_x;
  gv_node[0] = 0;
  gv_node[1] = tau_xy;
  gv_node[2] = tau_yy;
  gv_node[3] = vel_u*tau_xy+vel_v*tau_yy-Q_y;

  return;
}

void calc_matrix_A(double *q_hat, double n_x, double n_y, double *matrix_A)
{

  double r, u, v, e, ke, p, c;
  r = q_hat[0];
  u = q_hat[1]/q_hat[0];
  v = q_hat[2]/q_hat[0];
  e = q_hat[3];
  ke = 0.5*(u*u+v*v);
  p = (gammaVal-1)*(e-r*ke);
  c = sqrt(gammaVal*p/r);

  double un, h;
  un = n_x*u + n_y*v;
  h = (e+p)/r;

  double lambda[4];
  lambda[0] = un-c;
  lambda[1] = un;
  lambda[2] = un+c;
  lambda[3] = un;

  for ( int i = 0; i < 4; i++ ) lambda[i] = std::abs(lambda[i]);
  double eps = 0.01;
  for ( int i = 0; i < 4; i++ ) if ( std::abs(lambda[i]) <= 2*eps ) lambda[i] = lambda[i]*lambda[i]/(4*eps)+eps;

  double rt[4][4], lt[4][4];

  rt[0][0] = 1; rt[0][1] = 1; rt[0][2] = 1; rt[0][3] = 0;
  rt[1][0] = u-c*n_x; rt[1][1] = u; rt[1][2] = u+c*n_x; rt[1][3] = n_y;
  rt[2][0] = v-c*n_y; rt[2][1] = v; rt[2][2] = v+c*n_y; rt[1][3] =-n_x;
  rt[3][0] = h-un*c; rt[3][1] = ke; rt[3][2] = h+un*c; rt[3][3] = u*n_y-v*n_x;

  for ( int i = 0; i < 4; i++ )
  {
    for ( int j = 0; j < 4; j++ )
    {
      rt[i][j] *= lambda[i];
    }
  }

  double invc2 = 1/(c*c);
  lt[0][0]= (ke*(gammaVal-1)+c*un)*0.5*invc2;
  lt[0][1]= (-u*(gammaVal-1)-c*n_x)*0.5*invc2;
  lt[0][2]= (-v*(gammaVal-1)-c*n_y)*0.5*invc2;
  lt[0][3]= (gammaVal-1)*0.5*invc2;
  lt[1][0]= 1 -(gammaVal-1)*ke*invc2;
  lt[1][1]= (gammaVal-1)*u*invc2;
  lt[1][2]= (gammaVal-1)*v*invc2;
  lt[1][3]=-(gammaVal-1)*invc2;
  lt[2][0]= (ke*(gammaVal-1)-c*un)*0.5*invc2;
  lt[2][1]= (-u*(gammaVal-1)+c*n_x)*0.5*invc2;
  lt[2][2]= (-v*(gammaVal-1)+c*n_y)*0.5*invc2;
  lt[2][3]= (gammaVal-1)*0.5*invc2;
  lt[3][0]= v*n_x-u*n_y;
  lt[3][1]= n_y;
  lt[3][2]=-n_x;
  lt[3][3]= 0;

  //matrix_A[0-15]
  for ( int i = 0; i < 4; i++ )
  {
    for ( int j = 0; j < 4; j++ )
    {
      matrix_A[i*4+j] = 0;
      for ( int k = 0; k < 4; k++ )
      {
        matrix_A[i*4+j] += lt[i][k]*rt[k][j];
      }
    }
  }

  return;
}


void correct_Flux( essential *params, int i_elem, double *jacob, int i_start, int jCN, double *u_hat_face, int i_u, double n_x, double n_y )
{

  int *loc_u;
  loc_u = new int [params->nvar];
  double u_hat[4];

  for ( int i = 0; i < params->nvar; i++ )
    loc_u[i] = i_u*params->columnL + i*params->nelem + i_elem;
  for ( int i = 0; i < params->nvar; i++ )
  {
    u_hat[i] = u_hat_face[loc_u[i]];
  }
/*
  double *matrix_A = new double [params->nvar*params->nvar];

  //calc_matrix_A(double *q_hat, double n_x, double n_y, double *matrix_A)
  calc_matrix_A(u_hat, n_x, n_y, matrix_A);

  // edit jacob here
  for ( int i = 0; i < params->nvar; i++ )
  {
    for ( int j = 0; j < params->nvar; j++ )
    {
      jacob[i_start + i*jCN + j] += matrix_A[i*params->nvar+j];
      //std::cout << matrix_A[i*params->nvar+j] << " ";
    }
  }
*/

  //for rusanov
  for ( int i = 0; i < params->nvar; i++ )
  {
    jacob[i_start + i*jCN + i] += u_hat[i];
  }

  // trash matix_A
  //delete[] matrix_A;
  delete[] loc_u;
  return;
}


void F_flux_jacob( essential *params, int i_elem, double *jacob, int i_start, int jCN, double *u_vals, int i_u )
{
          // Evaluate interface flux jacobians;
          // store in F_I_LR G_I_BT
          // F_L, F_R, G_B, G_T respectively;
          // F_L;
          int *loc_u;
          loc_u = new int [params->nvar];
          double u_vel, v_vel, p, H;
          int i_block;
          for ( int l = 0; l < params->nvar; l++ )
            loc_u[l] = i_u*params->columnL + l*params->nelem + i_elem;
          u_vel = u_vals[loc_u[1]]/u_vals[loc_u[0]];
          v_vel = u_vals[loc_u[2]]/u_vals[loc_u[0]];
          p = (gammaVal-1)*(u_vals[loc_u[3]]-0.5*u_vals[loc_u[0]]*(u_vel*u_vel+v_vel+v_vel));
          H = (p+u_vals[loc_u[3]])/u_vals[loc_u[0]];
          //F jacob
          i_block = i_start;
          jacob[i_block + 0] = 0;
          jacob[i_block + 1] = 1;
          jacob[i_block + 2] = 0;
          jacob[i_block + 3] = 0;
          i_block += jCN;
          jacob[i_block + 0] = (gammaVal-1)*0.5*(u_vel*u_vel+v_vel*v_vel) - u_vel*u_vel;
          jacob[i_block + 1] = (3-gammaVal)*u_vel;
          jacob[i_block + 2] = (1-gammaVal)*v_vel;
          jacob[i_block + 3] = (gammaVal-1);
          i_block += jCN;
          jacob[i_block + 0] =-u_vel*v_vel;
          jacob[i_block + 1] = v_vel;
          jacob[i_block + 2] = u_vel;
          jacob[i_block + 3] = 0;
          i_block += jCN;
          jacob[i_block + 0] =-u_vel*( gammaVal*u_vals[loc_u[3]]/u_vals[loc_u[0]]
                                      - (gammaVal-1)*(u_vel*u_vel+v_vel*v_vel) );
          jacob[i_block + 1] = gammaVal*u_vals[loc_u[3]]/u_vals[loc_u[0]]
                              - (gammaVal-1)*0.5*(3*u_vel*u_vel+v_vel*v_vel);
          jacob[i_block + 2] = (1-gammaVal)*u_vel*v_vel;
          jacob[i_block + 3] = gammaVal*u_vel;
/*
          //linear advection;
          for ( int i = 0; i < params->nvar; i++ )
          {
            for ( int j = 0; j < params->nvar; j++ )
            {
              if ( i == j ) jacob[i_start+i*jCN+j] = 1;
              else jacob[i_start+i*jCN+j] = 0;
            }
          }
*/
  delete[] loc_u;
  return;
}

void G_flux_jacob( essential *params, int i_elem, double *jacob, int i_start, int jCN, double *u_vals, int i_u )
{
          // Evaluate interface flux jacobians;
          // store in F_I_LR G_I_BT
          // F_L, F_R, G_B, G_T respectively;
          // F_L;
          int *loc_u;
          loc_u = new int [params->nvar];
          double u_vel, v_vel, p, H;
          int i_block;
          for ( int l = 0; l < params->nvar; l++ )
            loc_u[l] = i_u*params->columnL + l*params->nelem + i_elem;
          u_vel = u_vals[loc_u[1]]/u_vals[loc_u[0]];
          v_vel = u_vals[loc_u[2]]/u_vals[loc_u[0]];
          p = (gammaVal-1)*(u_vals[loc_u[3]]-0.5*u_vals[loc_u[0]]*(u_vel*u_vel+v_vel+v_vel));
          H = (p+u_vals[loc_u[3]])/u_vals[loc_u[0]];
          //G jacob
          i_block = i_start;
          jacob[i_block + 0] = 0;
          jacob[i_block + 1] = 0;
          jacob[i_block + 2] = 1;
          jacob[i_block + 3] = 0;
          i_block += jCN;
          jacob[i_block + 0] =-u_vel*v_vel;
          jacob[i_block + 1] = v_vel;
          jacob[i_block + 2] = u_vel;
          jacob[i_block + 3] = 0;
          i_block += jCN;
          jacob[i_block + 0] = (gammaVal-1)*0.5*(u_vel*u_vel+v_vel*v_vel) - v_vel*v_vel;
          jacob[i_block + 1] = (1-gammaVal)*u_vel;
          jacob[i_block + 2] = (3-gammaVal)*v_vel;
          jacob[i_block + 3] = (gammaVal-1);
          i_block += jCN;
          jacob[i_block + 0] =-v_vel*( gammaVal*u_vals[loc_u[3]]/u_vals[loc_u[0]]
                                     - (gammaVal-1)*(u_vel*u_vel+v_vel*v_vel) );
          jacob[i_block + 1] = (1-gammaVal)*u_vel*v_vel;
          jacob[i_block + 2] = gammaVal*u_vals[loc_u[3]]/u_vals[loc_u[0]]
                             - (gammaVal-1)*0.5*(3*v_vel*v_vel+u_vel*u_vel);
          jacob[i_block + 3] = gammaVal*v_vel;
/*
          //linear advection;
          for ( int i = 0; i < params->nvar; i++ )
          {
            for ( int j = 0; j < params->nvar; j++ )
            {
              if ( i == j ) jacob[i_start+i*jCN+j] = 1;
              else jacob[i_start+i*jCN+j] = 0;
            }
          }
*/
  delete[] loc_u;
  return;
}


void writeSolution(essential *params, double *u, int iter)
{

  std::ofstream solution;
  char name[50];
  sprintf(name, "solution_%d.csv", iter);
  solution.open(name);
  double x, y, x_L =-10, x_R = 10, y_B =-10, y_T = 10;
  double dx = (x_R-x_L)/params->nelem_x, dy = (y_T-y_B)/params->nelem_y;
  double avg_u[4];
  int i_elem;
  for ( int i = 0; i < params->nelem_x; i++ )
  {
    for ( int j = 0; j < params->nelem_y; j++ )
    {
      i_elem = j*params->nelem_x+i;
      x = x_L + i*dx + dx/2.0;
      y = y_B + j*dy + dy/2.0;
      for ( int k = 0; k < params->nvar; k++ ) avg_u[k] = 0;
      for ( int k = 0; k < params->nse; k++ )
      {
        for ( int l = 0; l < params->nvar; l++ )
        {
          avg_u[l] += u[k*params->columnL+i_elem+l*params->nelem];
        }
      }
      solution << x << ", " << y;
      for ( int k = 0; k < params->nvar; k++ ) solution << ", "  << avg_u[k]/params->nse;
      solution << "\n";
    }
  }

}


void set_solnPoints(int p, double **soln_coords, double **weights)
{

  //std::cout << "hi" << std::endl;
  *soln_coords = new double [p+1];
  *weights = new double [p+1];
  //std::cout << "hope you are okay" << std::endl;
  if ( p == 0 )
  {
    (*soln_coords)[0] = 0;
    (*weights)[0] = 2.0;
  }
  else if ( p == 1 )
  {
    (*soln_coords)[0] =-0.577350269189625764509148780502;
    (*soln_coords)[1] = 0.577350269189625764509148780502;
    (*weights)[0] = 1.0;
    (*weights)[1] = 1.0;
  }
  else if ( p == 2 )
  {//0.0011270166538
    (*soln_coords)[0] =-0.774596669241483377035853079956;
    (*soln_coords)[1] = 0.0;
    (*soln_coords)[2] = 0.774596669241483377035853079956;
    (*weights)[0] = 0.555555555555555555555555555556;
    (*weights)[1] = 0.888888888888888888888888888889;
    (*weights)[2] = 0.555555555555555555555555555556;
  }
  else if ( p == 3 )
  {
    (*soln_coords)[0] =-0.861136311594052575223946488893;
    (*soln_coords)[1] =-0.339981043584856264802665759103;
    (*soln_coords)[2] = 0.339981043584856264802665759103;
    (*soln_coords)[3] = 0.861136311594052575223946488893;
    (*weights)[0] = 0.347854845137453857373063949222;
    (*weights)[1] = 0.652145154862546142626936050778;
    (*weights)[2] = 0.652145154862546142626936050778;
    (*weights)[3] = 0.347854845137453857373063949222;
  }

  std::cout << "solution points are: \n";
  for (int i = 0; i < p+1; i++)
  {
    std::cout << (*soln_coords)[i] << " ";
  }
  std::cout << std::endl;

  std::cout << "weights associated are: \n";
  for (int i = 0; i < p+1; i++)
  {
    std::cout << (*weights)[i] << " ";
  }
  std::cout << std::endl;
  return;
}

void set_lagrangeDerivs(int p, double *soln_coords, double **derivs)
{

  *derivs = new double [(p+1)*(p+1)];
  std::cout << "lagrangeDerivs\n";
  for (int i = 0; i < p+1; i++)
  {
    for (int j = 0; j < p+1; j++)
    {
      (*derivs)[i*(p+1) + j] = 0;
      for (int k = 0; k < p+1; k++)
      {
        if ( k==j ) continue;
        double mult = 1.0;
        for (int n = 0; n < p+1; n++)
        {
          if ( n==j || n==k ) continue;
          mult *= (soln_coords[i]-soln_coords[n])/(soln_coords[j]-soln_coords[n]);
        }
        // i*jmax + j
        //derivs[i][j] += 1/(soln_coords[j]-soln_coords[k])*mult;
        (*derivs)[i*(p+1) + j] += 1/(soln_coords[j]-soln_coords[k])*mult;
      }
      std::cout << (*derivs)[i*(p+1) + j] << " " ;
    }
    std::cout << "\n";
  }
  return;
}

void set_correctionDerivs(int p, double *soln_coords, double **hL, double **hR)
{

  *hL = new double[p+1];
  *hR = new double[p+1];
  
  if ( p == 0 )
  {
    (*hL)[0] = -0.5;
  }
  else if ( p == 1 )
  {
    for ( int i = 0; i < p+1; i++ ) 
      (*hL)[i] = 1.5*soln_coords[i] - 0.5;
  }
  else if ( p == 2 )
  {
    for ( int i = 0; i < p+1; i++ ) 
      (*hL)[i] = 0.25*(-15.0*pow(soln_coords[i],2)+6.0*soln_coords[i]+3.0);
  }
  else if ( p == 3 )
  {
    for ( int i = 0; i < p+1; i++ ) 
      (*hL)[i] = 0.125*(70*pow(soln_coords[i],3)-30*pow(soln_coords[i],2)-30*soln_coords[i]+6);
  }

  // uncomment below to set all 0.5 and -..
  //for ( int i = 0; i < p+1; i++ ) hL[i] = -0.5;
  for ( int i = 0; i < p+1; i++ ) (*hR)[i] = -(*hL)[p-i];

  std::cout << "left correction coeffs\n";
  for ( int i = 0; i < p+1; i++ ) 
  {
    std::cout << (*hL)[i] << " ";
  }
  std::cout << "\nright correction coeffs\n";
  for ( int i = 0; i < p+1; i++ )
  {
    std::cout << (*hR)[i] << " ";
  }
  std::cout << "\n";

  return;
}

void set_lagrangeInterCoeffs(int p, double *soln_coords, double **lagrInterL, double **lagrInterR)
{

  *lagrInterL = new double[p+1];
  *lagrInterR = new double[p+1];

  double left  =-1.0;
  double right = 1.0;

  //left coeffs
  std::cout << "left coeffs\n";
  for (int i = 0; i < p+1; i++)
  {
    (*lagrInterL)[i] = 1.0;
    for (int j = 0; j < p+1; j++)
    {
      if ( i == j ) continue;
      (*lagrInterL)[i] *= (left-soln_coords[j])
                         /(soln_coords[i]-soln_coords[j]);
    }
    std::cout << (*lagrInterL)[i] << " ";
  }
  //right coeffs
  std::cout << "\nright coeffs\n";
  for (int i = 0; i < p+1; i++)
  {
    (*lagrInterR)[i] = 1.0;
    for (int j = 0; j < p+1; j++)
    {
      if ( i == j ) continue;
      (*lagrInterR)[i] *= (right-soln_coords[j])
                         /(soln_coords[i]-soln_coords[j]);
    }
    std::cout << (*lagrInterR)[i] << " ";
  }
  std::cout << std::endl;
  return;
}



