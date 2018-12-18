// 1D Flux Reconstruction Code - 22.11.2018
// Semih Akkurt
// git - master/2D

// start doint dynamic memory

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


struct essential {
  double dt, jacob, j_x, j_y, ndim;
  int nvar, porder, nnode, nelem, nelem_x, nelem_y, columnL, maxIte, nRK, nse, nfe;
} ;

const double gammaVal=1.4;

void set_solnPoints(int p, double **soln_coords, double **weights);
void set_lagrangeDerivs(int p, double *soln_coords, double **derivs);
void set_lagrangeInterCoeffs(int p, double *soln_coords, double **lagrInterL, double **lagrInterR);
void set_correctionDerivs(int p, double *soln_coords, double **hL, double **hR);
void get_flux(double *nvar, double *flux);
void roe_flux(double *q_L, double *q_R, double n_x, double n_y, double *fluxI);

void superFunc( essential* params, double *u, double *f, double *g, 
                double *u_face, double *f_face, //double *g_face, 
                double *lagrInterL, double *lagrInterR );
void update( essential *params, double *u, double *f, double *f_face, 
             double *lagrDerivs, double *hL, double *hR );
void computeFlux(essential* params, double *u_face, double *f_face);



int main()
{
  double L, jacob, dt;
  L = 10;
  essential params;
  double L_x, L_y;
  //int nelem_x, nelem_y, nnode;
  //double J_x, J_y, Jacob;

  L_x = 10;
  L_y = 10;
  params.nelem_x = 100;
  params.nelem_y = 100;
  params.nnode = (params.nelem_x+1)*(params.nelem_y+1);
  params.j_x = L_x/params.nelem_x/2.0;
  params.j_y = L_y/params.nelem_y/2.0;
  params.jacob = params.j_x*params.j_y;

  params.ndim   = 2;
  params.nvar   = 4;
  params.porder = 2; // 0 || 1 || 2 || 3
  params.nse    = pow(params.porder+1,2);
  params.nfe    = 4*(params.porder+1); //only for quad
  params.dt     = 0.0001;
  params.nelem  = params.nelem_x*params.nelem_y;
  params.maxIte = 10;
  params.nRK = 4; //1 || 4
  params.columnL = params.nvar*params.nelem;
  //params.jacob  = L/params.nelem/2;

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
  double *u, *f, *g, *u_face, *f_face;//, *g_face;
  u = new double [params.nse*params.columnL];
  f = new double [params.nse*params.columnL];
  g = new double [params.nse*params.columnL];
  u_face = new double [params.nfe*params.columnL];
  f_face = new double [params.nfe*params.columnL];
  //g_face = new double [params.nfe*params.columnL];

  // 2N storage RK
  double *old_u;
  old_u = new double [params.nse*params.columnL];

  std::cout << "ndim: " << params.ndim << "\n";
  std::cout << "polynomial order is: " << params.porder << "\n";
  std::cout << "number of solution points per element: " << params.nse << "\n";
  std::cout << "number of face points per element: " << params.nfe << "\n";
  std::cout << "nvar: " << params.nvar << "\n";
  std::cout << "max ite: " << params.maxIte << "\n";
  std::cout << "nelem: " << params.nelem << "\n";
  std::cout << "dx: " << L/params.nelem << "\n";
  std::cout << "Lenght: " << L << "\n";
  std::cout << "jacob: " << params.jacob << "\n";
  std::cout << "dt: " << params.dt << "\n";
  std::cout << "end time: " << params.maxIte*params.dt << "\n";
  std::cout << "RK stage: " << params.nRK << std::endl;

  double *alpha;
  alpha = new double [params.nRK];

  if ( params.nRK == 1 )
  {
    alpha[0] = 1.0;
  }
  else if ( params.nRK == 4 )
  {
    alpha[0] = 0.25; alpha[1] = 1.0/3.0; alpha[2] = 0.5; alpha[3] = 1.0;
  }

  std::cout << "alpha: ";
  for ( int i = 0; i < params.nRK; i++ )
  {
    std::cout << alpha[i] << " ";
  }
  std::cout << std::endl;


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
    double rho_inf = 1, u_inf = 1, v_inf = 0, p_inf = 1;
  }

  std::cout << "main loop begins\n";

  // MAIN LOOP
  for ( int ite = 0; ite < params.maxIte; ite++ )
  {
    for ( int i = 0; i < params.nse*params.columnL; i++ ) old_u[i] = u[i];
    for ( int i = 0; i < params.nRK; i++ )
    {
      //std::cout << "ite: " << ite << " ";
      //superFunc creates f, u_face, and f_face arrays
      superFunc( &params, u, f, g, u_face, f_face, //g_face, 
                 lagrInterL, lagrInterR );

      //call flux function for each colocated uL uR pair of 2 neighbour elements
      // also update f_face array to have f_I{L/R}-f_D{L/R} as entries
      //computeFlux(&params, u_face, f_face);

      //update solution
      //update(&params, u, f, f_face, lagrDerivs, hL, hR);

      for ( int j = 0; j < params.nse*params.columnL; j++ )
      {
        u[j] = old_u[j] + alpha[i]*params.dt*u[j];
      }
    }
  }



/*
  // print solution out
  std::ofstream solution;
  solution.open("solution");
  for ( int i = 0; i < params.nelem; i++ )
  { //print the first var at the 0th node in each element, for second var add +nelem
    solution << u[0*params.columnL + i] << "\n";
    //for ( int j = 0; j < params.porder+1; j++ )
      //solution << u[j*params.columnL + i] << "\n";
  }
*/

/*
  //error
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
  return 0;
}




void superFunc( essential* params, double *u, double *f, double *g, 
                double *u_face, double *f_face, //double *g_face, 
                double *lagrInterL, double *lagrInterR )
{

  // interpolate solution to faces
  // compute flux on the solution nodes
  // interpolate flux to faces

  //int loc_q[nvar];
  int *loc_q = new int[params->nvar];

  //double vel_u, vel_v, p;

  int indx_L, indx_R, indx_B, indx_T, indx_elem;

  for ( int i_elem = 0; i_elem < params->nelem; i_elem++ )
  {
    for ( int j = 0; j < params->nvar; j++ ) loc_q[j] = i_elem+j*params->nelem; //column pos
    for ( int j = 0; j < params->nvar; j++ )
    {
      for ( int k = 0; k < params->nfe; k++ )
      {
        u_face[params->columnL*k + loc_q[j]] = 0;
        f_face[params->columnL*k + loc_q[j]] = 0;
        //g_face[params->columnL*k + loc_q[j]] = 0;
      }
    }
    //rows
    for ( int j = 0; j < params->porder+1; j++ )
    {
      for ( int k = 0; k < params->nvar; k++ )
      {
        indx_L = (4*(params->porder+1)-1-j)*params->columnL+loc_q[k];//i_elem;
        indx_R = (params->porder+1+j)*params->columnL+loc_q[k];//+i_elem;
        for ( int i = 0; i < params->porder+1; i++ )
        {
          indx_elem = (j*(params->porder+1)+i)*params->columnL+loc_q[k];//+i_elem;
          indx_B = i*params->columnL+loc_q[k];//+i_elem;
          indx_T = (3*(params->porder+1)-1-i)*params->columnL+loc_q[k];//+i_elem;
          u_face[indx_L] += lagrInterL[i]*u[indx_elem];
          f_face[indx_L] += lagrInterL[i]*f[indx_elem];
          //g_face[indx_L] += lagrInterL[i]*g[indx_elem];
          u_face[indx_R] += lagrInterR[i]*u[indx_elem];
          f_face[indx_R] += lagrInterR[i]*f[indx_elem];
          //g_face[indx_R] += lagrInterR[i]*g[indx_elem];
          u_face[indx_B] += lagrInterL[j]*u[indx_elem];
          f_face[indx_B] += lagrInterL[j]*g[indx_elem];//from g
          //g_face[indx_B] += lagrInterL[j]*g[indx_elem];
          u_face[indx_T] += lagrInterR[j]*u[indx_elem];
          f_face[indx_T] += lagrInterR[j]*g[indx_elem];//from g
          //g_face[indx_T] += lagrInterR[j]*g[indx_elem];
        }
      }
      // after k loop;
      // At this point all rho rhoU rhoV E for a point are already in the cache I hope
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
        g[indx_elem+loc_q[2]] = rho*vel_v*vel_v + p;
        g[indx_elem+loc_q[1]] = rho*vel_u*vel_v;
        f[indx_elem+loc_q[2]] = g[indx_elem+loc_q[1]];//rho*vel_v*vel_u;
        f[indx_elem+loc_q[3]] = (E+p)*vel_u;
        g[indx_elem+loc_q[3]] = (E+p)*vel_v;
      }
    }
  }

  return;
}

void computeFlux(essential* params, double *u_face, double *f_face)
{

  //int loc_q[nvar];
  //double u_L[nvar], u_R[nvar], f_I[nvar];
  int *loc_q = new int[params->nvar];
  double *u_L = new double[params->nvar];
  double *u_R = new double[params->nvar];
  double *f_I = new double[params->nvar];

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
      if ( i_x != params->nelem_x-1 ) next_elem = 1;
      else next_elem = 1-params->nelem; //if at the end
      pairL[0] = 4 ; pairL[1] = 5 ; pairL[2] = 6 ; pairL[3] = 7 ;
      pairR[0] = 15; pairR[1] = 14; pairR[2] = 13; pairR[3] = 12;
      for ( int i = 0; i < params->porder+1; i++ )
      {
        for ( int j = 0; j < params->nvar; j++ )
        {
          u_L[j] = u_face[pairL[i]*params->columnL + loc_q[j]];
          u_R[j] = u_face[pairR[i]*params->columnL + loc_q[j]+next_elem]; //left of the next element
        }
        roe_flux(u_R, u_L, 1.0, 0.0, f_I);
        //overwrite f_face
        for ( int j = 0; j < params->nvar; j++ )
        {
        indx_L = pairL[i]*params->columnL + loc_q[j];
        indx_R = pairR[i]*params->columnL + loc_q[j]+next_elem;
        f_face[indx_L] = f_I[j]*params->jacob - f_face[indx_L];
        f_face[indx_R] =-f_I[j]*params->jacob - f_face[indx_R];
        }
      }
      if ( i_y != params->nelem_y-1 ) next_elem = params->nelem_x;
      else next_elem = (1-params->nelem_y)*params->nelem_x; //if at the end
      pairL[0] = 11; pairL[1] = 10; pairL[2] = 9; pairL[3] = 9;
      pairR[0] = 0 ; pairR[1] = 1 ; pairR[2] = 2; pairR[3] = 3;
      for ( int i = 0; i < params->porder+1; i++ )
      {
        for ( int j = 0; j < params->nvar; j++ )
        {
          u_L[j] = u_face[pairL[i]*params->columnL + loc_q[j]];
          u_R[j] = u_face[pairR[i]*params->columnL + loc_q[j]+next_elem]; //left of the next element
        }
        roe_flux(u_R, u_L, 0.0, 1.0, f_I);
        //overwrite f_face
        for ( int j = 0; j < params->nvar; j++ )
        {
        indx_L = pairL[i]*params->columnL + loc_q[j];
        indx_R = pairR[i]*params->columnL + loc_q[j]+next_elem;
        f_face[indx_L] = f_I[j]*params->jacob - f_face[indx_L];
        f_face[indx_R] =-f_I[j]*params->jacob - f_face[indx_R];
        }
      }
    }
  }

  return;
}

void update( essential *params, double *u, double *f, double *f_face, 
             double *lagrDerivs, double *hL, double *hR )
{

  double *dummy = new double[params->porder+1];

  for ( int i = 0; i < params->columnL; i++ )
  {
    for ( int j = 0; j < params->porder+1; j++ ) dummy[j] = 0;
    for ( int j = 0; j < params->porder+1; j++ )
    {
      for ( int k = 0; k < params->porder+1; k++ )
      {
        dummy[j] += lagrDerivs[j*(params->porder+1)+k]*f[k*params->columnL+i];
      }
      //dummy[j] += hL[j]*f_face[0][i] + hR[j]*f_face[1][i];
      dummy[j] += hL[j]*f_face[i] + hR[j]*f_face[params->columnL+i];
    }
    //for ( int j = 0; j < params->porder+1; j++ ) f[j][i] = dummy[j];
    //update
    for ( int j = 0; j < params->porder+1; j++ )
      //u[j*params->columnL+i] -= dummy[j]*params->dt/params->jacob;
      u[j*params->columnL+i] = -dummy[j]/params->jacob;
  }

  return;
}






void get_flux(double *q, double *flux)
{
  double u, p;

  flux[0] = q[1];
  u = q[1]/q[0];
  p = (gammaVal-1)*(q[2]-0.5*q[1]*u);
  flux[1] = q[1]*u+p;
  flux[2] = (q[2]+p)*u;
  return;
}

void roe_flux(double *q_L, double *q_R, double n_x, double n_y, double *fluxI)
{

  double p_L, p_R, H_L, H_R;
  p_L = (gammaVal-1)*(q_L[2]-0.5*(q_L[1]*q_L[1]+q_L[2]*q_L[2])/q_L[0]);
  p_R = (gammaVal-1)*(q_R[2]-0.5*(q_R[1]*q_R[1]+q_R[2]*q_R[2])/q_R[0]);
  H_L = (q_L[2]+p_L)/q_L[0];
  H_R = (q_R[2]+p_R)/q_R[0];

  double rho_hat, u_hat, v_hat, H_hat, a_hat;

  rho_hat = sqrt(q_L[0]*q_R[0]);
  u_hat   = (sqrt(q_L[0])*q_L[1]/q_L[0] + sqrt(q_R[0])*q_R[1]/q_R[0])
           /(sqrt(q_L[0]) + sqrt(q_R[0]));
  v_hat   = (sqrt(q_L[0])*q_L[2]/q_L[0] + sqrt(q_R[0])*q_R[2]/q_R[0])
           /(sqrt(q_L[0]) + sqrt(q_R[0]));
  H_hat   = (sqrt(q_L[0])*H_L + sqrt(q_R[0])*H_R)
           /(sqrt(q_L[0]) + sqrt(q_R[0]));
  a_hat   = sqrt((gammaVal-1)*(H_hat-0.5*(pow(u_hat,2))));

  double lambda[4] = { u_hat*n_x+v_hat*n_y-a_hat,
                       u_hat*n_x+v_hat*n_y,
                       u_hat*n_x+v_hat*n_y+a_hat,
                       u_hat*n_x+v_hat*n_y };
  for ( int i = 0; i < 3; i++ ) lambda[i] = std::abs(lambda[i]);

  double r_eig[4][4] = { {1.0, 1.0, 1.0, 0.0},
                         {u_hat-a_hat*n_x, u_hat, u_hat+a_hat*n_x,-n_y},
                         {v_hat-a_hat*n_y, v_hat, v_hat+a_hat*n_y, n_x},
                         { H_hat-(u_hat*n_x+v_hat*n_y)*a_hat, 0.5*(pow(u_hat,2)+pow(v_hat,2)), 
                           H_hat+(u_hat*n_x+v_hat*n_y)*a_hat, -u_hat*n_y+v_hat*n_x } };

  double w0[4] = { (p_R-p_L - rho_hat*a_hat*( (q_R[1]*n_x+q_R[2]*n_y)/q_R[0]
                                            - (q_L[1]*n_x+q_L[2]*n_y)/q_L[0] ))/(2.0*a_hat*a_hat),
                  -(p_R-p_L)/(a_hat*a_hat) + q_R[0]-q_L[0],
                   (p_R-p_L + rho_hat*a_hat*( (q_R[1]*n_x+q_R[2]*n_y)/q_R[0]
                                            - (q_L[1]*n_x+q_L[2]*n_y)/q_L[0] ))/(2.0*a_hat*a_hat),
                   rho_hat*((-q_R[1]*n_y+q_R[2]*n_x)/q_R[0]+(q_L[1]*n_y-q_L[2]*n_x)/q_L[0]) };
  //w0[0] =-(p_R-p_L)/(2*a_hat*a_hat) + q_R[0] - q_L[0];
  //w0[1] = (p_R-p_L)/(2*a_hat*a_hat)
  //      + (q_R[1] - q_L[1] - u_hat*(q_R[0] - q_L[0]))/(2*a_hat);
  //w0[2] = (p_R-p_L)/(2*a_hat*a_hat)
  //      - (q_R[1] - q_L[1] - u_hat*(q_R[0] - q_L[0]))/(2*a_hat);


  double flux_L[4], flux_R[4], diss[4];

  for ( int i = 0; i < 4; i++ )
  {
    diss[i] = 0;
    for ( int j = 0; j < 4; j++ )
    {
      diss[i] += r_eig[i][j]*lambda[j]*w0[j];
    }
  }

  get_flux(q_L, flux_L);
  get_flux(q_R, flux_R);
  for ( int i = 0; i < 4; i++ ) fluxI[i] = 0.5*(flux_L[i] + flux_R[i] - diss[i]);
  //fluxI[1] = 0.5*(flux_L[1] + flux_R[1] - diss[1]);
  //fluxI[2] = 0.5*(flux_L[2] + flux_R[2] - diss[2]);

  return;
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
    for ( int i = 0; i < p+1; i++ ) (*hL)[i] = -0.5;
  }
  else if ( p == 2 )
  {
    for ( int i = 0; i < p+1; i++ ) (*hL)[i] = 1.5*soln_coords[i] - 0.5;
  }
  else if ( p == 3 )
  {
    for ( int i = 0; i < p+1; i++ ) 
      (*hL)[i] = 0.25*(-15.0*pow(soln_coords[i],2)+6.0*soln_coords[i]+3.0);
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



