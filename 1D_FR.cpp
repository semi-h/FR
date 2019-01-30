// 1D Flux Reconstruction Code - 22.11.2018
// Semih Akkurt
// git - master/1D_implicit

// start doing dynamic memory

// set initial condition
// evaluate derivatives of correction function at gauss legendre points and store
// compute F_D left and right lagrange fit evaluation constants


// loop
// using known solution, construct discontinious flux function
// call flux function for each cell boundary using reconstructed solution
// update solution points

//g++ -O2 -o run_FR 1D_FR.cpp -I/home/semih/lapack-3.8.0/LAPACKE/include -L/home/semih/lapack-3.8.0 -llapacke -llapack -lrefblas -lgfortran

//g++ -O2 -o run_FR 1D_FR.cpp -I/home/semih/lapack-3.8.0/LAPACKE/include -I/home/semih/lapack-3.8.0/CBLAS/include -L/home/semih/lapack-3.8.0 -llapacke -llapack -lcblas -lrefblas -lgfortran

#include <iostream>
#include <fstream>
#include <math.h>
#include <cmath>


struct essential {
  double dt, jacob;
  int nvar, porder, nelem, columnL, maxIte, nRK, nU;
} ;

const double gammaVal=1.4;

void set_solnPoints(int p, double **soln_coords, double **weights);
void set_lagrangeDerivs(int p, double *soln_coords, double **derivs);
void set_lagrangeInterCoeffs(int p, double *soln_coords, double **lagrInterL, double **lagrInterR);
void set_correctionDerivs(int p, double *soln_coords, double **hL, double **hR);
void get_flux(double *nvar, double *flux);
void roe_flux(double *q_L, double *q_R, double *fluxI);

void superFunc( essential* params, double *u, double *f, double *u_LR, double *f_LR, 
                double *lagrInterL, double *lagrInterR );
void update( essential *params, double *u, double *f, double *f_LR, 
             double *lagrDerivs, double *hL, double *hR );
void computeFlux(essential* params, double *u_LR, double *f_LR);


#include <lapacke.h>
#include <lapacke_utils.h>
#include <cblas.h>
int dene ()
{
   double a[5][3] = {1,1,1,2,3,4,3,5,2,4,2,5,5,4,3};
   double b[5][2] = {-10,-3,12,14,14,12,16,16,18,16};
   lapack_int info,m,n,lda,ldb,nrhs;
   int i,j;

   m = 5;
   n = 3;
   nrhs = 2;
   lda = 3;
   ldb = 2;

   info = LAPACKE_dgels(LAPACK_ROW_MAJOR,'N',m,n,nrhs,*a,lda,*b,ldb);

   for(i=0;i<n;i++)
   {
      for(j=0;j<nrhs;j++)
      {
         printf("%lf ",b[i][j]);
      }
      printf("\n");
   }
   return(info);
}

int main()
{
  dene();
  double L, jacob, dt;
  L = 200;
  essential params;

  params.nvar   = 1;
  params.porder = 3; // 0 || 1 || 2 || 3
  params.dt     = 0.075;
  params.nelem  = 100;
  params.maxIte = 1000;
  params.nRK = 4; //1 || 4
  params.columnL = params.nvar*params.nelem;
  params.jacob  = L/params.nelem/2;
  params.nU = (params.porder+1)*params.nvar*params.nelem;


  std::cout << "polynomial order is set " << params.porder << std::endl;


  double *soln_coords, *weights;
  set_solnPoints(params.porder, &soln_coords, &weights);

  double *lagrDerivs;
  set_lagrangeDerivs(params.porder, soln_coords, &lagrDerivs);
  
  double *lagrInterL, *lagrInterR;
  set_lagrangeInterCoeffs(params.porder, soln_coords, &lagrInterL, &lagrInterR);

  double *hL, *hR;
  set_correctionDerivs(params.porder, soln_coords, &hL, &hR);


  // solution arrays, array of structures
  double *u, *f, *u_LR, *f_LR;
  u = new double [(params.porder+1)*params.nelem*params.nvar];
  f = new double [(params.porder+1)*params.nelem*params.nvar];
  u_LR = new double [2*params.nelem*params.nvar];
  f_LR = new double [2*params.nelem*params.nvar];

  // 2N storage RK
  double *old_u;
  old_u = new double [(params.porder+1)*params.nelem*params.nvar];

  // implicit matrix
  double *bigA;
  bigA = new double [params.nU*params.nU];
  for ( int i = 0; i < params.nU*params.nU; i++ ) bigA[i] = 0;
  double *invA;
  invA = new double [params.nU*params.nU];
  for ( int i = 0; i < params.nU*params.nU; i++ ) invA[i] = 0;
  double *IA;
  IA = new double [params.nU*params.nU];
  for ( int i = 0; i < params.nU*params.nU; i++ ) IA[i] = 0;
  double *RHS; // right hand side
  RHS = new double [params.nU];
  for ( int i = 0; i < params.nU; i++ ) RHS[i] = 0;

  double *A;
  A = new double [(params.porder+1)*(params.porder+1)];
  int N = params.porder+1;
  int *IPIV = new int[N+1];
  int LWORK = N*N;
  double *WORK = new double[LWORK];
  int INFO;

  for ( int i = 0; i < params.porder+1; i++ )
  {
    for ( int j = 0; j < params.porder+1; j++ )
    {
      A[i*(params.porder+1)+j] = lagrDerivs[i*(params.porder+1)+j];
      if ( i == j ) A[i*(params.porder+1)+j] += 1/params.dt;
    }
  }

  //dgetrf_(&N,&N,A,&N,IPIV,&INFO);
  //dgetri_(&N,A,&N,IPIV,WORK,&LWORK,&INFO);

  INFO = LAPACKE_dgetrf(LAPACK_ROW_MAJOR,N,N,A,N,IPIV);
  std::cout << INFO << "\n";
  INFO = LAPACKE_dgetri(LAPACK_ROW_MAJOR,N,A,N,IPIV);//,WORK,&LWORK);
  std::cout << INFO << "\n";
  delete IPIV;
  delete WORK;
  for ( int i = 0; i < params.porder+1; i++ )
  {
    for ( int j = 0; j < params.porder+1; j++ )
    {
      std::cout << A[i*(params.porder+1)+j] << " ";
    }
    std::cout << "\n";
  }

  std::cout << "construct implicit matrix\n";
  int loc;
  for ( int i = 0; i < params.columnL; i++ )
  {
    for ( int j = 0; j < params.porder+1; j++ )
    {
      for ( int k = 0; k < params.porder+1; k++ )
      {
        loc = (i*(params.porder+1)+j)*params.nU+i*(params.porder+1)+k;
        bigA[loc] += lagrDerivs[j*(params.porder+1)+k]/params.jacob
                   - lagrInterL[k]*hL[j]/params.jacob;// - 0.5*lagrInterR[k]*hR[j];
        if ( i == 0 ) bigA[loc+(params.columnL-1)*(params.porder+1)] += lagrInterR[k]*hL[j]/params.jacob;
        else bigA[loc-(params.porder+1)] += lagrInterR[k]*hL[j]/params.jacob;
        if ( j == k ) bigA[loc] += 1/params.dt;
      }
    }
  }
  N = params.nU;
  LWORK = N*N;
  IPIV = new int[N+1];
  WORK = new double[LWORK];
  
  std::ofstream Amatrix;
  Amatrix.open("Amatrix");
  for ( int i = 0; i < params.nU; i++ )
  {
    for ( int j = 0; j < params.nU; j++ )
    {
      invA[i*params.nU+j] = bigA[i*params.nU+j];
      Amatrix << bigA[i*params.nU+j] << " ";
    }
    Amatrix << "\n";
  }

  std::cout << "bigA\n";
  INFO = LAPACKE_dgetrf(LAPACK_ROW_MAJOR,N,N,bigA,N,IPIV);
  std::cout << INFO << "\n";
  INFO = LAPACKE_dgetri(LAPACK_ROW_MAJOR,N,bigA,N,IPIV);//,WORK,&LWORK);
  std::cout << INFO << "\n";
  delete IPIV;
  delete WORK;

  Amatrix << "\n";
  Amatrix << "\n";
  for ( int i = 0; i < params.nU; i++ )
  {
    for ( int j = 0; j < params.nU; j++ )
    {
      Amatrix << bigA[i*params.nU+j] << " ";
    }
    Amatrix << "\n";
  }

  //double one, zero;
  //one = 1.0;
  //zero = 0.0;
  lapack_complex_float one,zero;
  //one = lapack_make_complex_float(1.0,0.0);
  //zero = lapack_make_complex_float(0.0,0.0);

  //cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, &one, bigA, N, invA, N, &zero, IA, N);

  Amatrix << "\n";
  Amatrix << "\n";
  for ( int i = 0; i < params.nU; i++ )
  {
    for ( int j = 0; j < params.nU; j++ )
    {
      Amatrix << IA[i*params.nU+j] << " ";
    }
    Amatrix << "\n";
  }

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

  double error = 0;
  double x, sigma=20, pi=3.141592653589793, mu=0;

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
  //double x, sigma=20, pi=3.141592653589793, mu=0;
  std::cout << "Gaussian bump with sigma = " << sigma << " and mu = " << mu << "\n";
  for ( int i = 0; i < params.nelem; i++ )
  {
    //x = i*L/nelem-5;
    for ( int j = 0; j < params.porder+1; j++ )
    {
    x = i*L/params.nelem-L/2+L/params.nelem/2 + soln_coords[j]*params.jacob;
    u[j*params.columnL + i] = 1/(sigma*sqrt(2*pi))*exp(-0.5*pow((x-mu)/sigma,2));
    old_u[j*params.columnL + i] = 0;
    }
  }
  }

  double *dummyU = new double[params.porder+1];
  std::cout << "main loop begins\n";
  // MAIN LOOP
  for ( int ite = 0; ite < params.maxIte; ite++ )
  {
    for ( int i = 0; i < params.columnL*(params.porder+1); i++ ) old_u[i] = u[i];
    for ( int i = 0; i < params.nRK; i++ )
    {
      //std::cout << "ite: " << ite << " ";
      //superFunc creates f, u_LR, and f_LR arrays
      superFunc(&params, u, f, u_LR, f_LR, lagrInterL, lagrInterR);

      //call flux function for each colocated uL uR pair of 2 neighbour elements
      // also update f_LR array to have f_I{L/R}-f_D{L/R} as entries
      computeFlux(&params, u_LR, f_LR);

      //update solution
      update(&params, u, f, f_LR, lagrDerivs, hL, hR);
      /*
      for ( int j = 0; j < params.columnL*(params.porder+1); j++ )
      {
        u[j] = old_u[j] + alpha[i]*params.dt*u[j];
      }
      */
      /*
      for ( int j = 0; j < params.columnL; j++ )
      {
        //u[j] = old_u[j] + alpha[i]*params.dt*u[j];
        // A*u;
        double *dummy = new double[params.porder+1];
        for ( int ii = 0; ii < params.porder+1; ii++ ) dummy[ii] = 0;
        for ( int ii = 0; ii < params.porder+1; ii++ )
        {
          for ( int jj = 0; jj < params.porder+1; jj++ )
          {
            dummy[ii] += A[ii*(params.porder+1)+jj]*u[jj*params.columnL+j];
          }
        }
        for ( int k = 0; k < params.porder+1; k++ )
        {
          u[k*params.columnL+j] = old_u[k*params.columnL+j] + alpha[i]*dummy[k];
        }
      }
      */
      
      int loc;
      for ( int ii = 0; ii < params.columnL; ii++ )
      {
        for ( int j = 0; j < params.porder+1; j++ ) dummyU[j] = 0;
        for ( int j = 0; j < params.porder+1; j++ )
        {
          for ( int k = 0; k < params.porder+1; k++ )
          {
            loc = (ii*(params.porder+1)+j)*params.nU+ii*(params.porder+1)+k;
            dummyU[j] += bigA[loc]*u[k*params.columnL + ii];
            if ( ii == 0 ) dummyU[j] += bigA[loc+(params.columnL-1)*(params.porder+1)]*u[k*params.columnL + ii + params.columnL-1];
            else dummyU[j] += bigA[loc-(params.porder+1)]*u[k*params.columnL + ii - 1];
          }
        }
        for ( int k = 0; k < params.porder+1; k++ )
        {
          //u[ii*(params.porder+1)+k] = old_u[ii*(params.porder+1)+k] + alpha[i]*dummyU[k];
          u[k*params.columnL+ii] = old_u[k*params.columnL+ii] + alpha[i]*dummyU[k];
        }
      }
      
      /*
      //dgetrs; requires dgetrf with bigA first
      for ( int j = 0; j < params.columnL; j++ )
      {
        for ( int k = 0; k < params.porder+1; k++ )
        {
        RHS[j*(params.porder+1)+k] = u[k*params.columnL+j];
        }
      }
      INFO = LAPACKE_dgetrs(LAPACK_ROW_MAJOR,'N',N,1,bigA,N,IPIV,RHS,1);
      for ( int j = 0; j < params.columnL; j++ )
      {
        for ( int k = 0; k < params.porder+1; k++ )
        {
          u[k*params.columnL+j] = RHS[j*(params.porder+1)+k];
        }
      }
      for ( int j = 0; j < params.columnL*(params.porder+1); j++ )
      {
        u[j] = old_u[j] + alpha[i]*u[j];
      }
      */
    }
  }


  std::ofstream solution;
  solution.open("solution");
  for ( int i = 0; i < params.nelem; i++ )
  { //print the first var at the 0th node in each element, for second var add +nelem
    //solution << u[0*params.columnL + i] << "\n";
    //for ( int j = 0; j < params.porder+1; j++ )
      //solution << u[j*params.columnL + i] << "\n";
  }

  //error
  error = 0;
  double value;
  std::cout << "Gaussian bump with sigma = " << sigma << " and mu = " << mu << "\n";
  for ( int i = 0; i < params.nelem; i++ )
  {
    //x = i*L/params.nelem-5;
    value = 0;
    for ( int k = 0; k < params.porder+1; k++ )
    {
      value += lagrInterL[k]*u[k*params.columnL + i];
    }
    solution << i*L/params.nelem-L/2 << " " << value << "\n";
    for ( int j = 0; j < params.porder+1; j++ )
    {
    //x = i*L/params.nelem-5 + soln_coords[j]*params.jacob;
    x = i*L/params.nelem-L/2+L/params.nelem/2 + soln_coords[j]*params.jacob;
    solution << x << " " << u[j*params.columnL+i] << "\n";
    //u[j*params.columnL + i] = 1/(sigma*sqrt(2*pi))*exp(-0.5*pow((x-mu)/sigma,2));
    error += pow(u[j*params.columnL + i]-1/(sigma*sqrt(2*pi))*exp(-0.5*pow((x-mu)/sigma,2)) , 2);
    }
    value = 0;
    for ( int k = 0; k < params.porder+1; k++ )
    {
      value += lagrInterR[k]*u[k*params.columnL + i];
    }
    solution << (i+1)*L/params.nelem-L/2 << " " << value << "\n";
  }

  std::cout << "error " << log10(sqrt(error)/sqrt(params.nelem*(params.porder+1))) << "\n";
  std::cout << log10(L/params.nelem) << "\n";

  return 0;
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

void roe_flux(double *q_L, double *q_R, double *fluxI)
{

  double p_L, p_R, H_L, H_R;
  p_L = (gammaVal-1)*(q_L[2]-0.5*pow(q_L[1],2)/q_L[0]);
  p_R = (gammaVal-1)*(q_R[2]-0.5*pow(q_R[1],2)/q_R[0]);
  H_L = (q_L[2]+p_L)/q_L[0];
  H_R = (q_R[2]+p_R)/q_R[0];

  double rho_hat, u_hat, H_hat, a_hat;

  rho_hat = sqrt(q_L[0]*q_R[0]);
  u_hat   = (sqrt(q_L[0])*q_L[1]/q_L[0] + sqrt(q_R[0])*q_R[1]/q_R[0])
           /(sqrt(q_L[0]) + sqrt(q_R[0]));
  H_hat   = (sqrt(q_L[0])*H_L + sqrt(q_R[0])*H_R)
           /(sqrt(q_L[0]) + sqrt(q_R[0]));
  a_hat   = sqrt((gammaVal-1)*(H_hat-0.5*pow(u_hat,2)));

  double lambda[3] = {u_hat, u_hat + a_hat, u_hat - a_hat};
  for ( int i = 0; i < 3; i++ ) lambda[i] = std::abs(lambda[i]);

  double r_eig[3][3];
  r_eig[0][0] = 1; r_eig[0][1] = 1; r_eig[0][2] = 1;
  r_eig[1][0] = u_hat; r_eig[1][1] = u_hat+a_hat; r_eig[1][2] = u_hat-a_hat;
  r_eig[2][0] = 0.5*pow(u_hat,2); r_eig[2][1] = H_hat+a_hat*u_hat; r_eig[2][2] = H_hat-a_hat*u_hat;

  double w0[3];
  w0[0] =-(p_R-p_L)/(2*a_hat*a_hat) + q_R[0] - q_L[0];
  w0[1] = (p_R-p_L)/(2*a_hat*a_hat)
        + (q_R[1] - q_L[1] - u_hat*(q_R[0] - q_L[0]))/(2*a_hat);
  w0[2] = (p_R-p_L)/(2*a_hat*a_hat)
        - (q_R[1] - q_L[1] - u_hat*(q_R[0] - q_L[0]))/(2*a_hat);


  double flux_L[3], flux_R[3], diss[3];

  for ( int i = 0; i < 3; i++ )
  {
    diss[i] = 0;
    for ( int j = 0; j < 3; j++ )
    {
      diss[i] += r_eig[i][j]*lambda[j]*w0[j];
    }
  }

  get_flux(q_L, flux_L);
  get_flux(q_R, flux_R);
  for ( int i = 0; i < 3; i++ ) fluxI[i] = 0.5*(flux_L[i] + flux_R[i] - diss[i]);
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

void superFunc( essential* params, double *u, double *f, double *u_LR, double *f_LR, 
                double *lagrInterL, double *lagrInterR )
{

  //int loc_q[nvar];
  int *loc_q = new int[params->nvar];

  double vel, p;

  for ( int i = 0; i < params->nelem; i++ )
  {
    for ( int j = 0; j < params->nvar; j++ ) loc_q[j] = i+j*params->nelem;
    for ( int k = 0; k < params->nvar; k++ )
    {
      u_LR[loc_q[k]] = 0; u_LR[params->columnL + loc_q[k]] = 0;
      f_LR[loc_q[k]] = 0; f_LR[params->columnL + loc_q[k]] = 0;
    }
    for ( int j = 0; j < params->porder+1; j++ )
    {
      for ( int k = 0; k < params->nvar; k++ )
      {
        u_LR[loc_q[k]]                   += u[j*params->columnL + loc_q[k]]*lagrInterL[j];
        u_LR[params->columnL + loc_q[k]] += u[j*params->columnL + loc_q[k]]*lagrInterR[j];
      }
      // euler
      /*
      f[j][loc_q[0]] = u[j][loc_q[1]];
      vel = u[j][loc_q[1]]/u[j][loc_q[0]];
      p = (gammaVal-1)*(u[j][loc_q[2]]-0.5*u[j][loc_q[1]]*vel);
      f[j][loc_q[1]] = u[j][loc_q[1]]*vel + p;
      f[j][loc_q[2]] = (u[j][loc_q[2]]+p)*vel;
      */
      // advection with a=1
      f[j*params->columnL + loc_q[0]] = 1*u[j*params->columnL + loc_q[0]];
      for ( int k = 0; k < params->nvar; k++ )
      {
        f_LR[loc_q[k]]                   += f[j*params->columnL + loc_q[k]]*lagrInterL[j];
        f_LR[params->columnL + loc_q[k]] += f[j*params->columnL + loc_q[k]]*lagrInterR[j];
      }
    }
  }

  return;
}

void computeFlux(essential* params, double *u_LR, double *f_LR)
{

  //int loc_q[nvar];
  //double u_L[nvar], u_R[nvar], f_I[nvar];
  int *loc_q = new int[params->nvar];
  double *u_L = new double[params->nvar];
  double *u_R = new double[params->nvar];
  double *f_I = new double[params->nvar];

  for ( int i = 0; i < params->nelem-1; i++ )
  {
    for ( int j = 0; j < params->nvar; j++ ) loc_q[j] = i+j*params->nelem;
    for ( int j = 0; j < params->nvar; j++ )
    {
      u_R[j] = u_LR[1*params->columnL + loc_q[j]];
      u_L[j] = u_LR[loc_q[j]+1]; //left of the next element
    }
    //roe_flux(u_R, u_L, f_I); // normal direction to right
    //lax_friedrich
    f_I[0] = u_R[0]; //pure upwind
    for ( int j = 0; j < params->nvar; j++ )
    {
      f_LR[1*params->columnL + loc_q[j]] = f_I[j] - f_LR[1*params->columnL + loc_q[j]];
      f_LR[loc_q[j]+1] = f_I[j] - f_LR[loc_q[j]+1]; //left of the next element
    }
  }
  //for one variable advection periodic;
  u_R[0] = u_LR[params->columnL + params->nelem-1]; //very right
  u_L[0] = u_LR[0]; //very left
  f_I[0] = u_R[0];
  f_LR[params->columnL + params->nelem-1] = f_I[0] - f_LR[params->columnL + params->nelem-1];
  f_LR[0] = f_I[0] - f_LR[0];
  return;
}

void update( essential *params, double *u, double *f, double *f_LR, 
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
      //dummy[j] += hL[j]*f_LR[0][i] + hR[j]*f_LR[1][i];
      dummy[j] += hL[j]*f_LR[i] + hR[j]*f_LR[params->columnL+i];
    }
    //for ( int j = 0; j < params->porder+1; j++ ) f[j][i] = dummy[j];
    //update
    for ( int j = 0; j < params->porder+1; j++ )
      //u[j*params->columnL+i] -= dummy[j]*params->dt/params->jacob;
      u[j*params->columnL+i] = -dummy[j]/params->jacob;
  }

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



