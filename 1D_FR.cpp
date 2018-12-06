// 1D Flux Reconstruction Code - 22.11.2018
// Semih Akkurt
// git -master

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

const int nvar=1;
const int porder=2;
const int nelem=1000;

const double gammaVal=1.4;

void set_solnPoints(int p, double (&soln_coords)[porder+1], double (&weights)[porder+1]);
void set_lagrangeDerivs(double (&soln_coords)[porder+1], double (&derivs)[porder+1][porder+1]);
void set_lagrangeInterCoeffs(double (&soln_coords)[porder+1], double (&lagrInterL)[porder+1], double (&lagrInterR)[porder+1]);
void set_correctionDerivs(double (&soln_coords)[porder+1], double (&hL)[porder+1], double (&hR)[porder+1]);
void get_flux(double (&q)[nvar], double (&flux)[nvar]);
void roe_flux(double (&q_L)[nvar], double (&q_R)[nvar], double (&fluxI)[nvar]);

void superFunc( double (&u)[porder+1][nelem*nvar], double (&f)[porder+1][nelem*nvar], 
                double (&u_LR)[2][nelem*nvar], double (&f_LR)[2][nelem*nvar], 
                double (&lagrInterL)[porder+1], double (&lagrInterR)[porder+1] );
void update( double (&u)[porder+1][nelem*nvar], double (&f)[porder+1][nelem*nvar], 
             double (&f_LR)[2][nelem*nvar], 
             double (&lagrDerivs)[porder+1][porder+1], 
             double (&hL)[porder+1], double (&hR)[porder+1], double jacob, double dt );
void computeFlux(double (&u_LR)[2][nelem*nvar], double (&f_LR)[2][nelem*nvar]);

void dene(double* base)
{
std::cout << base[4] << "\n";
}



int main()
{
  std::cout << "polynomial order is set " << porder << std::endl;

  double soln_coords[porder+1], weights[porder+1];
  set_solnPoints(porder, soln_coords, weights);

  double lagrDerivs[porder+1][porder+1];
  set_lagrangeDerivs(soln_coords, lagrDerivs);

  double lagrInterL[porder+1], lagrInterR[porder+1];
  set_lagrangeInterCoeffs(soln_coords, lagrInterL, lagrInterR);
  double hL[porder+1], hR[porder+1];
  set_correctionDerivs(soln_coords, hL, hR);

  //dene(&lagrDerivs[1][2]);

  // solution arrays, array of structures
  double u[porder+1][nelem*nvar], f[porder+1][nelem*nvar];
  double u_LR[2][nelem*nvar], f_LR[2][nelem*nvar];

  int maxite = 100000;
  std::cout << "max ite " << maxite << std::endl;

  double L, jacob, dt;
  L = 10;
  jacob = L/nelem/2;
  dt = 0.0001;//0.0011270166538

  std::cout << "nelem: " << nelem << "\n";
  std::cout << "dx: " << L/nelem << "\n";
  std::cout << "Lenght: " << L << "\n";
  std::cout << "jacob: " << jacob << "\n";
  std::cout << "dt: " << dt << "\n";
  std::cout << "end time: " << maxite*dt << "\n";
  // Initialization
  double rho_0 = 1, u_0 = 0, p_0 = 1;
  double rho_1 = 0.125, u_1 = 0, p_1 = 0.1;

  if ( nvar == 3 )
  {
  double q[nvar], flux[nvar];
  std::cout << "left side\n";
  q[0] = rho_0; q[1] = rho_0*u_0; q[2] = p_0/(gammaVal-1) + 0.5*rho_0*pow(u_0,2);
  for ( int i = 0; i < nelem/2; i++ )
  {
    for ( int j = 0; j < porder+1; j++ ) 
    {
      for ( int k = 0; k < nvar; k++ ) u[j][i+k*nelem] = q[k];
    }
  }
  std::cout << "right side\n";
  q[0] = rho_1; q[1] = rho_1*u_1; q[2] = p_1/(gammaVal-1) + 0.5*rho_1*pow(u_1,2);
  for ( int i = nelem/2; i < nelem; i++ )
  {
    for ( int j = 0; j < porder+1; j++ ) 
    {
      for ( int k = 0; k < nvar; k++ ) u[j][i+k*nelem] = q[k];
    }
  }
  }
  else if ( nvar == 1 )
  {
  //gaussian bump
  std::cout << "Advection solver\n";
  double x, sigma=0.2, pi=3.141592653589793, mu=0;
  std::cout << "Gaussian bump with sigma = " << sigma << " and mu = " << mu << "\n";
  for ( int i = 0; i < nelem; i++ )
  {
    //x = i*L/nelem-5;
    for ( int j = 0; j < porder+1; j++ )
    {
    x = i*L/nelem-5 + soln_coords[j]*jacob;
    u[j][i] = 1/(sigma*sqrt(2*pi))*exp(-0.5*pow((x-mu)/sigma,2));
    }
  }
  }

  std::cout << "main loop begins\n";
  // MAIN LOOP
  for ( int ite = 0; ite < maxite; ite++ )
  {
    //std::cout << "ite: " << ite << " ";
    //superFunc creates f, u_LR, and f_LR arrays
    superFunc(u, f, u_LR, f_LR, lagrInterL, lagrInterR);

    //call flux function for each colocated uL uR pair of 2 neighbour elements
    // also update f_LR array to have f_I{L/R}-f_D{L/R} as entries
    //DO SOMETHING
    computeFlux(u_LR, f_LR);


    // NO BC
    //q[0] = rho_0; q[1] = rho_0*u_0; q[2] = p_0/(gammaVal-1) + 0.5*rho_0*pow(u_0,2);
    //get_flux(q,flux);
    //for ( int i = 0; i < nvar; i++ ) f_LR[0][i*nelem] = 0;//flux[i];
    //q[0] = rho_1; q[1] = rho_1*u_1; q[2] = p_1/(gammaVal-1) + 0.5*rho_1*pow(u_1,2);
    //get_flux(q,flux);
    //for ( int i = 0; i < nvar; i++ ) f_LR[1][(i+1)*nelem-1] = 0;//flux[i];
    //f_LR[0][0] = 0; f_LR[0][nelem] = 0; f_LR[0][2*nelem] = 0;
    //f_LR[1][nelem-1] = 0; f_LR[1][2*nelem-1] = 0; f_LR[1][3*nelem-1] = 0;

    //update solution
    update(u, f, f_LR, lagrDerivs, hL, hR, jacob, dt);
  }

  //double q_L[nvar] = {1, 3, 4};
  //double q_R[nvar] = {2, 4, 5};
  //double fluxI[nvar];

  //roe_flux(q_L, q_R, fluxI);

  //std::cout << fluxI[0] << std::endl;

  //std::cout << u[2][550] << std::endl;


  std::ofstream solution;
  solution.open("solution");
  for ( int i = 0; i < nelem; i++ )
  {
    solution << u[0][i] << "\n";
    //for ( int j = 0; j < porder+1; j++ ) solution << u[j][i] << "\n";
  }

  //error
  double error = 0;
  double x, sigma=0.2, pi=3.141592653589793, mu=0;
  std::cout << "Gaussian bump with sigma = " << sigma << " and mu = " << mu << "\n";
  for ( int i = 0; i < nelem; i++ )
  {
    //x = i*L/nelem-5;
    for ( int j = 0; j < porder+1; j++ )
    {
    x = i*L/nelem-5 + soln_coords[j]*jacob;
    //u[j][i] = 1/(sigma*sqrt(2*pi))*exp(-0.5*pow((x-mu)/sigma,2));
    error += pow(u[j][i]-1/(sigma*sqrt(2*pi))*exp(-0.5*pow((x-mu)/sigma,2)) , 2);
    }
  }

  std::cout << "error " << sqrt(error)/sqrt(nelem*(porder+1)) << "\n";

  return 0;
}






void get_flux(double (&q)[nvar], double (&flux)[nvar])
{
  double u, p;

  flux[0] = q[1];
  u = q[1]/q[0];
  p = (gammaVal-1)*(q[2]-0.5*q[1]*u);
  flux[1] = q[1]*u+p;
  flux[2] = (q[2]+p)*u;
  return;
}

void roe_flux(double (&q_L)[nvar], double (&q_R)[nvar], double (&fluxI)[nvar])
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
  for ( int i = 0; i < nvar; i++ ) lambda[i] = std::abs(lambda[i]);

  double r_eig[nvar][nvar];
  r_eig[0][0] = 1; r_eig[0][1] = 1; r_eig[0][2] = 1;
  r_eig[1][0] = u_hat; r_eig[1][1] = u_hat+a_hat; r_eig[1][2] = u_hat-a_hat;
  r_eig[2][0] = 0.5*pow(u_hat,2); r_eig[2][1] = H_hat+a_hat*u_hat; r_eig[2][2] = H_hat-a_hat*u_hat;

  double w0[nvar];
  w0[0] =-(p_R-p_L)/(2*a_hat*a_hat) + q_R[0] - q_L[0];
  w0[1] = (p_R-p_L)/(2*a_hat*a_hat)
        + (q_R[1] - q_L[1] - u_hat*(q_R[0] - q_L[0]))/(2*a_hat);
  w0[2] = (p_R-p_L)/(2*a_hat*a_hat)
        - (q_R[1] - q_L[1] - u_hat*(q_R[0] - q_L[0]))/(2*a_hat);


  double flux_L[nvar], flux_R[nvar], diss[nvar];

  for ( int i = 0; i < nvar; i++ )
  {
    diss[i] = 0;
    for ( int j = 0; j < nvar; j++ )
    {
      diss[i] += r_eig[i][j]*lambda[j]*w0[j];
    }
  }

  get_flux(q_L, flux_L);
  get_flux(q_R, flux_R);
  fluxI[0] = 0.5*(flux_L[0] + flux_R[0] - diss[0]);
  fluxI[1] = 0.5*(flux_L[1] + flux_R[1] - diss[1]);
  fluxI[2] = 0.5*(flux_L[2] + flux_R[2] - diss[2]);

  return;
}


void set_solnPoints(int p, double (&soln_coords)[porder+1], double (&weights)[porder+1])
{
  if ( p == 0 )
  {
    soln_coords[0] = 0;
    weights[0] = 2.0;
  }
  if ( p == 1 )
  {
    soln_coords[0] =-0.577350269189625764509148780502;
    soln_coords[1] = 0.577350269189625764509148780502;
    weights[0] = 1.0;
    weights[1] = 1.0;
  }
  else if ( p == 2 )
  {//0.0011270166538
    soln_coords[0] =-0.774596669241483377035853079956;
    soln_coords[1] = 0.0;
    soln_coords[2] = 0.774596669241483377035853079956;
    weights[0] = 0.555555555555555555555555555556;
    weights[1] = 0.888888888888888888888888888889;
    weights[2] = 0.555555555555555555555555555556;
  }
  else if ( p == 3 )
  {
    soln_coords[0] =-0.861136311594052575223946488893;
    soln_coords[1] =-0.339981043584856264802665759103;
    soln_coords[2] = 0.339981043584856264802665759103;
    soln_coords[3] = 0.861136311594052575223946488893;
    weights[0] = 0.347854845137453857373063949222;
    weights[1] = 0.652145154862546142626936050778;
    weights[2] = 0.652145154862546142626936050778;
    weights[3] = 0.347854845137453857373063949222;
  }

  std::cout << "solution points are: \n";
  for (int i = 0; i < porder+1; i++)
  {
    std::cout << soln_coords[i] << " ";
  }
  std::cout << std::endl;

  return;
}

void superFunc( double (&u)[porder+1][nelem*nvar], double (&f)[porder+1][nelem*nvar], 
                double (&u_LR)[2][nelem*nvar], double (&f_LR)[2][nelem*nvar], 
                double (&lagrInterL)[porder+1], double (&lagrInterR)[porder+1] )
{

  int loc_q[nvar];

  double vel, p;

  for ( int i = 0; i < nelem; i++ )
  {
    for ( int j = 0; j < nvar; j++ ) loc_q[j] = i+j*nelem;
    //loc_q[0] = i; loc_q[1] = i+nelem; loc_q[2] = i+2*nelem;
    for ( int k = 0; k < nvar; k++ )
    {
      u_LR[0][loc_q[k]] = 0; u_LR[1][loc_q[k]] = 0;
      f_LR[0][loc_q[k]] = 0; f_LR[1][loc_q[k]] = 0;
    }
    for ( int j = 0; j < porder+1; j++ )
    {
      for ( int k = 0; k < nvar; k++ )
      {
        u_LR[0][loc_q[k]] += u[j][loc_q[k]]*lagrInterL[j];
        u_LR[1][loc_q[k]] += u[j][loc_q[k]]*lagrInterR[j];
      }
      /*
      f[j][loc_q[0]] = u[j][loc_q[1]];
      vel = u[j][loc_q[1]]/u[j][loc_q[0]];
      p = (gammaVal-1)*(u[j][loc_q[2]]-0.5*u[j][loc_q[1]]*vel);
      f[j][loc_q[1]] = u[j][loc_q[1]]*vel + p;
      f[j][loc_q[2]] = (u[j][loc_q[2]]+p)*vel;
      */
      f[j][loc_q[0]] = 1*u[j][loc_q[0]]; // advection
      for ( int k = 0; k < nvar; k++ )
      {
        f_LR[0][loc_q[k]] += f[j][loc_q[k]]*lagrInterL[j];
        f_LR[1][loc_q[k]] += f[j][loc_q[k]]*lagrInterR[j];
      }
    }
  }

  return;
}

void computeFlux(double (&u_LR)[2][nelem*nvar], double (&f_LR)[2][nelem*nvar])
{

  int loc_q[nvar];
  double u_L[nvar], u_R[nvar], f_I[nvar];

  for ( int i = 0; i < nelem-1; i++ )
  {
    for ( int j = 0; j < nvar; j++ ) loc_q[j] = i+j*nelem;
    //loc_q[0] = i; loc_q[1] = i+nelem; loc_q[2] = i+2*nelem;
    for ( int j = 0; j < nvar; j++ )
    {
      u_R[j] = u_LR[1][loc_q[j]];
      u_L[j] = u_LR[0][loc_q[j]+1]; //left of the next element
    }
    //roe_flux(u_R, u_L, f_I); // normal direction to right
    //lax_friedrich
    f_I[0] = u_R[0]; //pure upwind
    for ( int j = 0; j < nvar; j++ )
    {
      f_LR[1][loc_q[j]] = f_I[j] - f_LR[1][loc_q[j]];
      f_LR[0][loc_q[j]+1] = f_I[j] - f_LR[0][loc_q[j]+1]; //left of the next element
    }
  }
  //for advection periodic;
  u_R[0] = u_LR[1][nelem-1];
  u_L[0] = u_LR[0][0];
  f_I[0] = u_R[0];
  f_LR[1][nelem-1] = f_I[0] - f_LR[1][nelem-1];
  f_LR[0][0] = f_I[0] - f_LR[0][0];
  return;
}

void update( double (&u)[porder+1][nelem*nvar], double (&f)[porder+1][nelem*nvar], 
             double (&f_LR)[2][nelem*nvar], 
             double (&lagrDerivs)[porder+1][porder+1], 
             double (&hL)[porder+1], double (&hR)[porder+1], double jacob, double dt )
{

  double dummy[porder+1];

  for ( int i = 0; i < nvar*nelem; i++ )
  {
    for ( int j = 0; j < porder+1; j++ ) dummy[j] = 0;
/*
    for ( int k = 0; k < porder+1; k++ )
    {
      for ( int j = 0; j < porder+1; j++ )
      {
        dummy[k] += lagrDerivs[k][j]*f[k][i];
      }
      dummy[k] += hL[k]*f_LR[0][i] + hR[k]*f_LR[1][i];
    }
*/
    for ( int j = 0; j < porder+1; j++ )
    {
      for ( int k = 0; k < porder+1; k++ )
      {
        dummy[j] += lagrDerivs[j][k]*f[k][i];
      }
      dummy[j] += hL[j]*f_LR[0][i] + hR[j]*f_LR[1][i];
    }
    //for ( int j = 0; j < porder+1; j++ ) f[j][i] = dummy[j];
    /*if ( i == 500) for ( int j = 0; j < porder+1; j++ ) std::cout << lagrDerivs[0][j] << " ";
    if ( i == 500) std::cout << "\n";
    if ( i == 500) for ( int j = 0; j < porder+1; j++ ) std::cout << f[j][i] << " ";
    if ( i == 500) std::cout << "\n";
    if ( i == 500) for ( int j = 0; j < porder+1; j++ ) std::cout << dummy[j] << " ";
    if ( i == 500) std::cout << "\n";*/
    //update
    for ( int j = 0; j < porder+1; j++ ) u[j][i] -= dummy[j]*dt/jacob;
  }
/*
  //update
  for ( int j = 0; j < porder+1; j++ )
  {
    for ( int i = 0; i < nvar*nelem; i++ )
    {
      u[j][i] -= f[j][i]*dt/jacob;
    }
  }
*/
  return;
}


void set_lagrangeDerivs(double (&soln_coords)[porder+1], double (&derivs)[porder+1][porder+1])
{

  std::cout << "lagrangeDerivs\n";
  for (int i = 0; i < porder+1; i++)
  {
    for (int j = 0; j < porder+1; j++)
    {
      derivs[i][j] = 0.0;
      for (int k = 0; k < porder+1; k++)
      {
        if ( k==j ) continue;
        double mult = 1.0;
        for (int n = 0; n < porder+1; n++)
        {
          if ( n==j || n==k ) continue;
          mult *= (soln_coords[i]-soln_coords[n])/(soln_coords[j]-soln_coords[n]);
        }
        derivs[i][j] += 1/(soln_coords[j]-soln_coords[k])*mult;
      }
      std::cout << derivs[i][j] << " " ;
    }
    std::cout << "\n";
  }
  return;
}

void set_correctionDerivs(double (&soln_coords)[porder+1], double (&hL)[porder+1], double (&hR)[porder+1])
{


  
  if ( porder == 0 )
  {
    hL[0] = -0.5;
  }
  else if ( porder == 2 )
  {
  for ( int i = 0; i < porder+1; i++ ) hL[i] = 1.5*soln_coords[i] - 0.5;
  }
  else if ( porder == 3 )
  {
  for ( int i = 0; i < porder+1; i++ ) 
    hL[i] = 0.25*(-15.0*pow(soln_coords[i],2)+6.0*soln_coords[i]+3.0);
  }

  //for ( int i = 0; i < porder+1; i++ ) hL[i] = -0.5;
  for ( int i = 0; i < porder+1; i++ ) hR[i] = -hL[porder-i];

/*
  if ( porder==0 )
  {
    hL[0] =-0.5;
    hR[0] = 0.5;
  }
  else if ( porder == 2 )
  {
    for ( int i = 0; i < porder+1; i++ )
    {
      hL[i] = 0.5*(-7.5*pow(soln_coords[i],2) + 3*soln_coords[i] + 1.5 );
      hR[i] = 0.5*( 7.5*pow(soln_coords[i],2) + 3*soln_coords[i] - 1.5 );
    }
  }
  else if ( porder == 3 )
  {
    for ( int i = 0; i < porder+1; i++ )
    {
      hL[i] = 0.25*( 35.0*pow(soln_coords[i],3) - 15.0*pow(soln_coords[i],2)
                   - 15*soln_coords[i] + 3 );
      hR[i] = 0.25*( 35.0*pow(soln_coords[i],3) + 15.0*pow(soln_coords[i],2)
                   - 15*soln_coords[i] - 3 );
    }
  }

*/
  std::cout << "left correction coeffs\n";
  for ( int i = 0; i < porder+1; i++ ) 
  {
    std::cout << hL[i] << " ";
  }
  std::cout << "\n";
  std::cout << "right correction coeffs\n";
  for ( int i = 0; i < porder+1; i++ )
  {
    std::cout << hR[i] << " ";
  }
  std::cout << "\n";

  return;
}

void set_lagrangeInterCoeffs(double (&soln_coords)[porder+1], double (&lagrInterL)[porder+1], double (&lagrInterR)[porder+1])
{

  double left  =-1.0;
  double right = 1.0;

  //left coeffs
  std::cout << "left coeffs\n";
  for (int i = 0; i < porder+1; i++)
  {
    lagrInterL[i] = 1.0;
    for (int j = 0; j < porder+1; j++)
    {
      if ( i == j ) continue;
      lagrInterL[i] *= (left-soln_coords[j])
                      /(soln_coords[i]-soln_coords[j]);
    }
    std::cout << lagrInterL[i] << " ";
  }
  //right coeffs
  std::cout << "\nright coeffs\n";
  for (int i = 0; i < porder+1; i++)
  {
    lagrInterR[i] = 1.0;
    for (int j = 0; j < porder+1; j++)
    {
      if ( i == j ) continue;
      lagrInterR[i] *= (right-soln_coords[j])
                      /(soln_coords[i]-soln_coords[j]);
    }
    std::cout << lagrInterR[i] << " ";
  }
  std::cout << std::endl;
  return;
}



void lagrange_interpolate(double coord, double (&fvalues)[porder+1], double (&soln_coords)[porder+1])
{

return;
}



