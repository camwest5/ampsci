#include "AKF_akFunctions.h"

// double fv(double v){
//   if(v<0.1) return 0;
//   else if(v>750.) return 0;
//   else return 1./750.;
// }

double fv(double v, double phi=0);
double fv_au(double v_au, double phi=0);

//******************************************************************************
double fv_au(double v_au, double phi){
  double v = v_au * (FPC::c_SI/FPC::c); //will be in m/s
  v/=1.e3; //convert from m/s -> km/s
  return fv(v,phi);
}

//******************************************************************************
double fv(double v, double phi)
/*
Standard halo model for velocity distribution, in laboratory frame.
 f ~ v^2 exp(-v^2)
Note: distribution for DM particles that cross paths with Earth.
We should have: <v> = 370
XXX - Includes phase - BUT not 100% sure it's correct! XXX XXX XXX
*/
{

  double vesc = 544.; // galactic escape velocity
  double vL0 = 220.; // local frame velocity, average
  double vc  = 220; // circular velocity

  double vearth = 30.; //XXX update! ??

  double vl = vL0 + vearth*sin(phi);

  double Knorm1 = 0.91706*0.942099; //for phi=0
  double Kn = Knorm1*sqrt(M_PI)*vc*vl*370.;
  double A = pow(v,2)/Kn;

  // This is a rough way to enforce normalisation thoughout the year!
  if(phi!=0){ //save some evals if phi=0
    A /= 1. + 0.0587669*sin(phi);
    A /= 1.00137 - 0.00137*cos(2*phi) - 0.00013*sin(phi);//0.000123*sin(phi);
  }
  //Probably, this isn't good enough... XXX

  double arg1 = -pow((v-vl)/vc,2);

  if(v<=0){
    return 0; //just for safety - should never be called with v<0
  }else if(v<vesc-vl){
    double arg2 = -pow((v+vl)/vc,2);
    return A*(exp(arg1)-exp(arg2));
  }else if(v<vesc+vl){
    double arg2 = -pow(vesc/vc,2);
    return A*(exp(arg1)-exp(arg2));
  }else{
    return 0;
  }

}

//******************************************************************************
int main(void){


  double kms_to_au = 1.e3*(FPC::c/FPC::c_SI);
  std::cout<<300.*kms_to_au<<"\n";
  fv_au(300.*kms_to_au,0);

  //return 1;

  //define input parameters
  std::string akfn; //name of K file to read in

  //Open and read the input file:
  {
    std::ifstream ifs;
    ifs.open("dmeXSection.in");
    std::string jnk;
    ifs >> akfn;         getline(ifs,jnk);
    ifs.close();
  }

  // double num=12;
  // std::cout<<"\n{";
  // // for(double phi=0; phi<=2*3.15; phi +=0.5)
  // for(int i=0; i<=2*num; i++)
  // {
  //   double phi = 2*M_PI*i/num;
  // double K=0;
  // double dv = 0.001;
  // for(double v=0; v<900.; v+=dv){
  //   K += fv(v,phi);
  // }
  // std::cout<<"{"<<phi<<","<<K*dv<<"},";
  // }
  // std::cout<<"\n";
  // return 1;


  //Arrays to store results for outputting later:
  std::vector< std::vector< std::vector<float> > > AKenq;
  std::vector<std::string> nklst;
  double qmin,qmax,demin,demax;

  //Read in AK file
  //std::string akfn = "ak-Xe_L6"; //XXX
  std::cout<<"Opening file: "<<akfn<<".bin\n";
  AKF::akReadWrite(akfn,false,AKenq,nklst,qmin,qmax,demin,demax);

  int desteps = (int) AKenq.size();
  int num_states = (int) AKenq[0].size();
  int qsteps =  (int) AKenq[0][0].size();

  if(num_states != (int)nklst.size()) return 1;

  // Do q derivative on i grid:
  double dqonq = log(qmax/qmin)/(qsteps-1); //need to multiply by q
  std::cout<<"\n\n dqonq="<<dqonq<<"\n\n";

  std::cout<<qmin<<" "<<qmax<<" "<<demin<<" "<<demax<<"\n";
  std::cout<<qsteps<<" "<<num_states<<" "<<desteps<<"\n";


  double m=1.; //XXX in GeV
  //convert WIMP masses from GeV to a.u.
  m *= (1.e3/FPC::m_e_MeV);

  bool finite_med = true;
  //Vector mass:
  //NB: have both '0' case (easy), and 'inf' case! (changes coupling const!)
  double mv = 10.;
  //convert from MeV to au:
  mv *= (1./FPC::m_e_MeV);


  //double v=300*1.e-3; // typical v..integrate later...
  double vmax = 900.*kms_to_au;
  double dv   = 1.*kms_to_au; //?? enough? Input!

  //double dE = demin;

  //converts from MeV to au
  double qMeV = (1.e6/(FPC::Hartree_eV*FPC::c));

  //Numberical constant. Note: Inlcudes dqonq
  //NOTE: if alpgha_chi = alpha in the code.. kill c2 !!
  //NB: includes dv - Not good if not integrating over v!?!
  double Aconst = 8*M_PI*FPC::c2*dqonq*dv;

  std::cout<<"\nAconst="<<Aconst<<"\n";
  std::cout<<"\n\n demin="<<demin*FPC::Hartree_eV<<"\n";
  std::cout<<"demax="<<demax*FPC::Hartree_eV<<"\n\n";

  //find clostest dE to given "target"!
  //How to do q-grid integration?

  //XXX XXX XXX the 3s state showing zero... WHY?!?!?!

  //XXX define a v grid. Sort f(v) into Array.
  // This a) speeds it up
  // b) easier function-swapping
  // c) convert to atomic units

  //int ink=2; //loop later! XXX

  for(int ie=0; ie<desteps; ie++){
  for(int ink=0; ink<num_states; ink++){
    double a=0;
    double xe = double(ie)/(desteps-1); //XXX allow for single dE !!
    double dE = demin*pow(demax/demin,xe);
    double vmin = sqrt(dE*2/m);
    double K_nkde_dv=0;
    for(double v=vmin; v<vmax; v+=dv){
      double fvonv = fv_au(v)/v;
      double arg = pow(m*v,2)-2.*m*dE; //may be negative; skip!
      if(arg<0) continue; //also true if fv(v)=0
      double qminus = m*v - sqrt(arg);
      double qplus  = m*v + sqrt(arg);
      //std::cout<<qminus<<"/"<<qplus<<" "<<qminus/qMeV<<"/"<<qplus/qMeV<<"\n";
      double K_nkdev_dq=0;
      for(int iq=0; iq<qsteps; iq++){
        double x = double(iq)/(qsteps-1);
        double q = qmin*pow(qmax/qmin,x);
        if(q<qminus || q>qplus) continue;
        double dq_on_dqonq = q; //devide by dqonq - just a const.
        //Include dqonq in Aconst!
        double Fq = q*dq_on_dqonq; //extra q factor from dqonq (Jacobian)
        if(finite_med) Fq /= pow(q*q+mv*mv,2);
        K_nkdev_dq += fvonv*Fq*AKenq[ie][ink][iq];
      }
      K_nkde_dv += K_nkdev_dq; //dv included in Aconst
    }
    std::cout<<nklst[ink]<<": ";
    std::cout<<ie<<": "<<dE<<"; a="<<K_nkde_dv*Aconst<<"\n";
  }
  }



// for(int ink=0; ink<num_states; ink++){
//   double a_nk=0;
//   for(int iq=0; iq<qsteps; iq++){
//     double x = double(iq)/(qsteps-1);
//     double q = qmin*pow(qmax/qmin,x);
//     if(q<qminus || q>qplus) continue;
//     double dq_on_dqonq = q; //devide by dqonq - just a const.
//     //Include in Acont!
//     double Fq = q*dq_on_dqonq;
//     if(finite_med) Fq /= pow(q*q+mv*mv,2);
//     a_nk+=fvonv*Fq*AKenq[ie][ink][iq];
//   }
//   std::cout<<nklst[ink]<<"; a="<<a_nk*Aconst<<"\n";
//   a+=a_nk;
// }

std::cout<<AKenq[0][2][100]<<"\n";


  //Min/max m_chi
  //min/max m_v (or mu) = if 0, just 1 step
  // Also: for super massive case: linear! 1 step, absorb into x-section

  //Integrate over q,v
  //Clever way to do v integral (or constant v?)
  //nb: there is a v_min!

  //XXX also: integrate over dE (in bins?)


  return 0;
}
