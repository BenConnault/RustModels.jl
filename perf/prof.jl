module prof

using RustModels
using ProfileView
import RustModels: rsm,z2q,checkdp



include("../test/toymodels.jl")


srand(425)

# models=classicaltoys()
# theta0=[0.6,0.8]


dirm,sirm,dfrm,sfrm=hiddentoys()
# theta0 such that lambda0=[5.,15] and q0 roughly [0.750 0.250; 0.310 0.690]
theta0=[5.,15,1.1,-.8]


coef_jac!(sfrm,theta0)

Profile.clear()
@profile coef_jac!(sfrm,theta0);
ProfileView.view()

end #module