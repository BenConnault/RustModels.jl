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


coef!(dirm,theta0)
data=rand(dirm,100,100)

Profile.clear()
@profile mle(dirm,data);
ProfileView.view()

end #module