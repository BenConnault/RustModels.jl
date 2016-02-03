using RustModels
using ProfileView
import Markov.rsm



include("../test/toymodels.jl")


srand(425)
models=classicaltoys()
dirm=models[1]
theta0=[0.6,0.8]
coef!(dirm,theta0)
data=rand(dirm,100,100)

Profile.clear()
@profile mle(models[3],data);
ProfileView.view()
