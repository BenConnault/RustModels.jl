using RustModels
import Markov.rsm

include("../test/toymodels.jl")

function bench1()
	srand(425)
	models=classicaltoys()
	dirm=models[1]
	theta0=[0.6,0.8]
	coef!(dirm,theta0)
	data=rand(dirm,100,100)

	timings=zeros(4)
	t=1
	for model in models
		tic()
		@time mle(model,data)
		timings[t]=toq()
		t+=1
	end
	timings
end