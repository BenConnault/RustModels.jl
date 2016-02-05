module bench

using RustModels
import HiddenMarkovModels.rsm

include("../test/toymodels.jl")

#small classical Rust model
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

#small hidden Rust model
function bench2()
	srand(425)
	models=hiddentoys()
	dirm=models[1]
	# theta0 such that lambda0=[5.,15] and q0 roughly [0.750 0.250; 0.310 0.690]
	theta0=[5.,15,1.1,-.8]
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


end #module