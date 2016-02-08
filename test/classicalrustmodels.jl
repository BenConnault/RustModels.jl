theta0=rand(2)
dirm,sirm,dfrm,sfrm=classicaltoys()

data=rand(dirm,50)
llk=loglikelihood(dirm,data,theta0)


tol=1e-5
for model in (dirm,sirm,dfrm,sfrm)
	coef!(model,theta0)
	@test checkdp(model) <tol
	@test norm(llk-loglikelihood(model,data)) <tol
end

