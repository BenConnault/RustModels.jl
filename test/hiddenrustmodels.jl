dirm,sirm,dfrm,sfrm=hiddentoys()

# theta0 such that lambda0=[5.,15] and q0 roughly [0.750 0.250; 0.310 0.690]
theta0=[5.,15,1.1,-.8]

coef!(sfrm,theta0)
data=rand(sfrm,50,10)


llk=loglikelihood(sfrm,data)

tol=1e-5
for model in (dirm,sirm,dfrm,sfrm)
	coef!(model,theta0)
	@test dim(model)==4
	@test checkdp(model.rustcore) <tol
	# @test norm(llk-loglikelihood(model,data)) <tol
end







# # println("true m: ",round(reshape(sfrm.m,dy*dx,dy*dx),2))



# unq=Array(Int,0)
# bins=zeros(dy,dy)
# for i=1:length(data)
# 	for t=1:length(data[i])-1
# 		bins[data[i][t],data[i][t+1]]+=1
# 	end
# end	
# # println(find(sfrm.m[1,:,1,:].==0)==find(sfrm.m[1,:,2,:].==0))
# println(length(setdiff(find(bins.==0),find(sfrm.m[1,:,1,:].==0))))

# # loglikelihood(hiddenmodel,data,(theta0,q0))

# thetai=[3,10,1.,-1]



# emtwostep(sfrm,data,theta0)

# # println(loglikelihood(sfrm,data,theta0))
