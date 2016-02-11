using Calculus

tol=1e-3

function checkjac(model,param,k)
	dx,dy=size(model.hiddenlayer.mu)
	dk,da=size(model.rustcore.p)
	function ff(theta)
		coef!(model,theta)
		model.hiddenlayer.m[k]		
	end
	coef_jac!(model,param)
	ix,iy,jx,jy=ind2sub((dx,dy,dx,dy),k)
	norm(vec(Calculus.gradient(ff,param))-vec(slice(model.hiddenlayer.mjac,ix,iy,jx,jy,:)))
end

function checkccpjac(model,param,jj)
	dk,da=size(model.rustcore.p)
	function ff(theta)
		coef!(model,theta)
		model.rustcore.p[jj]		
	end
	ccpjac=coef_jac!(model,param)
	ik,ia=ind2sub((dk,da),jj)
	println(round(Calculus.gradient(ff,param)[3:end],5))
	println(round(slice(ccpjac,ik,ia,:),5))
	println()
	norm(vec(Calculus.gradient(ff,param)[3:end])-vec(slice(ccpjac,ik,ia,:)))
end


function checkccpjac2(model,param,ik)
	dk,da=size(model.rustcore.p)
	dtheta=size(model.aa,3)
	function ff(ia)
		function f(theta)
			coef!(model,theta)
			model.rustcore.p[ik,ia]
		end
		f		
	end
	ccpjac,jacq=coef_jac!(model,param)
	println(round(Calculus.gradient(ff(1),param)[dtheta+1:end],5))
	println(round(Calculus.gradient(ff(2),param)[dtheta+1:end],5))
	println(round(slice(ccpjac,ik,1,:),5))
	println(round(slice(ccpjac,ik,2,:),5))
	println()
	# norm(vec(Calculus.gradient(ff,param)[dtheta+1:end])-vec(slice(ccpjac,ik,ia,:)))
end


dx=3
days=200 #breaks at 125
# days=20  #breaks at 125
dk=dx*days*3
da=2
bmodel=hiddentoys(dx,days)[4]
if dx==2
	theta1=theta0
else
	theta1=[5.,15,1.0,0,-1,0,1,-1]
end
coef!(bmodel,theta1)
nzv1=find(x-> !(x≈0),bmodel.hiddenlayer.m)

for i=1:5
	# checkccpjac(bmodel,theta1,rand(1:dk*da))
	checkccpjac2(bmodel,theta1,rand(1:dk))
end

# for i=1:2
# 	@test checkjac(bmodel,theta1,rand(nzv1)) < tol
# end

# rnzv=rand(nzv1)
# @test checkjac(bmodel,theta1,rnzv) < tol
# @test checkjac(bmodel,theta1,rnzv) < tol

coef!(bmodel,theta1)
data=rand(bmodel,10)
ff(theta)=loglikelihood(bmodel,data,theta)
@test norm(vec(Calculus.gradient(ff,theta1))-vec(loglikelihood_jac(bmodel,data,theta1)[2])) < tol

# data=rand(bmodel,600,60)
# ff(theta)=loglikelihood(bmodel,data,theta)
# @test norm(vec(Calculus.gradient(ff,theta1))-vec(loglikelihood_jac(bmodel,data,theta1)[2])) < tol

