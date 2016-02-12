using Calculus


tol=1e-5

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

### first round of tests

sfrm=hiddentoys()[4]
# theta0 such that lambda0=[5.,15] and q0 roughly [0.750 0.250; 0.310 0.690]
theta0=[5.,15,1.1,-.8]
#find non-zeros m's (in linear indices)
#structural zero transition probabilities have trivial zero jacobians ...
coef!(sfrm,theta0)
nzv=find(x-> !(x≈0),sfrm.hiddenlayer.m)


for i=1:5
	@test checkjac(sfrm,theta0,rand(nzv)) < tol
end
rnzv=rand(nzv)
@test checkjac(sfrm,theta0,rnzv) < tol
@test checkjac(sfrm,theta0,rnzv) < tol

coef!(sfrm,theta0)
data=rand(sfrm,10)
ff(theta)=loglikelihood(sfrm,data,theta)
@test norm(vec(Calculus.gradient(ff,theta0))-vec(loglikelihood_jac(sfrm,data,theta0)[2])) < tol

data=rand(sfrm,600,60)
ff(theta)=loglikelihood(sfrm,data,theta)
@test norm(vec(Calculus.gradient(ff,theta0))-vec(loglikelihood_jac(sfrm,data,theta0)[2])) < tol

### second round of tests

dx=3
days=125 
bmodel=hiddentoys(dx,days)[4]
if dx==2
	theta1=theta0
else
	theta1=[5.,15,1.0,0,-1,0,1,-1]
end
coef!(bmodel,theta1)
nzv1=find(x-> !(x≈0),bmodel.hiddenlayer.m)

for i=1:5
	@test checkjac(bmodel,theta1,rand(nzv1)) < tol
end
rnzv=rand(nzv1)
@test checkjac(bmodel,theta1,rnzv) < tol
@test checkjac(bmodel,theta1,rnzv) < tol

coef!(bmodel,theta1)
data=rand(bmodel,10)
ff(theta)=loglikelihood(bmodel,data,theta)
@test norm(vec(Calculus.gradient(ff,theta1))-vec(loglikelihood_jac(bmodel,data,theta1)[2])) < tol

data=rand(bmodel,600,60)
ff(theta)=loglikelihood(bmodel,data,theta)
@test norm(vec(Calculus.gradient(ff,theta1))-vec(loglikelihood_jac(bmodel,data,theta1)[2])) < tol

