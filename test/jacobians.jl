using Calculus

dirm,sirm,dfrm,sfrm=hiddentoys()

# theta0 such that lambda0=[5.,15] and q0 roughly [0.750 0.250; 0.310 0.690]
theta0=[5.,15,1.1,-.8]

coef_jac!(sfrm,theta0)

#find non-zeros m's (in linear indices)
nzv=find(x-> !(x≈0),sfrm.hiddenlayer.m)

tol=1e-5

function checkjac(k)
	dx,dy=size(sfrm.hiddenlayer.mu)
	dk,da=size(sfrm.rustcore.p)
	function ff(theta)
		coef!(sfrm,theta0)
		sfrm.hiddenlayer.m[k]		
	end
	coef_jac!(sfrm,theta0)
	ix,iy,jx,jy=ind2sub((dx,dy,dx,dy),k)
	norm(vec(Calculus.gradient(ff,theta0))-vec(slice(sfrm.hiddenlayer.mjac,ix,iy,jx,jy,:))) < tol
end

for i=1:5
	@test checkjac(rand(nzv))
end


