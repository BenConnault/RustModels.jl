immutable ClassicalRustModel <: StatisticalModel
	
	#Linear Rust Model u[k,a]=aa[k,a,itheta]*theta + bb[k,a]
	aa::Array{Float64,3}		  	#aa[k,a,itheta]
	bb::Array{Float64,2}		  	#bb[k,a]

	#Initial distribution
	mu::Array{Float64,2}			#mu[k,a]

	core::RustModelCore
end



rustmodel(aa,bb,mu,beta,pi)=ClassicalRustModel(aa,bb,mu,rustmodelcore(beta,pi))

rustmodel(aa,bb,mu,beta,pi,horizonindex::Int)=ClassicalRustModel(aa,bb,mu,rustmodelcore(beta,pi,horizonindex))

checkdp(model::ClassicalRustModel)=checkdp(model.core)

function coef!(model::ClassicalRustModel,theta)
	dk,da=size(model.bb)
	dtheta=length(theta)
	for ia=1:da
		for ik=1:dk
			model.core.u[ik,ia]=model.bb[ik,ia]
			for itheta=1:dtheta
				model.core.u[ik,ia]+=model.aa[ik,ia,itheta]*theta[itheta]
			end
		end
	end
	solvedp!(model.core)
end


#sample (x,y) from a matrix of joint probabilities.
function wsample2(mu)
	dx,dy=size(mu)
	ind2sub((dx,dy),wsample(1:dx*dy,vec(mu)))
end


#an alternative would be to compute the transition matrix for y=(k,a) and
#draw from the Markov chain 
function rand(model::ClassicalRustModel,T::Int)
	dk,da=size(model.bb)
	data=zeros(Int,T,2)
	ka1=wsample2(model.mu)
	data[1,1]=ka1[1]
	data[1,2]=ka1[2]
	for t=2:T
		data[t,1]=wsample(1:dk,reshape(model.core.pi[data[t-1,2]][data[t-1,1],:],dk))
		data[t,2]=wsample(1:da,reshape(model.core.p[data[t,1],:],da))
	end
	data
end


function loglikelihood(model::ClassicalRustModel,data::Array{Int,2})
	T=size(data,1)
	llk=log(model.mu[data[1,1],data[1,2]])
	for t=2:T
		llk+=log(model.core.pi[data[t-1,2]][data[t-1,1],data[t,1]])
		llk+=log(model.core.p[data[t,1],data[t,2]])
	end
	llk/T
end

dim(model::ClassicalRustModel)=size(model.aa,3)