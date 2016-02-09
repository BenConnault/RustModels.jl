function coef!(model::HiddenRustModel,theta::Tuple)
	lambda=theta[1]
	q=theta[2]
	u!(model,lambda)
	pi!(model,q)
	# println(@code_warntype(pi!(model,sparse(q))))
	# error()
	solvedp!(model.rustcore)
	m!(model)
end

function coef!(model::HiddenRustModel,theta::Array{Float64,1})
	dx=size(model.hiddenlayer.mu,1)
	dlambda=size(model.aa,3)
	lambda=theta[1:dlambda]
	q=z2q(theta[dlambda+1:end])
	coef!(model,(lambda,q))
end

function coef_jac!(model::HiddenRustModel,theta::Array{Float64,1})
	dx=size(model.hiddenlayer.mu,1)
	dlambda=size(model.aa,3)
	lambda=theta[1:dlambda]
	q=z2q(theta[dlambda+1:end])
	# println(theta[dlambda+1:end])
	u!(model,lambda)
	pi!(model,q)
	solvedp!(model.rustcore)
	m!(model)
	mjac!(model,theta)
	# dy$size(model.hiddenlayer.mu,2)
	# println(round(reshape(model.m,dx*dy,dx*dy),2))
end

loglikelihood(model::HiddenRustModel,data)=loglikelihood(model.hiddenlayer,data)
loglikelihood_jac(model::HiddenRustModel,data)=loglikelihood_jac(model.hiddenlayer,data)
rand(model::HiddenRustModel,n::Int)=rand(model.hiddenlayer,n::Int)
dim(model::HiddenRustModel)=size(model.aa,3)+size(model.hiddenlayer.mu,1)*(size(model.hiddenlayer.mu,1)-1)