
loglikelihood(model::HiddenRustModel,data)=loglikelihood(model.hiddenlayer,data)
loglikelihood_jac(model::HiddenRustModel,data)=loglikelihood_jac(model.hiddenlayer,data)
rand(model::HiddenRustModel,n::Int)=rand(model.hiddenlayer,n::Int)
dim(model::HiddenRustModel)=size(model.aa,3)+size(model.hiddenlayer.mu,1)*(size(model.hiddenlayer.mu,1)-1)