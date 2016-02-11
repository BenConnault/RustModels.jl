module RustModels

import NLsolve
import Distributions: wsample
import StatsBase.StatisticalModel
import IterativeSolvers

importall DynamicDiscreteModels


# importall Markov when I get around to do a Markov.jl
import HiddenMarkovModels: rsm, nsm, z2q, q2z, z2qjac

export sa2y, y2sa, xs2k, k2xs,
	coef!, coef_jac!, rand, loglikelihood, loglikelihood_jac, dim, mle,
		rustmodel, hiddenrustmodel, checkdp



include("rustmodelcore.jl")
include("indexmanipulation.jl")
include("dynamicprogram.jl")
include("classicalrustmodel.jl")
include("hiddenrustmodel.jl")
include("jacobians.jl")
include("hiddenrustmodel-statisticalmodel-interface.jl")



end 
