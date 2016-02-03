using Plots
pyplot()

function lkcontour(model,data,theta0,xx,yy)
	thetahat=mle(model,data)
	ff(x,y)=loglikelihood(model,data,[x,y])
	gg=[ff(x,y)::Float64 for x=xx,y=yy]

	qlevels=[0,.25,.5,.75,0.8,0.9,0.95,0.975,0.99,1]
	levels=quantile(vec(gg),qlevels)

	contour(xx,yy,gg,fill=true,levels=levels,color=ColorGradient(:heat,[0,0.98,1]))
	scatter!([theta0[1]],[theta0[2]],c=:green,leg=false)
	scatter!([thetahat[1]],[thetahat[2]],c=:blue,leg=false)
end