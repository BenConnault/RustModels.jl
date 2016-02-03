sa2y(dims::Tuple,is::Int,ia::Int)=sub2ind(dims,is,ia)
y2sa(dims::Tuple,iy::Int)=ind2sub(dims,iy)

function sa2y(dims::Tuple,sa::Array{Int,2})
	T=size(sa,1)
	y=zeros(Int,T)
	for t=1:T
		y[t]=sa2y(dims,sa[t,1],sa[t,2])
	end
	y
end

sa2y(dims::Tuple,sa::Array{Array{Int,2},1})=map(sai->sa2y(dims,sai),sa)

function y2sa(dims::Tuple,y::Array{Int,1})
	T=length(y)
	sa=zeros(Int,T,2)
	for t=1:T
		sai=y2sa(dims,y[t])
		sa[t,1]=sai[1]
		sa[t,2]=sai[2]
	end
	sa
end

y2sa(dims::Tuple,y::Array{Array{Int,1},1})=map(yi->y2sa(dims,yi),y)

k2xs=y2sa
xs2k=sa2y
