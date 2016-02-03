
y=Array(Array{Int,1},2)
y[1]=[5,7,3]
y[2]=[10]

sa=Array(Array{Int,2},2)
sa[1]=[5 1; 2 2; 3 1]
sa[2]=[5 2]

dims=(5,2)

@test RustModels.sa2y(dims,sa[1])==y[1]
@test RustModels.sa2y(dims,sa)==y
@test RustModels.y2sa(dims,y[1])==sa[1]
@test RustModels.y2sa(dims,y)==sa