from micrograd.engine import Value
from micrograd.nn import MLP

input_dim = 3
layers = [6,3,1]

x = [[0,1,0],[1,0,0],[0,0,1],[1,1,1]]
y = [0,1,0,1]

model = MLP(input_dim,layers)

oldParams = model.parameters()
for data,label in zip(x,y):
    model.train_step(data,label)
newParams = model.parameters()

change =  oldParams-newParams
print(change)
model.zero_grad()