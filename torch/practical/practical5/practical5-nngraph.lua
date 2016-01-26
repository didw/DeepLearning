require 'nn'
require 'nngraph'

x1 = nn.Identity(10)()
x2 = nn.Identity(10)()
x3 = nn.Linear(20, 10)()

-- to check forward easiler
x3.data.module.weight:fill(1)
x3.data.module.bias:fill(0)

z1 = nn.CMulTable()({x2, x3})
z = nn.CAddTable()({x1, z1})

mlp = nn.gModule({x1, x2, x3}, {z})

inp1 = torch.Tensor(10):fill(1)
inp2 = torch.Tensor(10):fill(2)
inp3 = torch.Tensor(20):fill(1)

print(mlp:forward({inp1, inp2, inp3}))

-- should print 10 array of 41

-- it produce 'simple-graph.png.svg' file
graph.dot(mlp.fg, 'MLP', 'simple-graph.png')

