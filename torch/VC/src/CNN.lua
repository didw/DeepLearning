-- CNN module
torch.setdefaulttensortype('torch.FloatTensor')

local ParamBank = require 'ParamBank'

local CNN = {}

function CNN.cnn(opt)

  SpatialConvolution = nn.SpatialConvolutionMM
  SpatialMaxPooling = nn.SpatialMaxPooling
  ReLU = nn.ReLU
  SpatialSoftMax = nn.SpatialSoftMax

  local net = nn.Sequential()
  print('==> init a small overfeat network')
  net:add(SpatialConvolution(3, 96, 11, 11, 4, 4)) -- 231x231 -> 56x56
  net:add(ReLU())
  net:add(SpatialMaxPooling(2, 2, 2, 2)) -- 56x56 -> 16x16
  net:add(SpatialConvolution(96, 256, 5, 5, 1, 1)) -- 16x16 -> 12x12
  net:add(ReLU())
  net:add(SpatialMaxPooling(2, 2, 2, 2)) -- 12x12 -> 6x6
  net:add(SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1)) -- 6x6 -> 6x6
  net:add(ReLU())
  net:add(SpatialConvolution(512, 1024, 3, 3, 1, 1, 1, 1)) 
  net:add(ReLU())
  net:add(SpatialConvolution(1024, 1024, 3, 3, 1, 1, 1, 1)) -- 12x12 -> 12x12
  net:add(ReLU())
  net:add(SpatialMaxPooling(2, 2, 2, 2)) -- 12x12 -> 6x6
  net:add(SpatialConvolution(1024, 3072, 6, 6, 1, 1)) -- 6x6 -> 1x1
  net:add(ReLU())
  net:add(SpatialConvolution(3072, 4096, 1, 1, 1, 1))
  net:add(ReLU())
  net:add(SpatialConvolution(4096, 1000, 1, 1, 1, 1))
  net:add(nn.View(1000))
  print(net)


  -- init file pointer
  print('==> overwrite network parameters with pre-trained weigts')
  ParamBank:init("net_weight_0")
  ParamBank:read(        0, {96,3,11,11},    net:get(1).weight)
  ParamBank:read(    34848, {96},            net:get(1).bias)
  ParamBank:read(    34944, {256,96,5,5},    net:get(4).weight)
  ParamBank:read(   649344, {256},           net:get(4).bias)
  ParamBank:read(   649600, {512,256,3,3},   net:get(7).weight)
  ParamBank:read(  1829248, {512},           net:get(7).bias)
  ParamBank:read(  1829760, {1024,512,3,3},  net:get(9).weight)
  ParamBank:read(  6548352, {1024},          net:get(9).bias)
  ParamBank:read(  6549376, {1024,1024,3,3}, net:get(11).weight)
  ParamBank:read( 15986560, {1024},          net:get(11).bias)
  ParamBank:read( 15987584, {3072,1024,6,6}, net:get(14).weight)
  ParamBank:read(129233792, {3072},          net:get(14).bias)
  ParamBank:read(129236864, {4096,3072,1,1}, net:get(16).weight)
  ParamBank:read(141819776, {4096},          net:get(16).bias)
  ParamBank:read(141823872, {1000,4096,1,1}, net:get(18).weight)
  ParamBank:read(145919872, {1000},          net:get(18).bias)

  -- close file pointer
  ParamBank:close()

  return net
end

return CNN