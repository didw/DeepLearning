require 'torch'
require 'nn'

paths.dofile('alexnet_for_lstm.lua')
require 'lstm'

local AL, parent = torch.class('nn.AlexnetLstm', 'nn.Module')


function AL:__init(opt)
  
  self.input_size = opt.cropSize
  self.rnn_size = opt.rnnSize;
  self.cnn_size = opt.cnnSize;
  self.num_layers = opt.numLayers;
  self.num_class = opt.nClasses;

  local D, H, C = self.cnn_size, self.rnn_size, self.num_class

  self.net = nn.Sequential()
  self.rnns = {}
  self.net:add(createModel(opt.nGPU))
  for i = 1, self.num_layers do
    local prev_dim = H
    if i == 1 then prev_dim = D end
    local rnn
    rnn = nn.LSTM(prev_dim, H)
    rnn.remember_states = true
    table.insert(self.rnns, rnn)
    self.net:add(rnn)
    self.net:add(nn.Dropout(0.5))
  end

  -- After all the RNNs run, we will have a tensor of shape (N, T, H);
  -- we want to apply a 1D temporal convolution to predict scores for each
  -- vocab element, giving a tensor of shape (N, T, V). Unfortunately
  -- nn.TemporalConvolution is SUPER slow, so instead we will use a pair of
  -- views (N, T, H) -> (NT, H) and (NT, V) -> (N, T, V) with a nn.Linear in
  -- between. Unfortunately N and T can change on every minibatch, so we need
  -- to set them in the forward pass.
  self.view1 = nn.View(1, 1, -1):setNumInputDims(3)
  self.view2 = nn.View(1, -1):setNumInputDims(2)

  self.net:add(self.view1)
  self.net:add(nn.Linear(H, C))
  self.net:add(self.view2)
end


function AL:updateOutput(input)
  local N, T = input:size(1), input:size(2)
  self.view1:resetSize(N * T, -1)
  self.view2:resetSize(N, T, -1)
  return self.net:forward(input)
end


function AL:backward(input, gradOutput, scale)
  return self.net:backward(input, gradOutput, scale)
end


function AL:parameters()
  return self.net:parameters()
end


function AL:resetStates()
  for i, rnn in ipairs(self.rnns) do
    rnn:resetStates()
  end
end


