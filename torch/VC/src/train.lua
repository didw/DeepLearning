require 'torch'
require 'nn'
require 'cunn'
require 'nngraph'
require 'optim'
local VideoMinibatchLoader = require 'VideoMinibatchLoader'
local CNN = require 'CNN'
local LSTM = require 'LSTM'             -- LSTM timestep and utilities
local model_utils=require 'model_utils'
torch.setdefaulttensortype('torch.FloatTensor')


local cmd = torch.CmdLine()
cmd:text()
cmd:text('Training a video action classification model using CNN and LSTM')
cmd:text()
cmd:text('Options')
cmd:option('-datadir','../UCF-101/','directory name where stored video tensors')
cmd:option('-batch_size',2,'number of sequences to train on in parallel')
cmd:option('-seq_length',16,'number of timesteps to unroll to')
cmd:option('-rnn_size',256,'size of LSTM internal state')
cmd:option('-max_epochs',1,'number of full passes through the training data')
cmd:option('-savefile','model_autosave','filename to autosave the model (protos) to, appended with the,param,string.t7')
cmd:option('-save_every',100,'save every 100 steps, overwriting the existing file')
cmd:option('-print_every',10,'how many steps/minibatches between printing out the loss')
cmd:option('-cnn_size',1000,'classification size of CNN')
cmd:option('-class_size',10,'class size')
cmd:option('-train_type','nn','nn | cunn')
cmd:option('-threads',4,'nb of threads')
cmd:option('-seed',123,'torch manual random number generator seed')
cmd:text()

-- parse input params
local opt = cmd:parse(arg)

torch.setnumthreads(opt.threads)
-- preparation stuff:
torch.manualSeed(opt.seed)
opt.savefile = cmd:string(opt.savefile, opt,
  {save_every=true, print_every=true, savefile=true, vocabfile=true, datafile=true})
  .. '.t7'

local loader = VideoMinibatchLoader.create(
        opt.datadir, opt.batch_size, opt.seq_length, opt.class_size)
local cnn_size = opt.cnn_size  -- the number of distinct characters

-- define model prototypes for ONE timestep, then clone them
--
local protos = {}
protos.cnn       = CNN.cnn(opt)
protos.connect   = nn.Sequential():add(nn.Linear(opt.cnn_size, opt.rnn_size))
-- lstm timestep's input: {x, prev_c, prev_h}, output: {next_c, next_h}
protos.lstm      = LSTM.lstm(opt)
protos.softmax   = nn.Sequential():add(nn.Linear(opt.rnn_size, opt.class_size)):add(nn.LogSoftMax())
protos.criterion = nn.ClassNLLCriterion()

protos.connect:getParameters():uniform(-0.08, 0.08)
protos.softmax:getParameters():uniform(-0.08, 0.08)
protos.lstm:getParameters():uniform(-0.08, 0.08)

if train_type == 'cunn' then
  protos.cnn:cuda()
  protos.connect:cuda()
  protos.softmax:cuda()
  protos.lstm:cuda()
end

-- put the above things into one flattened parameters tensor
local params, grad_params = model_utils.combine_all_parameters(protos.cnn, protos.lstm, protos.softmax)

-- make a bunch of clones, AFTER flattening, as that reallocates memory
local clones = {}
for name,proto in pairs(protos) do
  print('cloning '..name)
  clones[name] = model_utils.clone_many_times(proto, opt.seq_length, not proto.parameters)
end

-- LSTM initial state (zero initially, but final state gets sent to initial state when we do BPTT)
local initstate_c = torch.zeros(opt.batch_size, opt.rnn_size)
local initstate_h = initstate_c:clone()

-- LSTM final state's backward message (dloss/dfinalstate) is 0, since it doesn't influence predictions
local dfinalstate_c = initstate_c:clone()
local dfinalstate_h = initstate_c:clone()

require 'image'
-- do fwd/bwd and return loss, grad_params
function feval(params_)
  if params_ ~= params then
    params:copy(params_)
  end
  grad_params:zero()
  
  ------------------ get minibatch -------------------
  local x, y = loader:next_batch()
  if train_type == 'cunn' then
    x:cuda()
    y:cuda()
  end
  ------------------- forward pass -------------------
  local cnns = {}                  -- input cnns
  local conn = {}            -- input embeddings
  local lstm_c = {[0]=initstate_c} -- internal cell states of LSTM
  local lstm_h = {[0]=initstate_h} -- output values of LSTM
  local predictions = {}           -- softmax outputs
  local loss = 0

  for t=1,opt.seq_length do
    cnns[t] = clones.cnn[t]:forward(x[{{},t}])

    conn[t] = clones.connect[t]:forward(cnns[t])

    -- we're feeding the *correct* things in here, alternatively
    -- we could sample from the previous timestep and embed that, but that's
    -- more commonly done for LSTM encoder-decoder models
    lstm_c[t], lstm_h[t] = unpack(clones.lstm[t]:forward{conn[t], lstm_c[t-1], lstm_h[t-1]})
    predictions[t] = clones.softmax[t]:forward(lstm_h[t])

    loss = loss + clones.criterion[t]:forward(predictions[t], y)
  end

  ------------------ backward pass -------------------
  -- complete reverse order of the above
  local dldlstm = {}                              -- d loss / d input embeddings
  local dldcnn  = {}                                  -- d loss / d input embeddings
  local dlstm_c = {[opt.seq_length]=dfinalstate_c}    -- internal cell states of LSTM
  local dlstm_h = {}                                  -- output values of LSTM
  for t=opt.seq_length,1,-1 do
    -- backprop through loss, and softmax/linear
    local doutput_t = clones.criterion[t]:backward(predictions[t], y)
    -- Two cases for dloss/dh_t: 
    --   1. h_T is only used once, sent to the softmax (but not to the next LSTM timestep).
    --   2. h_t is used twice, for the softmax and for the next step. To obey the
    --      multivariate chain rule, we add them.
    if t == opt.seq_length then
      assert(dlstm_h[t] == nil)
      dlstm_h[t] = clones.softmax[t]:backward(lstm_h[t], doutput_t)
    else
      dlstm_h[t]:add(clones.softmax[t]:backward(lstm_h[t], doutput_t))
    end

    -- backprop through LSTM timestep
    dldlstm[t], dlstm_c[t-1], dlstm_h[t-1] = unpack(clones.lstm[t]:backward(
        {conn[t], lstm_c[t-1], lstm_h[t-1]},
        {dlstm_c[t], dlstm_h[t]}
    ))

    dldcnn[t] = clones.connect[t]:backward(cnns[t], dldlstm[t])

    -- backprop through cnn
    clones.cnn[t]:backward(x[{{},t}], dldcnn[t])
  end

  ------------------------ misc ----------------------
  -- transfer final state to initial state (BPTT)
  initstate_c:copy(lstm_c[#lstm_c])
  initstate_h:copy(lstm_h[#lstm_h])

  -- clip gradient element-wise
  grad_params:clamp(-5, 5)

  return loss, grad_params
end

-- optimization stuff
local losses = {}
local optim_state = {learningRate = 1e-1}
local iterations = opt.max_epochs * loader.nbatches
for i = 1, 1000 do
  local _, loss = optim.adagrad(feval, params, optim_state)
  losses[#losses + 1] = loss[1]

  if i % opt.save_every == 0 then
    torch.save(opt.savefile, protos)
  end
  if i % opt.print_every == 0 then
    print(string.format("iteration %4d, loss = %6.8f, loss/seq_len = %6.8f, gradnorm = %6.4e", i, loss[1], loss[1] / opt.seq_length, grad_params:norm()))
  end
end

