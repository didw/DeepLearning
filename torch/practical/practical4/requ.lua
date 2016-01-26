require 'nn'

local ReQU = torch.class('nn.ReQU', 'nn.Module')

function ReQU:updateOutput(input)
  -- TODO
  self.output:resizeAs(input):copy(input)
  -- z = f(x) = x^2 if x > 0
  --            0   otherwise
  self.output[torch.lt(input,0)] = 0
  self.output:pow(2)
  return self.output
end

function ReQU:updateGradInput(input, gradOutput)
  -- TODO
  self.gradInput:resizeAs(gradOutput):copy(gradOutput)
  -- dz_dx = 2x
  self.gradInput[torch.lt(input,0)] = 0
  self.gradInput:cmul(input)
  self.gradInput:mul(2)
  return self.gradInput
end

