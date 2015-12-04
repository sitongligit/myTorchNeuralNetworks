
--- LSTM nn module. Implements a LSTM network with variable number of layers
-- and number of neurons per layer. (Right now every layer has the same number of neurons)
-- Also provides the convinient forward and backward through time functions.
-- Internally uses *nn* and *nngraph* torch packages. This module inherits 
-- from nn.Module therefore providing the capability to use it as any other nn
-- module and to build nngraphs with it.
-- For more info check:
-- https://github.com/torch/nn
-- and
-- https://github.com/torch/nngraph



torch.setdefaulttensortype('torch.FloatTensor')


require 'nn'
require 'cunn'
require 'cutorch'
require 'nngraph'
require 'optim'
require 'xlua'
require 'gnuplot'



local LSTM = torch.class('nn.LSTM', 'nn.Module')
 

--- Creator function.
-- @param opt Table with at least the following fileds:
-- rnn_size (num lstm cell per layer)
-- num_layers (number of hidden layers)
-- time_steps (by now fixed; sentence lenth)
-- gpuid (values in {-1, 1, 2, ...} = {CPU, GPU1, GPU2, ...})
function LSTM:__init(opt)

   -- require some utilities needed
    self.utils = require '../utils/misc'

    self.opt = opt

    -- set GPU
    self:setDevice(opt.gpuid)

    -- create the network
    self:createLSTM(opt.input_size, opt.num_layers, opt.rnn_size, opt.dropout)

end
 
--- Called by the nn.Module *forward* function.
-- As the LSTM is wrapped in nngraph module the forwardThroughTime function
-- also calles the updateOutput for every piece in the LSTM.
-- In this way there is no need to code this for every component.
-- @param input Input to the LSTM network. Expect a tensor with an example per row.
-- @return A tensor with the output of the network
-- @see forwardThroughTime
function LSTM:updateOutput(input)
    -- ship to GPU if 
    if self.cuda_enabled then input = input:cuda() end

    hidden_states, output = self:forwardThroughTime(input)
    self.output = output
    return output
end
 

--- Called by the nn.Module *backward* function to update the gradients w.r.t the input.
-- As the LSTM is wrapped in a nngraph module, the backwardThroughTime function
-- also calles the updateGradInput and accGradParameters for every piece in
-- the LSTM. In this way there is no need to code this for every component.
-- @param input is the input to the LSTM network. Expectes a tensor with an example per row.
-- @param gradOutput is the gradient of the loss w.r.t the output. Expects a tensor.
-- @return a tensor with the gradient of the loss w.r.t the input.
-- @see backwardThroughTime
function LSTM:updateGradInput(input, gradOutput)

    -- ship to the GPU if required
    if self.cuda_enabled then 
        input = input:cuda()
        gradOutput = gradOutput:cuda()
    end

    _, gradInput = self:backwardThroughTime(input, gradOutput)
    self.gradInput = gradInput

    return gradInput
end
 


 
--- Method to reset (zeroes) the parameters of the network.
function LSTM:reset()
    self.protos.lstm:getParameters():zero()
end

--- Interface to the nn module parameters function. 
-- @return two tensors, one for the flattened learnable parameters and
-- another for the gradients of the energy w.r.t to the learnable parameters.
function LSTM:parameters()
    return self.protos.lstm:parameters()
end


-- SHOULD NOT BE OVERRIDE --
-- --- Function interface to the nn module getParamters function.
-- -- @return a table with the learnable paramters and with the gradients of the network.
-- function LSTM:getParameters()
--     return self.protos.lstm:getParameters()
-- end

-- function LSTM:accGradParameters(input, gradOutput)
--     -- nothing done here by now...
-- end





--------------------------------------------------------------------------
--                            BPTT FUNCTIONS                            --
--------------------------------------------------------------------------

--- Propagates the input through time.
-- @param input is the input to the LSTM network. Expect a tensor with an example per row.
-- @return two tensors. 1st the hidden state for every time step and the 2nd is the output
-- of the last time step.
function LSTM:forwardThroughTime(input)

    ------------ forward pass ------------
    loss = 0
    rnn_state = {[0] = self.init_state_global}

    for t = 1, self.opt.time_steps do
        -- get specific time-step (select yields a batch_size x features matrix)
        local input_t = input:select(input:dim(),t)
        if input_t:dim() == 1 then input_t = input_t:reshape(1,input_t:size(1)):t() end

        -- forward propagate for every every time-step
        -- note the curly braces around the function call (to return a table)
        lst = self.protos.clones.lstm[t]:forward{input_t, unpack(rnn_state[t-1])}
        rnn_state[t] = {}
        for i = 1, #self.init_state do table.insert(rnn_state[t], lst[i]) end
    end

    -- return all time-step states and the output
    return rnn_state, lst[#lst]
end


--- Back propagates the error signal through time.
-- @param input is the input to the LSTM network. Expect a tensor with an example per row.
-- @param delta_output is the error signals to back propagate backwards. Expect a tensor.
-- @return two tensors. 1st the gradient of the hidden state for every time step and the 2nd is the gradient
-- w.r.t. the input of the last time step.
function LSTM:backwardThroughTime(input, delta_output)

    ------------ backward pass ------------
    -- init the state of the lstm backward pass through time
    drnn_state = {[self.opt.time_steps] = self.utils.clone_list(self.init_state, true)}
    drnn_state[self.opt.time_steps][2] = delta_output

    -- backward pass through time
    for t = self.opt.time_steps, 1, -1 do

        gradInput = self.protos.clones.lstm[t]:backward({input:select(input:dim(), t), unpack(rnn_state[t-1])}, drnn_state[t])

        drnn_state[t-1] = {}
        for l,df_di in ipairs(gradInput) do
            -- k == 1 is the gradient of the input (we don't use that by now...)
            if l > 1 then 
                drnn_state[t-1][l-1] = df_di
            end
        end
    end

    -- clamp params to avoid the vanishing or exploding gradient problems 
    _, lstm_dparams = self.protos.lstm:getParameters()
    lstm_dparams:clamp(-5, 5)

    return drnn_state, lstm_dparams
end




--------------------------------------------------------------------------
--                          GPU related METHODS                         --
--------------------------------------------------------------------------

--- Sets the nvidia GPU device if given or CPU otherwise
-- @param gpuid Integer (1 indexed) or -1 if CPU
function LSTM:setDevice(gpuid)
    if gpuid ~= -1 then

        -- set flag
        self.cuda_enabled = true

       -- save setDevice
       if (cutorch.getDevice() ~= gpuid) then
          cutorch.setDevice(gpuid)
       end

    else
        self.cuda_enabled = false
    end
end



--------------------------------------------------------------------------
--                           FACTORY METHODS                            --
--------------------------------------------------------------------------

--- Creates the proto LSTM core module and the unroll over time.
-- @param input_size integer to set the number of inputs to the network
-- @param num_layers integer to set the number of hidden layers of the network 
-- @param rnn_size integer to set the number of LSTM neurons per layer
-- @param dropout real number to set the dropout of the network
function LSTM:createLSTM(input_size, num_layers, rnn_size, dropout)

    -- create the core LSTM network
    self.protos = {}
    self.protos.lstm = createProtoLSTM(input_size, num_layers, rnn_size, dropout)
    -- self.params, self.grad_params = self.protos.lstm:parameters()

    self.params, self.grad_params = self.utils.combine_all_parameters(self.protos.lstm)
    self.params:uniform(-0.08, 0.08)

    -- clone states of the proto LSTM
    self.protos.clones = {}
    self.protos.clones['lstm'] = self.utils.clone_many_times(self.protos.lstm, self.opt.time_steps)

    -- init the hidden state of the network
    self.init_state = {}
    local h_init = torch.zeros(self.opt.batch_size, self.opt.rnn_size)
    for l = 1,self.opt.num_layers do
        table.insert(self.init_state, h_init:clone())
        table.insert(self.init_state, h_init:clone())
    end
    self.init_state_global = self.utils.clone_list(self.init_state)

    -- ship everything to the GPU if required
    if self.cuda_enabled then
        self.params = self.params:cuda()
        self.protos.lstm = self.protos.lstm:cuda()
        self.init_state = self.init_state:cuda()
        self.init_state_global = self.init_state_global:cuda()
    end
end


--- Private function to build the main LSTM architecture
-- the top layer and criterion are supposed to be build on top.
-- @param input_size integer to set the number of inputs to the network
-- @param num_layers integer to set the number of hidden layers of the network 
-- @param rnn_size integer to set the number of LSTM neurons per layer
-- @param dropout real number to set the dropout of the network
-- @return LSTM ngraph module wrapping all the components.
function createProtoLSTM(input_size, num_layers, rnn_size, dropout)


    local inputs = {}
    local outputs = {}

    -- inputs: x
    table.insert(inputs, nn.Identity()())
    for l = 1, num_layers do
        -- previous cell and hidden state for every layer l
        table.insert(inputs, nn.Identity()())       -- x(t)
        table.insert(inputs, nn.Identity()())       -- h(t-1)
    end

    for l = 1, num_layers do

        -- get previous cell and hidden states for this layer
        local prev_c = inputs[l*2]
        local prev_h = inputs[l*2+1]

        -- inputs for this layer
        local x
        local input_size_l
        if l == 1 then
            x = inputs[1]
            input_size_l = input_size
        else
            x = outputs[(l-1)*2]
            input_size_l = rnn_size
            -- dropout if applicable
            if dropout > 0 then
                x = nn.Dropout(dropout)(x)
            end
        end

        -- new input sum
        -- connections: input --> hidden & hidden --> hidden
        local i2h = nn.Linear(input_size_l, 4 * rnn_size)(x):annotate{name='i2h_'..l}
        local h2h = nn.Linear(rnn_size, 4 * rnn_size)(prev_h):annotate{name='h2h_'..l}
        local preactivations = nn.CAddTable()({i2h, h2h})

        -- get separate pre-activations to gates
        local reshaped_preactivations = nn.Reshape(4, rnn_size)(preactivations)
        local ig, fg, og, it = nn.SplitTable(2)(reshaped_preactivations):split(4)
        
        -- decode gates
        local in_gate = nn.Sigmoid()(ig)
        local forget_gate = nn.Sigmoid()(fg)
        local out_gate = nn.Sigmoid()(og)

        -- input
        local in_transform = nn.Tanh()(it)

        -- next carrousel state: transform current cell and gated out
        local next_c = nn.CAddTable()({
            nn.CMulTable()({forget_gate, prev_c}),
            nn.CMulTable()({in_gate, in_transform})
        })
        local c_transform = nn.Tanh()(next_c)
        local next_h = nn.CMulTable()({out_gate, c_transform})

        table.insert(outputs, next_c)
        table.insert(outputs, next_h)

    end     -- end layer iteration

    return nn.gModule(inputs, outputs)
end


return LSTM