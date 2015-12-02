torch.setdefaulttensortype('torch.FloatTensor')


require 'nn'
require 'cunn'
require 'cutorch'
require 'nngraph'
require 'optim'
require 'xlua'
require 'gnuplot'



local LSTM = torch.class('nn.LSTM', 'nn.Module')
 


function LSTM:__init(opt)

   -- require some utilities needed
    self.utils = require '../utils/misc'

    self.opt = opt

    -- set GPU
    self:setDevice(opt.gpuid)

    -- create the network
    self:createLSTM(opt.input_size, opt.num_layers, opt.rnn_size, opt.dropout)

end
 

function LSTM:updateOutput(input)
    -- this function is called by the nn.Module *forward* function.
    -- as the LSTM is wrapped in nngraph module the forwardThroughTime function
    -- also calles the updateOutput for every piece in the LSTM.
    -- In this way there is no need to code this for every component.

    -- ship to GPU if 
    if self.opt.cuda_enabled then input = input:cuda() end

    hidden_states, output = self:forwardThroughTime(input)
    self.output = output
    return output
end
 

function LSTM:updateGradInput(input, gradOutput)
    -- this function is called by the nn.Module *backward* function.
    -- as the LSTM is wrapped in a nngraph module, the backwardThroughTime function
    -- also calles the updateGradInput and accGradParameters for every piece in
    -- the LSTM.
    -- In this way there is no need to code this for every component.

    -- ship to the GPU if required
    if self.opt.cuda_enabled then 
        input = input:cuda()
        gradOutput = gradOutput:cuda()
    end

    _, gradInput = self:backwardThroughTime(input, gradOutput)
    self.gradInput = gradInput
    return gradInput
end
 

-- function LSTM:accGradParameters(input, gradOutput)
--     -- nothing done here by now...
-- end
 

function LSTM:reset()
    self.protos.lstm:getParameters():zero()
end


function LSTM:getParameters()
    return self.protos.lstm:getParameters()
end

function LSTM:parameters()
    return self.protos.lstm:parameters()
end





--------------------------------------------------------------------------
--                          HELPER FUNCTIONS                            --
--------------------------------------------------------------------------

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
    if self.opt.cuda_enabled then
        self.params = self.params:cuda()
        self.protos.lstm = self.protos.lstm:cuda()
        self.init_state = self.init_state:cuda()
        self.init_state_global = self.init_state_global:cuda()
    end


end


function createProtoLSTM(input_size, num_layers, rnn_size, dropout)
    -- private function to build the main LSTM architecture
    -- the top layer and criterion are supposed to be build on top

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