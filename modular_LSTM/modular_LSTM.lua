torch.setdefaulttensortype('torch.FloatTensor')


require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'xlua'
require 'gnuplot'


LSTM = {}
LSTM.__index = LSTM 


function createLSTM(input_size, num_layers, rnn_size, dropout)
    -- private function to build the main LSTM architecture
    -- on top of this the top layer and criterion are supposed to be build

    local inputs = {}
    local outputs = {}

    -- inputs: x
    table.insert(inputs, nn.Identity()())
    for l = 1, num_layers do
        -- previous cell and hidden state for every layer l
        table.insert(inputs, nn.Identity()())
        table.insert(inputs, nn.Identity()())
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
        -- connections: input --> hidden and hidden --> hidden
        local i2h = nn.Linear(input_size_l, 4 * rnn_size)(x):annotate{name='i2h_'..l}
        local h2h = nn.Linear(rnn_size, 4 * rnn_size)(prev_h):annotate{name='h2h_'..l}
        local preactivations = nn.CAddTable()({i2h, h2h})

        -- get separate pre-activations to gates
        local reshaped_preactivations = nn.Reshape(4, rnn_size)(preactivations)
        local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped_preactivations):split(4)
        
        -- decode gates
        local in_gate = nn.Sigmoid()(n1)
        local forget_gate = nn.Sigmoid()(n2)
        local out_gate = nn.Sigmoid()(n3)

        -- input
        local in_transform = nn.Tanh()(n4)

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


function LSTM.new(opt_params, input_size)
    -- class metatable stuff
    local self = {}
    setmetatable(self, LSTM)

    -- require some utilities needed
    self.utils = require '../utils/misc'

    -- copy all the parameters
    self.opt = opt_params

    -- create the core LSTM network
    self.protos = {}
    self.protos.lstm = createLSTM(input_size, opt_params.num_layers, opt_params.rnn_size, opt_params.dropout)
    -- self.params, self.grad_params = self.protos.lstm:parameters()

    self.params, self.grad_params = self.utils.combine_all_parameters(self.protos.lstm)
    self.params:uniform(-0.08, 0.08)


    -- clone states of the proto LSTM
    self.protos.clones = {}
    self.protos.clones['lstm'] = self.utils.clone_many_times(self.protos.lstm, self.opt.window_size)

    -- init the hidden state of the network
    self.init_state = {}
    local h_init = torch.zeros(self.opt.batch_size, self.opt.rnn_size)
    for l = 1,self.opt.num_layers do
        table.insert(self.init_state, h_init:clone())
        table.insert(self.init_state, h_init:clone())
    end
    self.init_state_global = self.utils.clone_list(self.init_state)

    return self
end


function LSTM.forward(self, input)

    ------------ forward pass ------------
    loss = 0
    rnn_state = {[0] = self.init_state_global}

    for t = 1, self.opt.window_size do
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
    return rnn_state, lst
end


function LSTM.backward(self, delta_output)

    ------------ backward pass ------------
    -- init the state of the lstm backward pass through time
    drnn_state = {[self.opt.window_size] = self.utils.clone_list(self.init_state, true)}
    drnn_state[self.opt.window_size][2] = delta_output

    -- backward pass through time
    for t = self.opt.window_size, 1, -1 do
        dlst = self.protos.clones.lstm[t]:backward({input:select(input:dim(), t), unpack(rnn_state[t-1])}, drnn_state[t])

        drnn_state[t-1] = {}
        for k,v in ipairs(dlst) do
            -- k == 1 is the gradient of the input (we don't use that by now...)
            if k > 1 then 
                drnn_state[t-1][k-1] = v 
            end
        end
    end

    -- clamp params to avoid the vanishing or exploding gradient problems 
    _, lstm_dparams = self.protos.lstm:getParameters()
    lstm_dparams:clamp(-5, 5)
end


--[[
    REMOVE this function, it makes no sense to save only the core of the network
]]
function LSTM.saveModel(self, val_acc, epoch)
    print('\nCheckpointing. Calculating validation accuracy...')
    print('Accuracy: '..val_acc)
    local savefile = string.format('%s/%s_epoch=%i_acc=%.4f.t7', self.opt.checkpoint_dir, self.opt.savefile, epoch, val_acc)
    print('Saving checkpoint to ' .. savefile .. '\n')
    local checkpoint = {}
    checkpoint.opt = self.opt
    checkpoint.protos = protos
    torch.save(savefile, checkpoint)
end





