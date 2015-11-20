torch.setdefaulttensortype('torch.FloatTensor')


require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'xlua'
require 'gnuplot'


LSTM = {}
LSTM.__index = LSTM 


function LSTM.new(loader, params)
    -- class metatable stuff
    local self = {}
    setmetatable(self, LSTM)

    -- require some utilities needed
    self.utils = require 'misc'

    -- copy all the parameters
    self.opt = params

    -- set the loader
    self.loader = loader

    return self
end


function createLSTM(input_size, rnn_size)
    -- private function to build the main LSTM architecture
    -- on top of this the top layer and criterion are supposed to be build

    local inputs = {}

    -- inputs: x, previous cell state, previous hidden state
    table.insert(inputs, nn.Identity()())
    table.insert(inputs, nn.Identity()())
    table.insert(inputs, nn.Identity()())

    local x = inputs[1]
    local prev_c = inputs[2]
    local prev_h = inputs[3]

    -- new input sum. connections: input --> hidden and hidden --> hidden
    local i2h = nn.Linear(input_size, 4 * rnn_size)(x)
    local h2h = nn.Linear(rnn_size, 4 * rnn_size)(prev_h)
    local preactivations = nn.CAddTable()({i2h, h2h})

    -- input
    local in_chunk = nn.Narrow(2, 3 * rnn_size + 1, rnn_size)(preactivations)
    local in_transform = nn.Tanh()(in_chunk)

    -- gates
    local pre_sigmoid_chunk = nn.Narrow(2, 1, 3 * rnn_size)(preactivations)
    local all_gates = nn.Sigmoid()(pre_sigmoid_chunk)
    local in_gate = nn.Narrow(2, 1, rnn_size)(all_gates)
    local forget_gate = nn.Narrow(2, rnn_size + 1, rnn_size)(all_gates)
    local out_gate = nn.Narrow(2, 2 * rnn_size + 1, rnn_size)(all_gates)

    -- next carrousel state: transform current cell and gated out
    local next_c = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate, in_transform})
    })
    local c_transform = nn.Tanh()(next_c)
    local next_h = nn.CMulTable()({out_gate, c_transform})

    local outputs = {}
    table.insert(outputs, next_c)
    table.insert(outputs, next_h)

    return nn.gModule(inputs, outputs)
end


function LSTM.saveModel(self, epoch)
    print('Checkpointing. Calculating validation accuracy..')
    local val_acc = 1 - self:validate()
    local savefile = string.format('%s/%s_epoch%.2f_%.4f.t7', self.opt.checkpoint_dir, self.opt.savefile, epoch, val_acc)
    print('Saving checkpoint to ' .. savefile .. '\n')
    local checkpoint = {}
    checkpoint.opt = self.opt
    checkpoint.protos = protos
    torch.save(savefile, checkpoint)
end



function LSTM.createRNN(self, input_size, output_size)
    -- create the model. Feed one input at a time (one value on the time series)
    self.protos = {}
    self.protos.lstm = createLSTM(input_size,self.opt.rnn_size)
    self.protos.top = nn.Sequential()
    self.protos.top:add(nn.Linear(self.opt.rnn_size, output_size))
    self.protos.criterion = nn.MSECriterion()

    -- clone states the proto LSTM
    if not self.protos.clones then
        self.protos.clones = {}
        self.protos.clones['lstm'] = self.utils.clone_many_times(self.protos.lstm, self.opt.window_size)
    end

    -- init the hidden state of the network
    print('Initiating hidden state...')
    self.init_state = {}
    local h_init = torch.zeros(self.opt.batch_size, self.opt.rnn_size)
    table.insert(self.init_state, h_init:clone())
    table.insert(self.init_state, h_init:clone())

    self.init_state_global = self.utils.clone_list(self.init_state)

    -- combine all the self.params and do random inicialization
    self.params, self.grad_params = self.utils.combine_all_parameters(self.protos.lstm, self.protos.top)
    print('Parameters: ' .. self.params:size(1))
    self.params:uniform(-0.08, 0.08)
end



function LSTM.train(self)

    ------------------- evalutation function enclosure -------------------
    local function feval(x)

        if x ~= self.params then
            self.params:copy(x)
        end
        self.grad_params:zero()

        -- get training time series
        -- 1D: batch size (z), 2D: num features (y), 3D: time dimension (x)
        input,y = self.loader:nextTrain()

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

        -- forward through the last time step
        prediction = self.protos.top:forward(lst[#lst])

        -- forward through the criterion
        loss = self.protos.criterion:forward(prediction, y)


        ------------ backward pass ------------
        -- loss and soft-max layer backward pass
        dloss = self.protos.criterion:backward(prediction, y)
        doutput_t = self.protos.top:backward(lst[#lst], dloss)

        -- init the state of the lstm backward pass through time
        drnn_state = {[self.opt.window_size] = self.utils.clone_list(self.init_state, true)}
        drnn_state[self.opt.window_size][2] = doutput_t

        -- backward pass through time
        for t = self.opt.window_size, 1, -1 do
            dlst = self.protos.clones.lstm[t]:backward({input:select(input:dim(), t), unpack(rnn_state[t-1])}, drnn_state[t])
            drnn_state[t-1] = {}
            table.insert(drnn_state[t-1], dlst[2])
            table.insert(drnn_state[t-1], dlst[3])
        end

        -- clamp params to avoid the vanishing or exploding gradient problems 
        _, lstm_dparams = self.protos.lstm:getParameters()
        lstm_dparams:clamp(-5, 5)

        return loss, self.grad_params
    end
    ------------------- evaluation function enclosure -------------------


    -- optimization state & params
    local optim_state = {learningRate = self.opt.learning_rate, alpha = self.opt.decay_rate}

    losses = {}
    lloss = 0

    local num_batches = self.loader.train_size / self.opt.batch_size
    local iterations = self.opt.max_epochs * num_batches

    -- iterate for all the epochs
    for i = 1, iterations do

        local epoch = i / num_batches

        _,local_loss = optim.rmsprop(feval, self.params, optim_state)
        losses[#losses + 1] = local_loss[1]
        lloss = lloss + local_loss[1]

        xlua.progress(i, iterations)

        -- print and garbage collect every now and then
        if i%10 == 0 then
            -- print('epoch ' .. epoch .. ' loss ' .. lloss / 10)
            lloss = 0
            collectgarbage()
        end

        -- learning rate decay
        if i % num_batches == 0 and opt.learning_rate_decay < 1 then
            if epoch >= opt.learning_rate_decay_after then
                local decay_factor = opt.learning_rate_decay
                optim_state.learningRate = optim_state.learningRate * decay_factor -- decay it
                print('decayed learning rate by a factor ' .. decay_factor .. ' to ' .. optim_state.learningRate)
            end
        end

        -- checkpoint: saving model
        if i % opt.save_every == 0 or i == iterations then
            self:saveModel(epoch)
        end        

    end

    tensored_loss = torch.zeros(#losses)
    for i,l in ipairs(losses) do
        tensored_loss[i] = l
    end

    gnuplot.figure()
    gnuplot.plot({'loss evolution', tensored_loss, 'lines ls 1'})
end


function LSTM.validate(self, draw)

    ------------------- evaluation function enclosure -------------------
    local function feval_val()
        -- get time series data
        x,y = self.loader:nextValidation()

        rnn_state = {[0] = self.init_state_global}

        -- go through all the time series
        for t = 1, self.opt.window_size do
            local input = x:select(x:dim(),t)
            if input:dim() == 1 then input = input:reshape(1,input:size(1)):t() end

            lst = self.protos.clones.lstm[t]:forward{input, unpack(rnn_state[t-1])}
            rnn_state[t] = {}
            for i = 1, #self.init_state do table.insert(rnn_state[t], lst[i]) end
        end

        -- propagate through the top layer the output of the last time-step
        predictions = self.protos.top:forward(lst[#lst])

        -- forward through the criterion
        loss = self.protos.criterion:forward(predictions, y)

        return predictions, y, loss
    end
    ------------------- evaluation function enclosure -------------------


    print('Validating')
    if draw then
        prediction = torch.zeros(self.loader.validation_size*self.opt.batch_size)
        gt = torch.zeros(self.loader.validation_size*self.opt.batch_size)
    end

    for i = 1,self.loader.validation_size do
        xlua.progress(i,self.loader.validation_size)
        preds, targets, err = feval_val()

        if draw then
            for j = 1,preds:size(1) do
                local index = (i-1) * self.opt.batch_size + j
                prediction[index] = preds[j]
                gt[index] = targets[j]
            end
        end
    end

    if draw then
        gnuplot.figure()
        gnuplot.plot({{'targets', gt, 'lines ls 1'},{'predictions', prediction, 'lines ls 2'}})
    end

    return loss
end





