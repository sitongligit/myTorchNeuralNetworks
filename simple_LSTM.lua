torch.setdefaulttensortype('torch.FloatTensor')


require 'torch'
require 'nn'
require 'nngraph'
require 'optim'

require 'series'

local utils = require 'misc'



function createLSTM(input_size, rnn_size)

    local inputs = {}

    table.insert(inputs, nn.Identity()())
    table.insert(inputs, nn.Identity()())
    table.insert(inputs, nn.Identity()())

    local x = inputs[1]
    local prev_c = inputs[2]
    local prev_h = inputs[3]

    local i2h = nn.Linear(input_size, 4 * rnn_size)(x)
    local h2h = nn.Linear(rnn_size, 4 * rnn_size)(prev_h)
    local preactivations = nn.CAddTable()({i2h, h2h})

    -- gates
    local pre_sigmoid_chunk = nn.Narrow(2, 1, 3 * rnn_size)(preactivations)
    local all_gates = nn.Sigmoid()(pre_sigmoid_chunk)

    -- input
    local in_chunk = nn.Narrow(2, 3 * rnn_size + 1, rnn_size)(preactivations)
    local in_transform = nn.Tanh()(in_chunk)

    local in_gate = nn.Narrow(2, 1, rnn_size)(all_gates)
    local forget_gate = nn.Narrow(2, rnn_size + 1, rnn_size)(all_gates)
    local out_gate = nn.Narrow(2, 2 * rnn_size + 1, rnn_size)(all_gates)

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




-- params
cmd = torch.CmdLine()
-- model params
cmd:option('-rnn_size', 10, 'Size of LSTM internal state')
cmd:option('window_size',15,'window size to look into the series')
-- optimization
cmd:option('-learning_rate', 1e-4, 'Learning rate')
cmd:option('-learning_rate_decay', 0.95, 'Learning rate decay')
cmd:option('-learning_rate_decay_after', 10, 'In number of epochs, when to start decaying the learning rate')
cmd:option('-decay_rate', 0.95, 'Decay rate for rmsprop')
cmd:option('-batch_size', 64, 'Batch size')
cmd:option('-max_epochs', 50, 'Number of full passes through the training data')
cmd:option('-dropout', 0.5, 'Dropout')
cmd:option('-init_from', '', 'Initialize network parameters from checkpoint at this path')
-- bookkeeping
cmd:option('-seed', 981723, 'Torch manual random number generator seed')
cmd:option('-save_every', 1000, 'No. of iterations after which to checkpoint')
cmd:option('-train_fc7_file', 'data/train_fc7.t7', 'Path to fc7 features of training set')
cmd:option('-train_fc7_image_id_file', 'data/train_fc7_image_id.t7', 'Path to fc7 image ids of training set')
cmd:option('-val_fc7_file', 'data/val_fc7.t7', 'Path to fc7 features of validation set')
cmd:option('-val_fc7_image_id_file', 'data/val_fc7_image_id.t7', 'Path to fc7 image ids of validation set')
cmd:option('-data_dir', 'data', 'Data directory')
cmd:option('-checkpoint_dir', 'checkpoints', 'Checkpoint directory')
cmd:option('-savefile', 'vqa', 'Filename to save checkpoint to')
-- gpu/cpu
cmd:option('-gpuid', -1, '0-indexed id of GPU to use. -1 = CPU')

-- argument parsing
opt = cmd:parse(arg or {})


-- create the model. Feed one input at a time (value on the time series)
protos = {}
protos.lstm = createLSTM(1,opt.rnn_size)
protos.top = nn.Sequential()
protos.top:add(nn.Linear(opt.rnn_size, 1))
protos.criterion = nn.MSECriterion()

-- create the loader
loader = Loader.new(opt.window_size)

-- compbine all the params and do random inicialization
params, grad_params = utils.combine_all_parameters(protos.lstm, protos.top)
print('Parameters: ' .. params:size(1))
params:uniform(-0.08, 0.08)

-- init the state and clone through time
init_state = {}

if not protos.clones then
    protos.clones = {}
    protos.clones['lstm'] = utils.clone_many_times(protos.lstm, opt.window_size)
end

-- init the hidden state of the network
local h_init = torch.zeros(opt.batch_size, opt.rnn_size)
table.insert(init_state, h_init:clone())
table.insert(init_state, h_init:clone())

local init_state_global = utils.clone_list(init_state)


function feval_val()

    count = 0

    -- get time series data
    x,y = loader:nextTrain()

    rnn_state = {[0] = init_state_global}

    -- go through all the time series
    for t = 1, opt.window_size do
        lst = protos.clones.lstm[t]:forward{x:select(2,t), unpack(rnn_state[t-1])}
        rnn_state[t] = {}
        for i = 1, #init_state do table.insert(rnn_state[t], lst[i]) end
    end

    prediction = protos.top:forward(lst[#lst])

    return prediction
end



function feval(x)

    if x ~= params then
        params:copy(x)
    end
    grad_params:zero()

    -- get training time series
    input,y = loader:nextValidation()

    ------------ forward pass ------------

    loss = 0
    rnn_state = {[0] = init_state_global}

    for t = 1, opt.window_size do
        lst = protos.clones.lstm[t]:forward{input:select(2,t), unpack(rnn_state[t-1])}
        rnn_state[t] = {}
        for i = 1, #init_state do table.insert(rnn_state[t], lst[i]) end
    end

    -- forward through the last time step
    prediction = protos.top:forward(lst[#lst])

    -- forward through the criterion
    loss = protos.criterion:forward(prediction, y)


    ------------ backward pass ------------

    -- loss and soft-max layer backward pass
    dloss = protos.criterion:backward(prediction, y)
    doutput_t = protos.top:backward(lst[#lst], dloss)

    -- init the state of the lstm backward pass through time
    drnn_state[opt.window_size] = {}

    -- backward pass through time
    for t = opt.window_size, 1, -1 do
        dlst = protos.clones.lstm[t]:backward({x:select(2, t), unpack(rnn_state[t-1])}, drnn_state[t])
        drnn_state[t-1] = {}
        table.insert(drnn_state[t-1], dlst[2])
        table.insert(drnn_state[t-1], dlst[3])
    end

    _, lstm_dparams = protos.lstm:getParameters()
    lstm_dparams:clamp(-5, 5)

    return loss, grad_params

end




-- optimization state & params
local optim_state = {learningRate = opt.learning_rate, alpha = opt.decay_rate}

losses = {}
lloss = 0
for i = 1, opt.max_epochs do
    _,local_loss = optim.rmsprop(feval, params, optim_state)
    losses[#losses + 1] = local_loss[1]

    lloss = lloss + local_loss[1]

    if i%10 == 0 then
        print('epoch ' .. i .. ' loss ' .. lloss / 10)
        lloss = 0
    end

    if i%10 == 0 then
        collectgarbage()
    end
end







