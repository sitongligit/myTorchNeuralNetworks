
require 'simple_LSTM'


-- params
cmd = torch.CmdLine()
-- model params
cmd:option('-rnn_size', 7, 'Size of LSTM internal state')
cmd:option('-num_layers', 1, 'Depth of the LSTM network')
cmd:option('-window_size',15,'window size to look into the series')
cmd:option('-feature_dims',6,'features of the time-series')
-- optimization
cmd:option('-opt_algorithm', 'rmsprop','Optimization algorithm for the training pahse. {sgd, rmsprop}')
cmd:option('-learning_rate', 1e-4, 'Learning rate')
cmd:option('-learning_rate_decay', 0.95, 'Learning rate decay')
cmd:option('-learning_rate_decay_after', 10, 'In number of epochs, when to start decaying the learning rate')
cmd:option('-decay_rate', 0.95, 'Decay rate for rmsprop')
cmd:option('-batch_size', 10, 'Batch size')
cmd:option('-max_epochs', 5, 'Number of full passes through the training data')
cmd:option('-dropout', 0.5, 'Dropout')
cmd:option('-init_from', '', 'Initialize network parameters from checkpoint at this path')
-- bookkeeping
cmd:option('-seed', 981723, 'Torch manual random number generator seed')
cmd:option('-save_every', 10000, 'No. of iterations after which to checkpoint')
cmd:option('-train_file', 'data/train_data.t7', 'Path to features of training set')
cmd:option('-val_file', 'data/val_data.t7', 'Path to features of validation set')
cmd:option('-data_dir', 'data', 'Data directory')
cmd:option('-checkpoint_dir', 'checkpoints', 'Checkpoint directory')
cmd:option('-savefile', 'multiple_sinus_lstm', 'Filename to save checkpoint to')
-- gpu/cpu
cmd:option('-gpuid', -1, '0-indexed id of GPU to use. -1 = CPU')

-- argument parsing
opt = cmd:parse(arg or {})


-- create a data loader
if opt.feature_dims == 1 then
    require '../utils/LoaderSeries'
    loader = LoaderSeries.new(opt.batch_size, opt.window_size)
else
    require '../utils/LoaderMultifeatureSeries'
    loader = LoaderMultifeatureSeries.new(opt.feature_dims, opt.batch_size, opt.window_size)
end

-- create the lstm
local output_size = 1
local input_size = opt.feature_dims

-- create the top layer for adding in top of the LSTM
local top_layer = nn.Sequential():add(nn.Linear(opt.rnn_size, output_size))
local criterion = nn.MSECriterion()

-- create the LSTM handler class instance
my_lstm = LSTM.init(loader, opt)

-- create the network
my_lstm:createRNN(input_size, top_layer, criterion)

-- train the lstm
my_lstm:train()

-- evaluate the lstm
my_lstm:validate(true)

io.read()
