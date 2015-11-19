
require 'simple_LSTM'


-- params
cmd = torch.CmdLine()
-- model params
cmd:option('-rnn_size', 5, 'Size of LSTM internal state')
cmd:option('-window_size',15,'window size to look into the series')
cmd:option('-feature_dims',1,'features of the time-series')
-- optimization
cmd:option('-learning_rate', 1e-4, 'Learning rate')
cmd:option('-learning_rate_decay', 0.95, 'Learning rate decay')
cmd:option('-learning_rate_decay_after', 10, 'In number of epochs, when to start decaying the learning rate')
cmd:option('-decay_rate', 0.95, 'Decay rate for rmsprop')
cmd:option('-batch_size', 4, 'Batch size')
cmd:option('-max_epochs', 50000, 'Number of full passes through the training data')
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


-- create a data loader
if opt.feature_dims == 1 then
    require 'Loader_series'
    loader = Loader.new(opt.batch_size, opt.window_size)
else
    require 'Loader_multifeature_series'
    loader = Loader.new(opt.feature_dims, opt.batch_size, opt.window_size)
end

-- create the lstm
my_lstm = LSTM.new(loader, opt)

my_lstm:buildLSTM(opt.feature_dims,1)

-- train the lstm
my_lstm:train()

-- evaluate the lstm
my_lstm:validate()
