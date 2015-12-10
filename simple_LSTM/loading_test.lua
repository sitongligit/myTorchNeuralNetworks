
simpleLSTM = require 'simpleLSTM'


-- params
cmd = torch.CmdLine()
-- model params
cmd:option('-rnn_size', 35, 'Size of LSTM internal state')
cmd:option('-num_layers', 1, 'Depth of the LSTM network')
cmd:option('-window_size',10,'window size to look into the series')
cmd:option('-feature_dims',1,'features of the time-series')
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




-- laoder
require '../utils/LoaderSeries'
loader = LoaderSeries.new(opt.batch_size, opt.window_size)

-- lstm loading
my_lstm = simpleLSTM(loader, opt)
my_lstm = my_lstm:loadModel('checkpoints/multiple_sinus_lstm_epoch=0_acc=0.9978.t7')


my_lstm:validate(true)
io.read()