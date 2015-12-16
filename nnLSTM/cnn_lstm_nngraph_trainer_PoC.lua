Trainer = require '../utils/Trainer'
LSTM = require 'nnLSTM'

require 'nn'
require 'nngraph'



-- params
cmd = torch.CmdLine()

-- model params
cmd:option('-rnn_size', 81, 'Size of LSTM internal state')
cmd:option('-num_layers', 1, 'Depth of the LSTM network')
cmd:option('-time_steps',81,'window size to look into the series')
cmd:option('-input_size',1,'features of the time-series')
-- optimization
cmd:option('-opt_algorithm', 'rmsprop','Optimization algorithm for the training pahse. {sgd, rmsprop}')
cmd:option('-learning_rate', 1e-4, 'Learning rate')
cmd:option('-learning_rate_decay', 0.95, 'Learning rate decay')
cmd:option('-learning_rate_decay_after', 3, 'In number of epochs, when to start decaying the learning rate')
cmd:option('-decay_rate', 0.95, 'Decay rate for rmsprop')
cmd:option('-batch_size', 10, 'Batch size')
cmd:option('-max_epochs', 5, 'Number of full passes through the training data')
cmd:option('-dropout', 0.5, 'Dropout')
cmd:option('-load_from', '', 'Initialize network parameters from checkpoint at this path')
-- bookkeeping
cmd:option('-seed', 981723, 'Torch manual random number generator seed')
cmd:option('-save_every', 15000, 'No. of iterations after which to checkpoint')
cmd:option('-train_file', '../../BTC_DATA/2015-11-20-BTC_USD_TRAIN.txt', 'Path to features of training set')
cmd:option('-val_file', 'data/val_data.t7', 'Path to features of validation set')
cmd:option('-data_dir', '../../BTC_DATA', 'Data directory')
cmd:option('-checkpoint_dir', 'checkpoints', 'Checkpoint directory')
cmd:option('-savefile', 'experimental', 'Filename to save checkpoint to')
-- gpu/cpu
cmd:option('-gpuid', -1, '0-indexed id of GPU to use. -1 = CPU')



-- argument parsing
opt = cmd:parse(arg or {})


---------------------------------------------------------------------------
--                              AUX FUNCTION                             --
---------------------------------------------------------------------------

function create_cnn_model()
    -- building the network:
    local model = nn.Sequential()
    model:add(nn.Reshape(1,9,9))
    -- layer 1:
    model:add(nn.SpatialConvolution(1,16,2,2))          -- 16 x 8 x 8
    model:add(nn.Tanh())
    model:add(nn.SpatialAveragePooling(2,2,2,2))        -- 16 x 4 x 4
    model:add(nn.Reshape(16*4*4))
    model:add(nn.Linear(16*4*4,opt.rnn_size))
    model:add(nn.Tanh())
    
    return model
end


---------------------------------------------------------------------------
--                                MAIN BODY                              --
---------------------------------------------------------------------------

function main()

    local output_size = 1
    local input_size = opt.input_size

    -- load the model or create it
    if opt.load_from ~= '' then
        print('Loading model from file: ' .. opt.load_from)
        local checkpoint = torch.load(opt.load_from)
        model = checkpoint.model

        -- little hack to change the training...
        checkpoint.opt.max_epochs = opt.max_epochs 
        checkpoint.opt.train_file = opt.train_file
        opt = checkpoint.opt
    
    else
        -- create the LSTM core & the CNN model
        print('Creating the model from scratch....')
        local inputs = nn.Identity()()
        local lstm = LSTM.new(opt)(inputs)
        local cnn_net = create_cnn_model()(lstm)
        local net = nn.Linear(opt.rnn_size, 1)(cnn_net)
        local criterion = nn.MSECriterion()

        -- create the decoder, a top layer on top of the LSTM
        model = {}
        model.nnet = nn.gModule({inputs}, {net})
        model.criterion = criterion
    end

    -- create a data loader
    Loader = require '../utils/LoaderSeries'
    loader = Loader.new(opt.batch_size, opt.time_steps)


    -- print('Creating LSTM model:')
    print('--------------------------------------------------------------')
    print('      > Input size: '..input_size)
    print('      > Output size: '..output_size)
    print('      > Number of layers: '..opt.num_layers)
    print('      > Number of units per layer: '..opt.rnn_size)
    print('      > Criterion: '..tostring(model.criterion))
    print('--------------------------------------------------------------')


    -- train the lstm
    trainer = Trainer.new(model, loader, opt)
    trainer:train()

    -- evaluate the lstm
    trainer:validate(true)

    io.read()
end






main()