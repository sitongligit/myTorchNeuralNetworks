CNN = require 'nnCNN'
Trainer = require '../utils/Trainer'
Loader = require '../utils/LoaderMNIST'


-- params
cmd = torch.CmdLine()

-- model params
cmd:option('-num_layers', 2, 'Depth of the CNN network')
cmd:option('-input_w',32,'Input image width')
cmd:option('-input_h',32,'Input image height')
cmd:option('-input_planes',1,'Input image planes (B&N = 1, RGB = 3)')
cmd:option('-pooling','max','Type of pooling layer')
-- optimization (training params)
cmd:option('-opt_algorithm', 'rmsprop','Optimization algorithm for the training pahse. {sgd, rmsprop}')
cmd:option('-learning_rate', 1e-4, 'Learning rate')
cmd:option('-learning_rate_decay', 0.95, 'Learning rate decay')
cmd:option('-learning_rate_decay_after', 3, 'In number of epochs, when to start decaying the learning rate')
cmd:option('-decay_rate', 0.95, 'Decay rate for rmsprop')
cmd:option('-batch_size', 10, 'Batch size')
cmd:option('-max_epochs', 5, 'Number of full passes through the training data')
cmd:option('-load_from', '', 'Initialize network parameters from checkpoint at this path')
-- bookkeeping
cmd:option('-seed', 981723, 'Torch manual random number generator seed')
cmd:option('-save_every', 5000, 'No. of iterations after which to checkpoint')
cmd:option('-train_file', '', 'Path to features of training set')
cmd:option('-val_file', '', 'Path to features of validation set')
cmd:option('-data_dir', '/Users/josemarcos/Desktop/myTorchLSTM/data', 'Data directory')
cmd:option('-checkpoint_dir', 'checkpoints', 'Checkpoint directory')
cmd:option('-savefile', 'cnn_test', 'Filename to save checkpoint to')
-- gpu/cpu
cmd:option('-gpuid', -1, '0-indexed id of GPU to use. -1 = CPU')



-- argument parsing
opt = cmd:parse(arg or {})



--- Temporal function to test the module:
-- TODO: read from config file to make things easier...
function getConf()
    -- input configuration table
    local input_conf = {}
    input_conf.num_planes = opt.input_planes
    input_conf.height = opt.input_h
    input_conf.width = opt.input_w

    -- hidden configuration table
    local hidden_conf = {}
    hidden_conf.num_layers = opt.num_layers

    -- hack to work with the input as hidden layer number 0
    hidden_conf.layers = {}
    hidden_conf.layers[0] = {}
    hidden_conf.layers[0].out_planes = 1

    for l = 1, opt.num_layers do
        layer_conf = {}
        layer_conf.in_planes = hidden_conf.layers[l-1].out_planes
        layer_conf.out_planes = 32/l
        layer_conf.kW = 2*l
        layer_conf.kH = 2*l
        layer_conf.dW = 2
        layer_conf.dH = 2
        layer_conf.pooling_type = 'max'
        hidden_conf.layers[l] = layer_conf
    end
    
    -- top layer configuration
    local output_size = 10

    topo_conf = {}
    topo_conf.conv_type = 'spatial'
    topo_conf.input = input_conf
    topo_conf.hidden = hidden_conf
    topo_conf.output = output_size

    return topo_conf
end



function main()
    -- create a CNN core
    local conf = getConf()
    cnn = CNN.new(conf)

    -- add on top a softMax layer to classify digits
    local nnet = nn.Sequential()
    nnet:add(cnn)
    nnet:add(nn.SoftMax())

    -- wrap together neural net + criterion
    model = {}
    model.nnet = nnet
    model.criterion = nn.ClassNLLCriterion()

    -- train the system to lear how to identify MNIST digits
    -- get the loader
    local loader = Loader.new(opt.batch_size, opt.data_dir)

    -- get the trainer
    trainer = Trainer.new(model, loader, opt)

    -- train
    trainer:train()

    -- validate
    trainer:validate()

    io.read()
end


main()





