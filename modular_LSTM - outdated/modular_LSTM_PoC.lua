
modularLSTM = require 'modularLSTM'
utils = require '../utils/misc'

-- params
cmd = torch.CmdLine()
-- model params
cmd:option('-rnn_size', 50, 'Size of LSTM internal state')
cmd:option('-num_layers', 1, 'Depth of the LSTM network')
cmd:option('-window_size',15,'window size to look into the series')
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
cmd:option('-save_every', 15000, 'No. of iterations after which to checkpoint')
cmd:option('-train_file', 'data/train_data.t7', 'Path to features of training set')
cmd:option('-val_file', 'data/val_data.t7', 'Path to features of validation set')
cmd:option('-data_dir', 'data', 'Data directory')
cmd:option('-checkpoint_dir', 'checkpoints', 'Checkpoint directory')
cmd:option('-savefile', 'multiple_sinus_lstm', 'Filename to save checkpoint to')
-- gpu/cpu
cmd:option('-gpuid', -1, '0-indexed id of GPU to use. -1 = CPU')

-- argument parsing
opt = cmd:parse(arg or {})





function validate(RNN, loader, draw)

    ------------------- evaluation function enclosure -------------------
    local function feval_val()
        -- get time series data
        x,y = loader:nextValidation()

        -- forward through the lstm core
        RNN.lstm:forward(x)

        -- propagate through the top layer the output of the last time-step
        predictions = RNN.top:forward(lst[#lst])

        -- forward through the criterion
        loss = RNN.criterion:forward(predictions, y)

        return predictions, y, loss
    end
    ------------------- evaluation function enclosure -------------------

    local iterations = loader.validation_size / opt.batch_size

    if draw then
        prediction = torch.zeros(loader.validation_size)
        gt = torch.zeros(loader.validation_size)
    end

    for i = 1,iterations do
        -- xlua.progress(i,iterations)
        preds, targets, err = feval_val()

        if draw then
            for j = 1,preds:size(1) do
                local index = (i-1) * opt.batch_size + j
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




function train(RNN, loader)

    print('\n\nTraining network:')
    print('--------------------------------------------------------------')
    print('      > Optimization algorithm: '.. opt.opt_algorithm)
    print('      > Total LSTM number of params: '.. RNN.lstm.params:size(1))
    print('      > Learning rate: '.. opt.learning_rate)
    print('      > Batch size: '.. opt.batch_size)
    print('      > Max num. of epochs: '.. opt.max_epochs)
    print('--------------------------------------------------------------')


    ------------------- evalutation function enclosure -------------------
    local function feval(parameters)

        -- get the data
        input, y = loader:nextTrain()

        -- get net params and reset gradients
        if parameters ~= all_params then
               all_params:copy(parameters)
        end
        all_grad_params:zero()

        -- forward pass
        rnn_state, lst = RNN.lstm:forward(input)
        
        -- forward through the last layer
        prediction = RNN.top:forward(lst[#lst])

        -- forward through the criterion
        loss = RNN.criterion:forward(prediction, y)

        -- loss and soft-max layer backward pass
        dloss = RNN.criterion:backward(prediction, y)
        doutput_t = RNN.top:backward(lst[#lst], dloss)

        RNN.lstm:backward(doutput_t)
        

        return loss, all_grad_params
    end
    ------------------- evaluation function enclosure -------------------


    -- optimization state & params
    local optim_state = {learningRate = opt.learning_rate, alpha = opt.decay_rate}

    losses = {}
    lloss = 0

    local num_batches = loader.train_size / opt.batch_size
    local iterations = opt.max_epochs * num_batches

    -- iterate for all the epochs
    for i = 1, iterations do

        local epoch = i / num_batches

        all_params, all_grad_params = utils.combine_all_parameters(RNN.lstm.protos.lstm, RNN.top)

        _,local_loss = optim.rmsprop(feval, all_params, optim_state)
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
            local val_err = validate(RNN, loader, false)
            saveModel(RNN, 1-val_err, epoch)
        end        

    end

    tensored_loss = torch.zeros(#losses)
    for i,l in ipairs(losses) do
        tensored_loss[i] = l
    end

    gnuplot.figure()
    gnuplot.plot({'loss evolution', tensored_loss, 'lines ls 1'})
end



--[[
    -- FORGET BY NOW ABOUT PERSISTENCY...
    -- Pendiente tambien guardar el loader y toda la informacion que pueda hacer falta
]]
function saveModel(RNN, val_acc, epoch)
    print('\nCheckpointing. Calculating validation accuracy...')
    print('Accuracy: '..val_acc)
    local savefile = string.format('%s/%s_epoch=%i_acc=%.4f.t7', opt.checkpoint_dir, opt.savefile, epoch, val_acc)
    print('Saving checkpoint to ' .. savefile .. '\n')
    local checkpoint = {}
    checkpoint.opt = opt
    checkpoint.RNN = RNN
    torch.save(savefile, checkpoint)
end




---------------------------------------------------------------------------
--                                MAIN BODY                              --
---------------------------------------------------------------------------

function main()
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

    -- create the LSTM core
    local my_lstm = modularLSTM.new(opt,input_size)

    -- create the top layer for adding in top of the LSTM network
    local top_layer = nn.Sequential():add(nn.Linear(opt.rnn_size, output_size))
    local criterion = nn.MSECriterion()

    -- create the decoder, a top layer on top of the LSTM
    RNN = {}
    RNN.lstm = my_lstm
    RNN.top = top_layer
    RNN.criterion = criterion

    print(RNN.top.parameters)
    print(RNN.lstm.protos.lstm)
    -- io.read()

    print('Creating LSTM RNN:')
    print('--------------------------------------------------------------')
    print('      > Input size: '..input_size)
    print('      > Output size: '..output_size)
    print('      > Number of layers: '..opt.num_layers)
    print('      > Number of units per layer: '..opt.rnn_size)
    print('      > Top layer: '..tostring(top_layer:get(1)))
    print('      > Criterion: '..tostring(criterion))
    print('--------------------------------------------------------------')



    -- train the lstm
    train(RNN, loader)

    -- evaluate the lstm
    validate(RNN, loader, true)

    io.read()
end


--[[
    -- forget by now about persistency...

function loadTest()
    -- create a data loader
    if opt.feature_dims == 1 then
        require '../utils/LoaderSeries'
        loader = LoaderSeries.new(opt.batch_size, opt.window_size)
    else
        require '../utils/LoaderMultifeatureSeries'
        loader = LoaderMultifeatureSeries.new(opt.feature_dims, opt.batch_size, opt.window_size)
    end

    local checkpoint = torch.load('checkpoints/multiple_sinus_lstm_epoch=3_acc=0.9962.t7') 
    local RNN = checkpoint.RNN

    print(checkpoint)
    io.read()
    print(RNN.lstm)
    io.read()

    -- evaluate the lstm
    validate(RNN, loader, true)

    io.read()
end


loadTest()
]]



main()
