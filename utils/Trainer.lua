
--- Trainer is a general class to train nn or nngraph neural network models.
-- Provides real time error plotting if desired and the capability to save and load models
-- at different stages of the training.

require 'gnuplot'

local Trainer = torch.class('Trainer')


--- Initialization function.
-- @param model struct with two fields. nnet: nn o nngraph model, criterion: any torch nn criterion.
-- @param loader module to load each batch. Should provide the functions *nextTrain()* and *nextValidation()*
-- @param opt options table containing at least the next fields:
-- @param type Type of problem {classification, regression}
-- <br>
-- <ul>
-- <li> opt_algorithm: optimization algorithm name (by now only rmsprop)  </li>
-- <li> learning_rate: integer in {0,1} </li>
-- <li> decay_rate: learning rate decay rate. Integer in {0,1} which would multiply the learning rate after some
-- time. </li>
-- <li> learning_rate_decay_after: number of epochs before starting to decay the learning rate </li>
-- <li> batch_size: integer determining the size of the batch. </li>
-- <li> max_epochs: maximum number of epochs to train. Training could finish earlier if other stoppping criterion
-- is reached. </li>
-- <li> save_every: number of iterations after which the model should be saved. </li>
-- <li> checkpoint_dir: path to the folder where to save the checkpoints. </li>
-- <li> save_file: name which the chekpoints will be saved with. </li>
-- </lu>
function Trainer:__init(model, loader, opt, type)
    self.type = type or 'regression'
    self.opt = opt
    self.model = model
    self.loader = loader
end



--- Training function.
-- Trains a neural network model during a maximum number of iterations 
-- (iterations = max_epochs * batch_size)
-- If a learning rate decay value less than 1 is provided, after 
-- 'learning_rate_decay_after' epochs the learning rate would be decayed by 
-- the 'decay_rate'.
-- The function also saves a current state of the model every 'save_every' 
-- number of epochs. Saving the current model and the options provided to the trainer.
-- During the training process this functin also plots the training and validation error
-- evolution allowing to prevent an overfitting by studying the error curves interaction.
-- PENDING: Implement early stopping by 'convergence', ''upper bound accuracy' and 
-- 'overfitting' by error comparison...
-- @see saveModel
function Trainer:train()

    print('\n\nTraining network:')
    print('--------------------------------------------------------------')
    print('      > Optimization algorithm: '.. self.opt.opt_algorithm)
    print('      > Total number of params: '.. self.model.nnet:getParameters():size(1))
    print('      > Learning rate: '.. self.opt.learning_rate)
    print('      > Batch size: '.. self.opt.batch_size)
    print('      > Max num. of epochs: '.. self.opt.max_epochs)
    print('--------------------------------------------------------------')


    -- get params and gradient of the parameters
    local params, grads = self.model.nnet:getParameters()


    ------------------- evalutation function enclosure -------------------
    local function feval(parameters)

        -- get the data
        input, y = self.loader:nextTrain()

        -- get net params and reset gradients
        if parameters ~= params then
               params:copy(parameters)
        end
        grads:zero()

        -- forward pass
        output = self.model.nnet:forward(input)

        -- forward through the criterion
        loss = self.model.criterion:forward(output, y)

        -- loss and soft-max layer backward pass
        dloss = self.model.criterion:backward(output, y)
        self.model.nnet:backward(input, dloss)
        
        return loss, grads
    end
    ------------------- evaluation function enclosure -------------------


    -- optimization state & params
    local optim_state = {learningRate = self.opt.learning_rate, alpha = self.opt.decay_rate}

    train_loss = {}
    validation_loss = {}
    all_train_loss = {}
    lloss = 0

    -- computing the number of iterations
    local num_batches = self.loader.train_size / self.opt.batch_size
    local iterations = self.opt.max_epochs * num_batches

    -- iterate for all the epochs
    for i = 1, iterations do

        local epoch = i / num_batches

        -- One optimization step
        _,local_loss = optim.rmsprop(feval, params, optim_state)
        all_train_loss[#all_train_loss + 1] = local_loss[1]

        xlua.progress(i, iterations)

        -- collect garbage every few iterations...
        if i%10 == 0 then
            collectgarbage()
        end

        -- learning rate decay
        if i % num_batches == 0 and self.opt.learning_rate_decay < 1 then
            if epoch >= self.opt.learning_rate_decay_after then
                local decay_factor = self.opt.learning_rate_decay
                optim_state.learningRate = optim_state.learningRate * decay_factor
                print('decayed learning rate by a factor ' .. decay_factor .. ' to ' .. optim_state.learningRate)
            end
        end

        -- checkpoint: saving model
        if i % self.opt.save_every == 0 or i == iterations then
            local val_err = self:validate(false)
            self:saveModel(1-val_err, epoch)

            train_loss[#train_loss+1] = all_train_loss[#all_train_loss]
            validation_loss[#validation_loss+1] = val_err

            -- print another point of the plot (validation vs training error)
            local t_tensored_loss = torch.Tensor(train_loss)
            local train_plot = {'training loss evolution', t_tensored_loss[{{1,#train_loss}}], 'lines ls 3'}
            local v_tensored_loss = torch.Tensor(validation_loss)
            local validation_plot = {'validation loss evolution', v_tensored_loss[{{1,#validation_loss}}], 'lines ls 2'}

            gnuplot.plot({train_plot, validation_plot})

        end        

    end

    -- plot the loss evolution
    if self.type == 'regression' then
        local tensored_loss = torch.Tensor(all_train_loss)
        gnuplot.figure()
        gnuplot.plot({'loss evolution', tensored_loss, 'lines ls 1'})
    end
end



--- Validation function.
-- Performs a forward pass over all the validation data to compute the loss
-- value of the given model. If the draw value is set to true plots the current fit
-- to the data. (Only possible fot regression problems)
-- @param draw Boolean value to switch on/off the plotting of the prediction and the targets.
-- @return The loss (real value) of the current model on the validation dataset.
function Trainer:validate(draw)

    ------------------- evaluation function enclosure -------------------
    local function feval_val()
        -- get time series data
        x,y = self.loader:nextValidation()

        -- forward through the lstm core
        output = self.model.nnet:forward(x)

        -- forward through the criterion
        loss = self.model.criterion:forward(output, y)

        return output, y, loss
    end
    ------------------- evaluation function enclosure -------------------

    local iterations = self.loader.validation_size / self.opt.batch_size

    if draw then
        prediction = torch.zeros(loader.validation_size)
        gt = torch.zeros(loader.validation_size)
    end

    for i = 1,iterations do
        -- xlua.progress(i,iterations)
        preds, targets, err = feval_val()

        if draw then
            for j = 1,preds:size(1) do
                local index = (i-1) * self.opt.batch_size + j
                prediction[index] = preds[j]
                gt[index] = targets[j]
            end
        end
    end

    -- draw the prediction vs the target
    if draw then
        gnuplot.figure()
        gnuplot.plot({{'targets', gt, 'lines ls 1'},{'predictions', prediction, 'lines ls 2'}})
    end

    return loss
end


--- Saves the model and the options given to the trainer in a t7 Torch format.
-- A checkpoint is basicaly a table with two values: 'opt' and 'model'.
-- The first containing the table with all the options given to the trainer and the former the model,
-- depending on what format was the model provided in, this could be in a 'nngraph' or a 'nn' Torch form.
-- @param acc Accuracy of the the current model. The name of the checkpoint reflect this accuracy.
-- @param epoch Current epoch of the training process.
function Trainer:saveModel(acc,epoch)

    -- some printing to sea evolution
    print('\nCheckpointing...')
    print('Accuracy: '.. acc )

    -- saving model, loader and command options
    local savefile = string.format('%s/%s_epoch=%i_acc=%.4f.t7', self.opt.checkpoint_dir, self.opt.savefile, epoch, acc)
    print('Saving checkpoint to ' .. savefile .. '\n')
    local checkpoint = {}
    checkpoint.opt = self.opt
    checkpoint.model = self.model
    torch.save(savefile, checkpoint)
end


return Trainer

