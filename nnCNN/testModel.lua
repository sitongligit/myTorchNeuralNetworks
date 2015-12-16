Loader = require '../utils/LoaderMNIST'
CM = require '../utils/ConfusionMatrix'
CNN = require 'nnCNN'
require 'image'
require 'xlua'
require 'nn'


-- laod model
local checkpoint = torch.load('./checkpoints/cnn_test_avg_p2_epoch=4_acc=0.9901.t7')
local opt = checkpoint.opt
local nnet = checkpoint.model.nnet
local criterion = checkpoint.model.criterion

-- create the model
model = {}
model.nnet = nnet
model.criterion = criterion

-- create loader
opt.data_dir = '/Users/josemarcos/Desktop/myTorchNeuralNetworks/data'
local loader = Loader.new(opt.batch_size, opt.data_dir)




-- load confusion matrix class for evaluations
local classes = {'1','2','3','4','5','6','7','8','9','10'}
local cm = CM.new(classes)

all_targets = {}
all_probs = {}

-- test and create confusion matrix
local iterations = 250
for j = 1,iterations do
    xlua.progress(j,iterations)
    inputs, targets = loader:nextValidation()
    predictions_probs = nnet:forward(inputs)
    predictions, pred_classes = torch.max(predictions_probs,2)
    pred_classes = pred_classes:squeeze()

    -- convert predictions and targets to a table of strings
    splitter = nn.SplitTable(1, pred_classes:size(1))
    pred_classes = splitter:forward(pred_classes)
    targets = splitter:forward(targets)

    for i = 1, #pred_classes do
        pred_classes[i] = tostring(pred_classes[i])
        targets[i] = tostring(targets[i])

        -- temporal para debugar
        local saving_path = './predictions/b'..tostring(j)..'_im'..i..'_pred'..pred_classes[i]..'.png'
        image.save(saving_path, inputs[i]:reshape(32, 32))

        -- append to draw ROC curves
        -- all_targets[#all_targets+1] = targets[i]
        -- all_probs[#all_probs+1] = predictions_probs[i] 
    end

    cm:feedMatrix(pred_classes, targets)
end

cm:computeMetrics()
print(cm:getMatrix())