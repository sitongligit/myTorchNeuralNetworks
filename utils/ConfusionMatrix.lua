
torch.setdefaulttensortype('torch.FloatTensor')

require 'xlua'



local ConfusionMatrix = torch.class('ConfusionMatrix')

 ------------------------------------------------------------------------------
------------------------ SETTER & CREATOR FUNCTIONS ----------------------------
 ------------------------------------------------------------------------------


function ConfusionMatrix:__init(clases)

    -- set all the clases as strings & keep an index to all
    self.classes = {}
    self.class_2_index = {}
    for i,c in ipairs(clases) do
        self.classes[i] = tostring(c)
        self.class_2_index[tostring(c)] = i
    end

    -- create the matrix
    self.num_classes = #self.classes
    self.matrix = torch.zeros(self.num_classes, self.num_classes)

    return self
end


function ConfusionMatrix:emptyMatrix()
    if self.matrix ~= nil then
        self.matrix:zero()
        return true
    end
    return false
end


function ConfusionMatrix:feedMatrix(predictions, gt)
    assert(#predictions == #gt, 'Predictions and gt arrays do not have the same length...')

    -- populate the matrix with the predictions
    for i = 1, #predictions do
        -- retrieve the index for the given classes
        local gt_index = self.class_2_index[tostring(gt[i])]
        local pred_index = self.class_2_index[tostring(predictions[i])]

        -- check correctness of all the classes given to populate
        assert(gt_index ~= nil, 'GT label '..gt[i]..' not found among the given classes')
        assert(pred_index ~= nil, 'pred label '..predictions[i]..' not found among the given classes')

        -- update matrix contents
        self.matrix[pred_index][gt_index] = self.matrix[pred_index][gt_index] + 1
    end
end






 ---------------------------------------------------------------------
------------------------ MEASURE FUNCTIONS ----------------------------
 ---------------------------------------------------------------------


function ConfusionMatrix:computeMetrics()

    assert(self.matrix)

    local function remove_NAN_and_check_dim(t)
        local clean_t = t[t:eq(t)]
        if clean_t:dim() > 0 then
            return clean_t
        end
        return false
    end

    -- accuracy info
    local diag = self.matrix[torch.eye(self.num_classes):byte()]
    local acc = torch.sum(diag)/torch.sum(self.matrix)

    -- multiclass self.precission, self.recall and F-score containers
    self.precission = torch.Tensor(self.num_classes)
    self.recall = torch.Tensor(self.num_classes)
    self.f_score = torch.Tensor(self.num_classes)

    for i =1,self.num_classes do
        self.precission[i] = self.matrix[i][i] / torch.sum(self.matrix[i])       -- sum over rows
        self.recall[i] = self.matrix[i][i] / torch.sum(self.matrix[{{},i}])      -- sum over columns ()
        self.f_score[i] = 2*(self.precission[i] * self.recall[i]) / (self.precission[i] + self.recall[i])
    end

	-- print the mean of all the measures without the NaN values....
    print('Classification accuracy:', acc)

    local temp = remove_NAN_and_check_dim(self.precission)
    if temp then print('Mean precission:', torch.mean(temp)) end

    temp = remove_NAN_and_check_dim(self.recall)
    if temp then print('Mean recall', torch.mean(temp)) end
    
    temp = remove_NAN_and_check_dim(self.f_score)
    if temp then print('Mean F-score', torch.mean(temp)) end

    return acc, self.precission, self.recall, self.f_score
end


function ConfusionMatrix:computeROC(probabilities, gt)
    -- computing ROC curves by binarizing the problem:
    -- we compute 1 vs all confusion matrix for every class and from there
    -- we get True Positive Rate (tpr) and False Positive Rate (fpr)

    local num_examples = probabilities:size(1)

    -- table to hold all the binary confusion matrices
    local tpr = {}
    local fpr = {}
    local confusion_matrices = {}
    for _,c in ipairs(self.classes) do
        c = self.class_2_index[c]
        confusion_matrices[c] = torch.zeros(2,2)
        tpr[c] = torch.zeros(100)
        fpr[c] = torch.zeros(100)
    end

    -- for every threshold...
    for threshold = 1,100 do

        xlua.progress(threshold,100)

        -- go throw al the examples 
        for e = 1,num_examples do
            local gt_label = self.class_2_index[tostring(gt[e])]
            local prediction_prob, prediction_label = torch.max(probabilities[e],1)
            prediction_prob = prediction_prob:squeeze()
            prediction_label = prediction_label:squeeze()

            -- index of the real class 
            local c = self.class_2_index[tostring(gt_label)]

            -- binarize the problem
            local function binarize_label(prediction_label, label)
                if label == prediction_label then return 1 end
                return 2
            end

            local function switch_label(label)
                if label == 1 then return 2
                else return 1
                end
            end

            -- binarize the GT label
            local bin_gt_label = binarize_label(prediction_label, gt_label)
            local bin_pred_label = binarize_label(prediction_label, prediction_label)

            -- relabel the prediction depending on the threshold
            if prediction_prob <= threshold/100 then bin_pred_label = switch_label(bin_pred_label) end

            -- populate the binary confusion matrix for the current GT class
            confusion_matrices[c][bin_pred_label][bin_gt_label] = confusion_matrices[c][bin_pred_label][bin_gt_label] + 1
        end

        -- compute fpr and tpr for every class
        for _,c in ipairs(self.classes) do
            c = self.class_2_index[tostring(c)]
            local tp = confusion_matrices[c][1][1]
            local fn = confusion_matrices[c][1][2]
            local fp = confusion_matrices[c][2][1]
            local tn = confusion_matrices[c][2][2]
            tpr[c][threshold] = tp / (tp + fn)
            fpr[c][threshold] = fp / (fp + tn)
        end
    end
    return fpr,tpr
end






 --------------------------------------------------------------------
------------------------ GETTER FUNCTIONS ----------------------------
 --------------------------------------------------------------------

function ConfusionMatrix:getMeasuresForClass(class_label)
    local cl_index = self.class_2_index[class_label]
    return self.precission[cl_index], self.recall[cl_index], self.f_score[cl_index]
end


function ConfusionMatrix:getMatrix()
    return self.matrix
end



return ConfusionMatrix








