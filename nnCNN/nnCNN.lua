
--- CNN nn module. Implements a CNN network with variable number of layers
-- and number of filters per layer.
-- 
-- For more info check: <br>
-- https://github.com/torch/nn <br>nnCNN



torch.setdefaulttensortype('torch.FloatTensor')


require 'nn'
require 'cunn'
require 'cutorch'
require 'optim'
require 'xlua'


local CNN = torch.class('nn.CNN', 'nn.Module')


--- Initialization function (called by the nn parent Module *new()* function)
-- @param opt is a table with the following fields: <br>
-- <lu>
-- <li> conv_type: type of convolutional neurons {temporal, spatial, volumetric} </li> 
-- </lu>
function CNN:__init(topology_conf)
    if not topology_conf then
        topology_conf = getDefaultConf()
    end
    self.model = createCNNModel(topology_conf)
end

---
function CNN:updateOutput(input)
    self.output = self.model:updateOutput(input)

    -- step by step fporward pass
    -- local output = input
    -- print(self.model)
    -- for l = 1, self.model:size() do
    --      output = self.model:get(l):forward(output)
    --      print('Output size for layer '..l..':\n')
    --      print(output:size())
    --      io.read()
    -- end
    -- self.output = output

    return self.output
end


---
function CNN:updateGradInput(input, gradOutput)
    self.gradInput = self.model:updateGradInput(input, gradOutput)
    return self.gradInput 
end

function CNN:accGradParameters(input,gradOutput,scale)
    self.model:accGradParameters(input, gradOutput, scale)
end

--- Method to reset (zeroes) the parameters of the network.
function CNN:reset()
    self.model:reset()
end

--- Interface to the nn module parameters function. 
-- @return two tensors, one for the flattened learnable parameters and
-- another for the gradients of the energy w.r.t to the learnable parameters.
function CNN:parameters()
    return self.model:parameters()
end


--------------------------------------------------------------------------
--                           FACTORY METHODS                            --
--------------------------------------------------------------------------


--- Creates a CNN model of one of the three possible following types: 
-- {temporal, spatial, volumetric}
-- @return nn Module wrapping a Convolutional Neural Network
function createCNNModel(topology_conf)

    local conv_type = topology_conf.conv_type

    if conv_type == 'temporal' then
        print('Not implemented yet...')
        return

    elseif conv_type == 'spatial' then
        -- 2D convolution
        input_conf, hidden_conf, output_size = topology_conf.input, topology_conf.hidden, topology_conf.output
        return createSpatialCNN(input_conf, hidden_conf, output_size)

    elseif conv_type == 'volumetric' then
        print('Not implemented yet...')
        return

    else print(conv_type .. ' not implemented yet...')
    end
end


--- Creates a SpatialConvolutional Network with as many hidden layers as wanted.
-- @param input Table specifying the input number of planes (num_planes), height and width
-- @param hidden Table specifying the number of hidden layers (num_layers) and for each layer 
-- should contain the number of input planes or filters (in_planes), the number
-- of output planes (out_planes), kernel size (kW,kH), step of the convolution (dW,dH) and
-- the amount of zeros to pad with the input planes (padW,padH) where the default is zero.
-- The same applies to the pooling kernel: size (pkW,pkH), step of the convolution (pdW,pdH) and
-- the amount of zeros to pad with the input planes (ppadW,ppadH) where the default is zero.
-- @return a nn CNN module
function createSpatialCNN(input, hidden, output_size)
    -- as the loaders provide the input flatten, we reshize as an image
    local model = nn.Sequential()
    model:add(nn.Reshape(input.num_planes,input.height, input.width))

    local output_width = input.width
    local output_height = input.height

    for l = 1, hidden.num_layers do
        -- hidden layer l configuration:

        -- features configuration
        local input_planes = hidden.layers[l].in_planes
        local output_planes = hidden.layers[l].out_planes

        -- convolution configuration
        local kW = hidden.layers[l].kW
        local kH = hidden.layers[l].kH
        local dW = hidden.layers[l].dW or 1   -- stride of the convolution
        local dH = hidden.layers[l].dH or 1
        local padH = hidden.layers[l].padH or 0
        local padW = hidden.layers[l].padW or 0

        -- pooling configuration
        local pkW = hidden.layers[l].pkW
        local pkH = hidden.layers[l].pkH
        local pdW = hidden.layers[l].pdW or 1   -- stride of the pooling
        local pdH = hidden.layers[l].pdH or 1
        local ppadH = hidden.layers[l].ppadH or 0
        local ppadW = hidden.layers[l].ppadW or 0


        -- create the set of layers
        model:add(nn.SpatialConvolution(input_planes,output_planes,kW,kH, dW, dH, padW, padH))
        model:add(nn.Tanh())

        -- update the size of the output after the conv. operation
        output_width  = torch.floor((output_width  + 2*padW - kW) / dW + 1)
        output_height = torch.floor((output_height + 2*padH - kH) / dH + 1)

        if hidden.layers[l].pooling_type == 'average' then
            model:add(nn.SpatialAveragePooling(pkW,pkH, pdW, pdH, ppadW, ppadH))
        elseif hidden.layers[l].pooling_type == 'max' then 
            model:add(nn.SpatialMaxPooling(pkW,pkH, pdW, pdH, ppadW, ppadH))
        end

        -- update the size of the output after the conv. operation
        output_width  = torch.floor((output_width  + 2*ppadW - pkW) / pdW + 1)
        output_height = torch.floor((output_height + 2*ppadH - pkH) / pdH + 1)

    end

    -- linearize the output and connect to a fully connected MLP
    local conv_output_size = hidden.layers[hidden.num_layers].out_planes * output_height * output_width
    model:add(nn.Reshape(conv_output_size))
    model:add(nn.Linear(conv_output_size, output_size))
    model:add(nn.Tanh())

    
    return model

end



function getDefaultConf()
    -- input configuration table
    local input_conf = {}
    input_conf.num_planes = 1
    input_conf.height = 9
    input_conf.width = 9

    -- hidden configuration table
    local hidden_conf = {}
    hidden_conf.num_layers = 1
    for l = 1, hidden_conf.num_layers do
        layer_conf = {}
        layer_conf.in_planes = 1
        layer_conf.out_planes = 16
        layer_conf.kW = 2
        layer_conf.kH = 2
        layer_conf.dW = 2
        layer_conf.dH = 2
        layer_conf.pooling_type = 'max'
        hidden_conf.layers = {[l] = layer_conf}
    end
    
    -- top layer configuration
    local output_size = 1

    topo_conf = {}
    topo_conf.conv_type = 'spatial'
    topo_conf.input = input_conf
    topo_conf.hidden = hidden_conf
    topo_conf.output = output_size

    return topo_conf
end


-- function create_convolution_layer(conv_type, opt)
--     local is_supported_layer = {['spatial']=true}
--     assert(is_supported_layer[conv_type], 'Layer type currently not supported....')

--     if conv_type == 'spatial' then
--         return nn.SpatialConvolution(opt.n_input_planes, opt.n_output_planes, opt.kw, opt.kh)
--     end

-- end


-- function create_transformation_layer(transf_layer_type)
--     -- track admited layer
--     local is_supported_layer = {['tahn']=true, ['sigmoid']=true, ['linear']=true}
--     assert(is_supported_layer[transf_layer_type], 'Layer type currently not supported....')

--     -- create and return the requested layer...
--     if transf_layer_type == 'tahn' then
--         return nn.Tanh()
--     elseif transf_layer_type == 'sigmoid' then
--         return nn.Sigmoid()
--     elseif transf_layer_type == 'linear' then
--         return nn.Linear()
-- end


-- function create_pooling_layer(pooling_type, opt)
--     local is_supported_layer = {['spatial max']=true, ['spatial average']=true}
--     assert(is_supported_layer[pooling_type], 'Layer type currently not supported....')

--     -- create the pooling layer
--     if pooling_type == 'spatial max' then
--         return nn.SpatialMaxPooling(opt.kW, opt.kH, opt.dW, opt.dH)
--     elseif pooling_type == 'spatial average' then
--         return nn.SpatialAveragePooling(opt.kW, opt.kH, opt.dW, opt.dH)
--     end
-- end




return CNN