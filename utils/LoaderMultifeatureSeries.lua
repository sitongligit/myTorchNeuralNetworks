torch.setdefaulttensortype('torch.FloatTensor')




LoaderMultifeatureSeries = torch.class('LoaderMultifeatureSeries')


function sumOfSines(x)
    x = x/180*math.pi
    return torch.sin(x) - torch.sin(x*math.pi/2) + torch.sin(x*3) - torch.sin(x*torch.sin(x))
end


function LoaderMultifeatureSeries:__init(batch_size, window_size, num_features)

    -- data generation
    self.data = {}
    self.batch_size = batch_size
    self.train_size = 50000
    self.validation_size = 1000
    self.data.train = sumOfSines(torch.range(1,self.train_size))
    self.data.validation = sumOfSines(torch.range(self.train_size+1, self.train_size + self.validation_size))

    -- batch counters
    self.num_features = num_features
    self.window_size = window_size
    self.train_batch_counter = 0
    self.validation_batch_counter = 0

    return self
end

function LoaderMultifeatureSeries:nextTrain()
    local x
    if self.num_features > 1 then x = torch.zeros(self.batch_size, self.window_size, self.num_features)
    else x = torch.zeros(self.batch_size, self.window_size) end
    local y = torch.zeros(self.batch_size,1)

    for i = 1, self.batch_size do
        self.train_batch_counter = (self.train_batch_counter + 1) % (self.train_size - self.window_size) + 1

        -- multi-dimensional features
        if self.num_features > 1 then
            local temp = self.data.train[{{self.train_batch_counter, self.train_batch_counter+self.window_size-1}}]
            for f = 1, self.num_features do
                x[{i,{},f}] = temp:clone()
            end

        -- uni-dimensional features
        else x[i] = temp:clone() end 
        
        y[i] = self.data.train[self.train_batch_counter+self.window_size]
    end

    y = y:reshape(y:size(1),1)
    return x,y
end

function LoaderMultifeatureSeries:nextValidation()
    local x
    if self.num_features > 1 then x = torch.zeros(self.batch_size, self.window_size, self.num_features)
    else x = torch.zeros(self.batch_size, self.window_size) end
    local y = torch.zeros(self.batch_size, 1)

    for i = 1, self.batch_size do
        self.validation_batch_counter = (self.validation_batch_counter + 1) % (self.validation_size - self.window_size) + 1

        -- multi-dimensional features
        if self.num_features > 1 then
            local temp = self.data.validation[{{self.validation_batch_counter, self.validation_batch_counter+self.window_size-1}}]
            for f = 1, self.num_features do
                x[{i,{},f}] = temp:clone()
            end

        -- uni-dimensional features
        else x[i] = temp:clone() end 
        
        y[i] = self.data.validation[self.validation_batch_counter+self.window_size]
    end

    y = y:reshape(y:size(1),1)
    return x,y
end


return LoaderMultifeatureSeries




