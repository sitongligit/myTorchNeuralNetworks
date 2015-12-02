torch.setdefaulttensortype('torch.FloatTensor')


local LoaderSeries = torch.class('LoaderSeries')

function sumOfSines(x)
    x = x/180*math.pi
    return torch.sin(x) - torch.sin(x*math.pi/2) + torch.sin(x*3)
end


function LoaderSeries:__init(batch_size, window_size)

    -- data generation
    self.data = {}
    self.train_size = 50000
    self.validation_size = 10000
    self.batch_size = batch_size

    self.data.train = sumOfSines(torch.range(1,self.train_size))
    self.data.validation = sumOfSines(torch.range(self.train_size+1, self.train_size + self.validation_size))

    -- batch counters
    self.window_size = window_size
    self.train_batch_counter = 0
    self.validation_counter = 0

    return self
end

function LoaderSeries:nextTrain()
    -- data structure:
    -- 1D (y): batch_size, 2D (x): time-steps (sentence size in words etc)
    local x = torch.zeros(self.batch_size, self.window_size)
    local y = torch.zeros(self.batch_size,1)
    for i = 1, self.batch_size do
        self.train_batch_counter = (self.train_batch_counter + 1) % (self.train_size - self.window_size) + 1
        x[i] = self.data.train[{{self.train_batch_counter, self.train_batch_counter+self.window_size-1}}]
        y[i] = self.data.train[self.train_batch_counter+self.window_size]
    end
    y = y:reshape(y:size(1),1)
    return x,y
end

function LoaderSeries:nextValidation()
    -- data structure:
    -- 1D (y): batch_size, 2D (x): time-steps (sentence size in words etc)
    local x = torch.zeros(self.batch_size, self.window_size)
    local y = torch.zeros(self.batch_size, 1)
    for i = 1, self.batch_size do
        self.validation_counter = (self.validation_counter + 1) % (self.validation_size - self.window_size) + 1
        x[i] = self.data.validation[{{self.validation_counter, self.validation_counter+self.window_size-1}}]
        y[i] = self.data.validation[self.validation_counter+self.window_size]
    end
    y = y:reshape(y:size(1),1)
    return x,y
end

return LoaderSeries



