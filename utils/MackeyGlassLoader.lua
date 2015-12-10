torch.setdefaulttensortype('torch.FloatTensor')


local MackeyGlassLoader = torch.class('MackeyGlassLoader')
-- MackeyGlassLoader = {}
-- MackeyGlassLoader.__index = MackeyGlassLoader


function MackeyGlassLoader:__init(batch_size, steps, gamma, beta, tau, n)
    self.gamma = gamma
    self.beta = beta
    self.tau = tau
    self.n = n

    self.train_size = 50000
    self.validation_size = 10000

    self.batch_size = batch_size
    self.time_steps = steps

    self.train_batch_counter = 0
    self.validation_batch_counter = 0

    self.data = {}
    self.data.train = MackeyGlassEquation(gamma, beta, tau, n, self.train_size)
    self.data.validation = MackeyGlassEquation(gamma, beta, tau, n, self.validation_size)

    return self
end

function MackeyGlassLoader:nextTrain()
    local x = torch.zeros(self.batch_size, self.time_steps)
    local y = torch.zeros(self.batch_size,1)
    for i = 1, self.batch_size do
        self.train_batch_counter = (self.train_batch_counter + 1) % (self.train_size - self.time_steps) + 1
        x[i] = self.data.train[{{self.train_batch_counter, self.train_batch_counter+self.time_steps-1}}]
        y[i] = self.data.train[self.train_batch_counter+self.time_steps]
    end
    y = y:reshape(y:size(1),1)
    return x,y
end


function MackeyGlassLoader:nextValidation()
    -- data structure:
    -- 1D (y): batch_size, 2D (x): time-steps (sentence size in words etc)
    local x = torch.zeros(self.batch_size, self.time_steps)
    local y = torch.zeros(self.batch_size, 1)
    for i = 1, self.batch_size do
        self.validation_batch_counter = (self.validation_batch_counter + 1) % (self.validation_size - self.time_steps) + 1
        x[i] = self.data.train[{{self.validation_batch_counter, self.validation_batch_counter+self.time_steps-1}}]
        y[i] = self.data.train[self.validation_batch_counter+self.time_steps]
    end
    y = y:reshape(y:size(1),1)
    return x,y
end


function MackeyGlassEquation(gamma, beta, tau, n, seq_length)
    -- deafult params for the Mackey-Glass equation
    local gamma = gamma or .2
    local beta = beta or .1
    local tau = tau or 17
    local n = n or 10
    local seq_length = seq_length or 700

    tau = math.min(17, tau)

    -- generate Mackey-Glass time series
    y = {0.9697, 0.9699, 0.9794, 1.0003, 1.0319, 1.0703, 1.1076, 1.1352, 1.1485, 1.1482, 1.1383, 1.1234, 1.1072, 1.0928, 1.0820, 1.0756, 1.0739, 1.0759} 

    for t = #y, seq_length do
        y[t+1] = y[t] - beta*y[t] + gamma*y[t-tau]/(1+math.pow(y[t-tau],n)) 
    end
    return torch.Tensor(y)
end



return MackeyGlassLoader

