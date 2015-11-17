torch.setdefaulttensortype('torch.FloatTensor')


Loader = {}
Loader.__index = Loader




function MackeyGlassEquation(X,gamma, beta, tau)
    -- deafult params for the Mackey-Glass equation
    local gamma = gamma or 1
    local beta = beta or 2
    local tau = tau or 2

    -- solve the delayed differential equation using 4th order Runge-Kutta method
    print('not implemented yet....')
end


function sumOfSines(x)
    x = x/180*math.pi
    return torch.sin(x) - torch.sin(x*math.pi/2) + torch.sin(x*3)
end


function Loader.new(window_size)
    local self = {}
    setmetatable(self, Loader)

    -- data generation
    self.data = {}
    self.train_size = 50000
    self.validation_size = 10000
    self.data.train = sumOfSines(torch.range(1,self.train_size))
    self.data.validation = sumOfSines(torch.range(self.train_size+1, self.train_size + self.validation_size))

    -- batch counters
    self.window_size = window_size
    self.train_batch_counter = 0
    self.validation_counter = 0

    return self
end

function Loader.nextTrain(self)
    self.train_batch_counter = (self.train_batch_counter + 1) % self.train_size
    local x = self.data.train[{{self.train_batch_counter, self.train_batch_counter+self.window_size-1}}]
    local y = self.data.train[self.train_batch_counter+self.window_size]
    return x,y
end

function Loader.nextValidation(self)
    self.validation_counter = (self.validation_counter + 1) % self.validation_size
    local x = self.data.train[{{self.validation_counter, self.validation_counter+self.window_size-1}}]
    local y = self.data.train[self.validation_counter+self.window_size]
    return x,y
end





