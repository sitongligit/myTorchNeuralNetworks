
local LoaderMNIST = torch.class('LoaderMNIST')


function LoaderMNIST:__init(batch_size, data_path)
    self.data_path = data_path
    self.train_batch_counter = 1
    self.validation_batch_counter = 1
    self.batch_size = batch_size
    self:downloadData()
end


function LoaderMNIST:downloadData()
    tar = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/mnist.t7.tgz'

    if not paths.dirp(paths.concat(self.data_path, 'mnist.t7')) then
        -- check if tar.gz file already present
        if not paths.dirp(paths.concat(self.data_path, 'mnist.t7.tgz')) then 
            os.execute('wget -P' .. self.data_path.. ' ' .. tar)
        end
        -- extract if folder not found
       os.execute('tar xvf ' .. paths.concat(self.data_path,paths.basename(tar)) .. ' -C ' .. self.data_path)
    end

    train_file = paths.concat(self.data_path,'mnist.t7/train_32x32.t7')
    test_file = paths.concat(self.data_path,'mnist.t7/test_32x32.t7')


    print '==> loading dataset'

    -- We load the dataset from disk, it's straightforward
    self.train_data = torch.load(train_file,'ascii')
    self.test_data = torch.load(test_file,'ascii')

    self.train_size = self.train_data.labels:size(1)
    self.validation_size = self.test_data.labels:size(1)

    -- print('Training Data:')
    -- print(self.train_data)
    -- print()

    -- print('Test Data:')
    -- print(self.test_data)
    -- print()
end


function LoaderMNIST:nextTrain()
    -- get a batch slice of input data 
    -- (we flatten the squared images to be compliant with all the data loaders)
    local range = {self.train_batch_counter, self.train_batch_counter+self.batch_size-1}
    local x = self.train_data.data[{range,{},{},{}}]
    local x = x:reshape(self.batch_size, 32*32):float()
    -- labels (return the 1-hot-encoding vector)
    local y = self.train_data.labels[{range}]:float()
    -- y = oneHotEncode(y)

    -- update training batch pointer
    self.train_batch_counter = (self.train_batch_counter + self.batch_size) % (self.train_data.data:size(1) - self.batch_size) + 1

    return x,y
end


function LoaderMNIST:nextValidation()
    -- get a batch slice of input data 
    -- (we flatten the squared images to be compliant with all the data loaders)
    local range = {self.validation_batch_counter, self.validation_batch_counter+self.batch_size-1}
    local x = self.test_data.data[{range,{},{},{}}]
    local x = x:reshape(self.batch_size, 32*32):float()
    -- labels (return the 1-hot-encoding vector)
    local y = self.test_data.labels[{range}]:float()
    -- y = oneHotEncode(y)

    -- update training batch pointer
    self.validation_batch_counter = (self.validation_batch_counter + self.batch_size) % (self.test_data.data:size(1) - self.batch_size) + 1

    return x,y
end



function oneHotEncode(labels)
    local encoded_labels = torch.zeros(labels:size(1), 10)
    for i = 1,labels:size(1) do
        encoded_labels[i][labels[i]] = 1
    end
    return encoded_labels
end





return LoaderMNIST
