require 'gnuplot'


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


cmd = torch.CmdLine()
cmd:option('-gamma', 0.2, 'Gamma parameter')
cmd:option('-beta', 0.1, 'Beta parameter')
cmd:option('-tau', 17, 'Tau delay parameter')
cmd:option('-n', 9, 'power parameter')
cmd:option('-seq_length', 1000, 'Length of the sequence to generate')
opt = cmd:parse(arg or {})

mg = {}
plots = {}

for i = 1, 2 do

    -- random params
    local gamma = torch.random(15,25)/100
    local beta = torch.random(5,15)/100

    mg[#mg+1] = MackeyGlassEquation(gamma, beta, opt.tau, opt.n, opt.seq_length)
    plots[#plots+1] = {'g='..gamma..' b='..beta..' n='..opt.n..' tau='..opt.tau,mg[i],'lines ls '..i+1}
end

mg[#mg+1] = MackeyGlassEquation(opt.gamma, opt.beta, opt.tau, opt.n, opt.seq_length)
plots[#plots+1] = {'Mackey Glass time series',mg[#mg],'lines ls '..1}

gnuplot.figure()
gnuplot.plot(plots)
io.read()
