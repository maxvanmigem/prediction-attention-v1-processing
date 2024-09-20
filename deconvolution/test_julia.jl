using DataFrames
using Unfold
using UnfoldMakie, CairoMakie # for plotting
using UnfoldSim

data, evts = UnfoldSim.predef_eeg()

times_cont = range(0,length=200,step=1/100) # we simulated with 100hz for 0.5 seconds

f,ax,h = plot(times_cont,data[1:200])
vlines!(evts[evts.latency .<= 200, :latency] ./ 100;color=:black) # show events, latency in samples!
ax.xlabel = "time [s]"
ax.ylabel = "voltage [µV]"
f

show(first(evts, 6), allcols = true)

# Unfold supports multi-channel, so we could provide matrix ch x time, which we can create like this from a vector:
data_r = reshape(data, (1,:))
# cut the data into epochs
data_epochs, times = Unfold.epoch(data = data, tbl = evts, τ = (-0.4, 0.8), sfreq = 100); # channel x timesteps x trials
size(data_epochs)

typeof(data_epochs)

f = @formula 0 ~ 1 + condition + continuous # note the formulas left side is `0 ~ ` for technical reasons`

m = fit(UnfoldModel, f, evts, data_epochs, times);

m

first(coeftable(m), 6)

