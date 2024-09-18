using DataFrames
using CSV
using Unfold
using UnfoldMakie, CairoMakie
using UnfoldSim

data_input = "C:/Users/mvmigem/Documents/data/project_1/preprocessed/mastoid_ref_csv/raw_mastoidref_01.csv"
events_input = "C:/Users/mvmigem/Documents/data/project_1/preprocessed/mastoid_ref_csv/events_01.csv"
data = DataFrame(CSV.File(data_input))
evts = DataFrame(CSV.File(events_input))

times_cont = range(0,length=200,step=1/100) # we simulated with 100hz for 0.5 seconds

f,ax,h = plot(times_cont,data[1:200])
vlines!(evts[evts.latency .<= 200, :latency] ./ 100;color=:black) # show events, latency in samples!
ax.xlabel = "time [s]"
ax.ylabel = "voltage [µV]"
f

# Unfold supports multi-channel, so we could provide matrix ch x time, which we can create like this from a vector:
data_r = Matrix(data)
data = Nothing
# cut the data into epochs
data_epochs, times = Unfold.epoch(data = data_r, tbl = evts, τ = (-0.1, 0.5), sfreq = 516); # channel x timesteps x trials
size(data_epochs)

typeof(data_epochs)

f = @formula 0 ~ 1 + condition + continuous # note the formulas left side is `0 ~ ` for technical reasons`

m = fit(UnfoldModel, f, evts, data_epochs, times);

m

first(coeftable(m), 6)

results = coeftable(m)
plot_erp(results)

