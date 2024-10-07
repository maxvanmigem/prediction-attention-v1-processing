using Unfold
using UnfoldSim
using UnfoldMakie,CairoMakie
using CSV
using DataFrames
using Effects
using StatsModels

data_input = "C:/Users/mvmigem/Documents/data/project_1/preprocessed/mastoid_ref_csv/raw_mastoidref_21.csv"
events_input = "C:/Users/mvmigem/Documents/data/project_1/preprocessed/mastoid_ref_csv/events_21.csv"
data = DataFrame(CSV.File(data_input))
evts = DataFrame(CSV.File(events_input))
rename!(evts,:sample => :latency)
filter!(row -> !(row.event_codes == 99),evts)
evts.position = string.(evts.position)
evts.sequence = string.(evts.sequence)
data_1, evts_1 = UnfoldSim.predef_eeg()

select!(data,"POz")

data_r = vec(Matrix(data))

basisfunction = firbasis(τ=(-0.4,0.8),sfreq=128,name="myFIRbasis")

f = @formula 0 ~ 1 + position + sequence + expectation + attention + expectation&attention
# f = @formula 0 ~ 1 + condition + continuous # note the formulas left side is `0 ~ ` for technical reasons`
bfDict = Dict(Any=>(f,basisfunction))
contrasts = Dict(:position=>EffectsCoding(),
                :sequence=>EffectsCoding(),
                :expectation=>EffectsCoding(),
                :attention=>EffectsCoding())
m = fit(UnfoldModel,bfDict,evts,data_r,contrasts=contrasts)

results = coeftable(m)
plot_erp(results)

design = Dict(:sequence => ["2"],:expectation => ["regular","odd"], :attention => ["attended","unattended"])
eff = effects(design,m)
plot_erp(eff; mapping = (; color = :expectation))



