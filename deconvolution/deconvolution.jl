using Unfold
using UnfoldSim
using UnfoldMakie,CairoMakie
using CSV
using DataFrames

# data_input = "C:/Users/mvmigem/Documents/data/project_1/preprocessed/mastoid_ref_csv/raw_mastoidref_01.csv"
# events_input = "C:/Users/mvmigem/Documents/data/project_1/preprocessed/mastoid_ref_csv/events_01.csv"
# data = DataFrame(CSV.File(data_input))
# evts = DataFrame(CSV.File(events_input))

data_1, evts_1 = UnfoldSim.predef_eeg()

# select!(data,(:POz))

# data_r = vec(Matrix(data))

basisfunction = firbasis(τ=(-0.1,.5),sfreq=100,name="myFIRbasis")

# f = @formula 0 ~ 1 + position + sequence
f = @formula 0 ~ 1 + condition + continuous # note the formulas left side is `0 ~ ` for technical reasons`

bf_vec = bfDict = ["stimulus"=>(f1,basisfunction1), "response"=>(f2,basisfunction2)]

m = fit(UnfoldModel,bf_vec,evts_1,data_1);
