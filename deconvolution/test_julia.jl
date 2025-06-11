using Unfold
using UnfoldSim
using UnfoldMakie,CairoMakie
using CSV
using DataFrames
using Effects
using StatsModels


data_path = "C:/Users/mvmigem/Documents/data/project_1/preprocessed/mastoid_ref_csv/"
raw_path = data_path * "raw-selected/"
destination_path = "C:/Users/mvmigem/Documents/data/project_2/model_fit_project1/"
# destination_path = "C:/Users/mvmigem/Documents/data/project_1/overlap_corrected/seq-1-p3/$i/" # for seq 1 p3 analysis
model_fits = []
if !isdir(destination_path)
    mkdir(destination_path)
end
# Data paths
# destination_path = "C:/Users/mvmigem/Documents/data/project_1/overlap_corrected/variable-electrode/"
event_path = data_path * "events/"
# raw_path = data_path * "raw-POz/"
# raw_path = data_path * "raw-selected/"


event_dir_list = readdir(event_path)
raw_dir_list = readdir(raw_path)


# data = DataFrame(CSV.File(raw_path*raw_dir_list[1]))
# evts = DataFrame(CSV.File(event_path*event_dir_list[1]))
for (evp,rawp) in zip(event_dir_list,raw_dir_list)

    # Read in the data
    data = DataFrame(CSV.File(raw_path*rawp))
    evts = DataFrame(CSV.File(event_path*evp))
    # Change the column names to fit toolbox
    rename!(evts,:sample => :latency)
    filter!(row -> !(row.event_codes == 99),evts)
    evts.position = string.(evts.position)
    evts.sequence = string.(evts.sequence)
    
    # selected electrode
    # selected_electrode = string(evts[1,:selected_electrode])
    select!(data,Not("time"))
    # Transform to data to Matrix
    data_r = vec(Matrix(data))
    # Define basisfunction
    basisfunction = firbasis(τ=(-0.1,1),sfreq=512,name="myFIRbasis")
    # Define the linear formula 
    f = @formula 0 ~ 1 + sequence + position + expectation*attention 
    # f = @formula 0 ~ 1 + condition + continuous # note the formulas left side is `0 ~ ` for technical reasons`
    bfDict = Dict(Any=>(f,basisfunction))
    # Specify contrasts
    contrasts = Dict(:position=>EffectsCoding(),
                    :sequence=>EffectsCoding(),
                    :expectation=>EffectsCoding(),
                    :attention=>EffectsCoding())
    # Fit
    m = fit(UnfoldModel,bfDict,evts,data_r,contrasts=contrasts)
    push!(model_fits, m)

end

t = model_fits[2]

results = coeftable(t)
plot_erp(results)