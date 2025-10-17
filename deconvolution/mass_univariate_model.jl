using CSV
using DataFrames
using Effects
using Unfold
using PyMNE
using StatsModels

# Out of loop stuff
# Save effects and model
destination_path = "C:/Users/mvmigem/Documents/data/project_1/uncorrected_rERPs/"
data_path = "C:/Users/mvmigem/Documents/data/project_1/preprocessed/"
event_path = data_path * "mastoid_ref_csv/events/"
event_dir_list = readdir(event_path)

for evp in event_dir_list
    # Load events and file paths
    evts = DataFrame(CSV.File(event_path*evp))
    sub = evts[1,:subject]
    subject_num = lpad(evts[1,:subject],2,"0")
    # Define filename and check if the data was already made
    filename = "uc-rerp-sub-$subject_num.csv"
    m_filename = "uc-rERP-model-sub-$subject_num.jld2"
        if isfile(destination_path*filename)
        continue
    end
    # Load MNE fif files with PyMNE 
    raw_path = data_path*"mastoid_raw/main_clean_mastoidref_$subject_num-raw.fif"
    py_data = PyMNE.io.read_raw_fif(raw_path)
    data = py_data.get_data()
    data = Array(PyArray(data))

    rename!(evts,:sample => :latency)
    filter!(row -> !(row.event_codes == 99),evts)
    evts.position = string.(evts.position)
    evts.sequence = string.(evts.sequence)
    evts.attention = string.(evts.attention)
    evts.expectation = string.(evts.expectation)

    # cut the data into epochs
    data_epochs, times = Unfold.epoch(data = data, tbl = evts, τ = (-0.1, 0.45), sfreq = 512); # channel x timesteps x trials
    size(data_epochs)

    f = @formula 0 ~ 1 + sequence + position + expectation*attention 
    m = fit(UnfoldModel, [Any=>(f, times)], evts, data_epochs);

    setup = string(evts[1,:setup])
    setdown = string(evts[1,:setdown])

    design = Dict(:sequence =>["2"],:position => [setup,setdown] ,:attention => ["attended","unattended"],:expectation => ["regular","odd"])
    # design = Dict(:sequence =>["1"],:position => [setup,setdown] ,:attention => ["attended","unattended"]) # for seq 1 anlysis
    eff = effects(design,m)
    eff.subject .= string(evts[1,:subject])

    # Give back the channel names
    ch_names = pyconvert(Vector{String}, py_data.ch_names)
    eff[!,:channel] = [ch_names[num] for num in eff[!, :channel]]

    # Save effects and model
    CSV.write(destination_path*filename, eff)
    save(joinpath(destination_path, m_filename), m; compress = true);
end