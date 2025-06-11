using Unfold
using UnfoldSim
using UnfoldMakie,CairoMakie
using CSV
using DataFrames
using Effects
using StatsModels

# el_list = ["Pz","POz","PO4","PO3"] #for P3 analysis of seq 1
el_list = ["Fp1", "AF7", "AF3", "F1", "F3", "F5", "F7", 
           "FT7", "FC5", "FC3", "FC1", "C1", "C3", "C5", 
           "T7", "TP7", "CP5", "CP3", "CP1", "P1", "P3", 
           "P5", "P7", "P9", "PO7", "PO3","O1", "Iz", "Oz",
           "POz", "Pz", "CPz", "Fpz", "Fp2", "AF8", "AF4",
           "AFz", "Fz", "F2", "F4", "F6", "F8", "FT8", "FC6",
           "FC4", "FC2","FCz", "Cz", "C2", "C4", "C6", "T8", 
           "TP8", "CP6", "CP4", "CP2","P2", "P4", "P6", "P8", 
           "P10", "PO8", "PO4", "O2"
           ]
for i in el_list
    data_path = "C:/Users/mvmigem/Documents/data/project_1/preprocessed/mastoid_ref_csv/"
    raw_path = data_path * "raw-$i/"
    destination_path = "C:/Users/mvmigem/Documents/data/project_1/overlap_corrected/$i/"
    # destination_path = "C:/Users/mvmigem/Documents/data/project_1/overlap_corrected/seq-1-p3/$i/" # for seq 1 p3 analysis
    if !isdir(destination_path)
        mkdir(destination_path)
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
            select!(data,i)
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

            setup = string(evts[1,:setup])
            setdown = string(evts[1,:setdown])
            
            design = Dict(:sequence =>["2"],:position => [setup,setdown] ,:attention => ["attended","unattended"],:expectation => ["regular","odd"])
            # design = Dict(:sequence =>["1"],:position => [setup,setdown] ,:attention => ["attended","unattended"]) # for seq 1 anlysis
            eff = effects(design,m)
            eff.subject .= string(evts[1,:subject])
            eff.selected_electrode .= i
            # Save evoked
            subject_num = lpad(evts[1,:subject],2,"0")
            filename = "corrected_$(i)_evoked_$subject_num.csv"
            CSV.write(destination_path*filename, eff)

        end
    end
end