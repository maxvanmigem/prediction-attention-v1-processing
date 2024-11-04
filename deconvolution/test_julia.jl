
data_path = "C:/Users/mvmigem/Documents/data/project_1/preprocessed/mastoid_ref_csv/"
event_path = data_path * "events/"
raw_path = data_path * "raw-POz/"

event_dir_list = readdir(event_path)
raw_dir_list = readdir(raw_path)

for i in raw_dir_list
    println(i)
end
