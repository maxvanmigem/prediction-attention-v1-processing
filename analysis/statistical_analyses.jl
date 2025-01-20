using DataFrames
using MixedModels 
using StatsModels
using CSV
using CategoricalArrays

df_path = "C:/Users/mvmigem/Documents/data/project_1/compiled_dataframes/c1_long_20ms_df.csv"

df = DataFrame(CSV.File(df_path))

df.subject = categorical(df.subject)
df.attention = categorical(df.attention)
df.expectation = categorical(df.expectation)
df.visual_field = categorical(df.visual_field)

df_up = filter(row -> row.visual_field == "up", df)
df_down = filter(row -> row.visual_field == "down", df)

contrasts = Dict(:attention=>EffectsCoding(),
                :visual_field=>EffectsCoding(),
                :expectation=>EffectsCoding(),
                :subject=>EffectsCoding())


# fit up and down tailored_amp ultra_amp general_amp
f = @formula(general_amp ~ 1 + attention + expectation + visual_field +
    attention&expectation + attention&visual_field + expectation&visual_field +
    (1 + attention | subject) + (expectation | subject))

fit_all = fit(MixedModel,f,df,contrasts=contrasts)

# fit only up tailored_amp ultra_amp general_amp
f = @formula(tailored_amp ~ 1 + attention + expectation+ attention&expectation +
    (1 + attention | subject) + (expectation | subject))

fit_up = fit(MixedModel,f,df_up,contrasts=contrasts)

# fit only down tailored_amp ultra_amp general_amp
fit_down = fit(MixedModel,f,df_down,contrasts=contrasts)