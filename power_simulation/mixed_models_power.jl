using DataFrames
using MixedModels, MixedModelsSim
using Random

n_subj = 10
n_item = 100

items_between = Dict(:attention => ["attended", "unattended"])

rng = MersenneTwister(48)  # specify our random number generator for reproducibility
design = simdat_crossed(rng, n_subj, n_item;
                        item_btwn = items_between)

pretty_table(first(design, 5))  

df = pooled!(DataFrame(design))
first(df, 5)

contrasts = Dict(:attention => EffectsCoding(base="attended"))

form = @formula(dv ~ 1 + attention +
                    (1 + attention | subj) +
                    (1 | item))

m0 = fit(MixedModel, form, design; contrasts=contrasts)

vc = VarCorr(m0)

re_subj = create_re(0.5,0.9)
re_item = create_re(0.3)

# update!(m0; subj=re_subj, item=re_item)
VarCorr(m0)

show(m0.θ)

θ = createθ(m0; subj=re_subj, item= re_item)
show(θ)

σ = 1.05;
β = [4.02, 0.278];

coefnames(m0)


# typically we would use a lot more simulations

# Lines to rerun and adjust
form = @formula(dv ~ 1 + attention +
                    (1 + attention | subj) +
                    (1 + attention | item))

re_subj = create_re(1,0.5)
re_item = create_re(1,0.5)

n_subj = 35
n_item = 800

design = simdat_crossed(rng, n_subj, n_item;
                        item_btwn = items_between)

m0 = fit(MixedModel, form, design; contrasts=contrasts)


# but we want to be quick in this example
sim = parametricbootstrap(MersenneTwister(5658), 1000, m0;
                          β=β, σ=σ, θ = createθ(m0; subj=re_subj, item = re_item))

ptbl = power_table(sim)

pretty_table(ptbl)