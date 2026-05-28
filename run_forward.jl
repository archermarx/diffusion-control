@show Threads.nthreads()

# TODO: either bundle normalize_data in this repo or expose this as an arg
# TODO TODO: split julia code from main hall_diffusion repo and make separate package that we can just add
# Currently, we just find the hall_diffusion repo and reuse its Julia environment.
diffusion_dir = "/home/archermarks/projects/hall_diffusion"
using Pkg
Pkg.activate(diffusion_dir)
include(joinpath(diffusion_dir, "julia", "normalize_data.jl"))

"""
save_sim(sim, params)

Save a simulation to a dictionary after averaging it in time for the specified interval.
In addition to axially-resolved fields, we also write out certain time-dependent global quantities (thrust, current)
as well as the params with which the simulation was run.
"""
function save_sim(sim, params = nothing)
    avg = if length(sim.frames) > 1
        het.time_average(sim, length(sim.frames) ÷ 2)
    else
        sim
    end

    out_dict = Dict(
        :sim => het.serialize(avg),
        :time => Dict(
            :time_s => sim.t .|> Float32,
            :discharge_current_A => het.discharge_current(sim) .|> Float32,
            :thrust_mN => het.thrust(sim) .|> Float32,
        ),
        :params => params,
    )

    return out_dict
end

input_dir = ARGS[1]
output_dir = ARGS[2]
input_files = readdir(input_dir, join=true)

Threads.@threads for i in eachindex(input_files)
    in_file = input_files[i]
    base = splitext(basename(in_file))[1]
    out_file = joinpath(output_dir, "$(base).npz")

    sol = het.run_simulation(in_file)
    params = Dict(
        :anode_mass_flow_rate_kg_s => sol.config.propellants[].flow_rate_kg_s,
        :neutral_velocity_m_s => sol.config.propellants[].velocity_m_s,
        :discharge_voltage_v => sol.config.discharge_voltage,
        :cathode_coupling_voltage_v => sol.config.cathode_coupling_voltage,
        :magnetic_field_scale => sol.config.magnetic_field_scale,
        :wall_loss_scale => sol.config.wall_loss_model.loss_scale,
    )

    sim_dict = save_sim(sol, params)
    output = load_single_sim(sim_dict)
    out_dict = Dict(
        "params" => Float32.(output.params[2]),
        "data" => Float32.(output.fields[2]),
        "fourier" => Float32.(output.fourier[2]),
        "perf" => Float32.(output.performance[2]),
    )

    NPZ.npzwrite(out_file, out_dict)
end