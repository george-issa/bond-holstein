using LinearAlgebra
using Random
using Printf
using MPI
using JLD2

using SmoQyDQMC
import SmoQyDQMC.LatticeUtilities as lu
import SmoQyDQMC.JDQMCFramework as dqmcf
import SmoQyDQMC.JDQMCMeasurements as dqmcm
import SmoQyDQMC.MuTuner as mt

############ Local real-space observables (densities & phonons) ############
# Bin-averaged; written once per bin as CSVs under:
#   <datafolder>/equal-time/real-space/local_density/bin-<bin>_pID-<pID>.csv
#   <datafolder>/equal-time/real-space/phonon_position/bin-<bin>_pID-<pID>.csv

# We treat the G as the single-spin equal-time Green's matrix.
# So n_per_spin(i) = 1 - G[ii], and n_total(i) = 2 * n_per_spin(i) (spin-degenerate).
# If truly spinful (distinct-up/down) matrices are written, replace accordingly.

mutable struct _LocalAcc
    N::Int
    n_meas::Int
    nsum_per_spin::Vector{Float64}  # length N
    xsum_mode1::Vector{Float64}     # length N (phonon_x)
    x2sum_mode1::Vector{Float64}
    xsum_mode2::Vector{Float64}     # length N (phonon_y)
    x2sum_mode2::Vector{Float64}
    L::Int
end

function _LocalAcc(L::Int)
    N = L^2
    _LocalAcc(N, 0, zeros(N), zeros(N), zeros(N), zeros(N), zeros(N), L)
end

@inline function _reset!(acc::_LocalAcc)
    fill!(acc.nsum_per_spin, 0.0)
    fill!(acc.xsum_mode1, 0.0); fill!(acc.x2sum_mode1, 0.0)
    fill!(acc.xsum_mode2, 0.0); fill!(acc.x2sum_mode2, 0.0)
    acc.n_meas = 0
end

# Map (x,y) -> site index in row-major (x:1..L, y:1..L)
@inline _sid(x,y,L) = (y-1)*L + x

# Accumulate one measurement:
# - electron density from diag(G)
# - phonon displacements: average over τ for each mode (mode 1 = x, mode 2 = y)
function _accumulate!(acc::_LocalAcc, G::AbstractMatrix, epp; Lτ::Int)
    L = acc.L
    # n_per_spin = 1 - G(ii)
    @inbounds for y in 1:L, x in 1:L
        i = _sid(x,y,L)
        acc.nsum_per_spin[i] += (1.0 - real(G[i,i]))
    end

    # The code elsewhere reshapes electron_phonon_parameters.x to (L,L,2,Lτ),
    # where the 3rd index = mode (1 => x-bond phonon, 2 => y-bond phonon).
    X = reshape(epp.x, (L, L, 2, Lτ))

    # per-site average over τ for each mode
    @inbounds for y in 1:L, x in 1:L
        i = _sid(x,y,L)
        # mode 1 (x)
        s1 = 0.0
        @inbounds @simd for τ in 1:Lτ
            s1 += X[x,y,1,τ]
        end
        m1 = s1 / Lτ
        acc.xsum_mode1[i]  += m1
        acc.x2sum_mode1[i] += m1*m1

        # mode 2 (y)
        s2 = 0.0
        @inbounds @simd for τ in 1:Lτ
            s2 += X[x,y,2,τ]
        end
        m2 = s2 / Lτ
        acc.xsum_mode2[i]  += m2
        acc.x2sum_mode2[i] += m2*m2
    end

    acc.n_meas += 1
    nothing
end

# csv writer
function _write_csv(path::AbstractString,
    header::AbstractVector{<:AbstractString},
    rows::AbstractVector{<:AbstractVector})
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(header, ","))
        for r in rows
            println(io, join(r, ","))
        end
    end
    @printf("[local-observables] wrote %s\n", path)
end

function _finalize_and_write!(acc::_LocalAcc, datafolder::String; bin::Int, pID::Int)
    @assert acc.n_meas > 0 "No local measurements accumulated in this bin."
    L, N = acc.L, acc.N
    fac = 1.0 / acc.n_meas

    # ---- densities (per spin and total) ----
    rows_n = Vector{Vector{Any}}(undef, N)
    idx = 1
    for y in 1:L, x in 1:L
        i = _sid(x,y,L)
        n_per_spin = acc.nsum_per_spin[i] * fac
        n_tot = 2.0 * n_per_spin
        rows_n[idx] = Any[x, y, i, n_per_spin, n_tot]; idx += 1
    end
    dens_dir = joinpath(datafolder, "local_density", "position",
                        @sprintf("bin-%d_pID-%d", bin, pID))
    _write_csv(joinpath(dens_dir, @sprintf("bin-%d_pID-%d.csv", bin, pID)),
               ["x","y","site","n_per_spin","n_total"],
               rows_n)

    # ---- phonons (mode 1 = x-bond, mode 2 = y-bond); mean & std over measurements ----
    rows_ph = Vector{Vector{Any}}(undef, N)
    idx = 1
    for y in 1:L, x in 1:L
        i = _sid(x,y,L)
        m1 = acc.xsum_mode1[i] * fac
        v1 = max(acc.x2sum_mode1[i] * fac - m1*m1, 0.0)
        s1 = sqrt(v1)
        m2 = acc.xsum_mode2[i] * fac
        v2 = max(acc.x2sum_mode2[i] * fac - m2*m2, 0.0)
        s2 = sqrt(v2)
        rows_ph[idx] = Any[x, y, i, m1, s1, m2, s2]; idx += 1
    end
    phon_dir = joinpath(datafolder, "phonon_position", "position",
                        @sprintf("bin-%d_pID-%d", bin, pID))
    _write_csv(joinpath(phon_dir, @sprintf("bin-%d_pID-%d.csv", bin, pID)),
               ["x","y","site","Xx_mean","Xx_std","Xy_mean","Xy_std"],
               rows_ph)

    # Console peek
    @printf("[local-observables] bin %d pID %d (n_meas=%d): site (1,1) -> n_tot=%.4f, Xx=%.4f, Xy=%.4f\n",
            bin, pID, acc.n_meas, rows_n[1][5], rows_ph[1][4], rows_ph[1][6])
    nothing
end
############ end local real-space observables ###########


# Define top-level function for running DQMC simulation
function run_bond_holstein_square_simulation(
    comm, sID,
    Ω, α, β, L,
    N_burnin, N_updates, N_bins, cdw_start;
    filepath = "."
)

    # datafolder name prefix
    datafolder_prefix = @sprintf "bond_holstein_square_w%.2f_a%.4f_b%.2f_L%d" Ω α β L

    # Get the MPI comm rank, which fixes the process ID (pID).
    pID = MPI.Comm_rank(comm)

    # Initialize an instance of the SimulationInfo type.
    simulation_info = SimulationInfo(
        filepath = filepath,
        datafolder_prefix = datafolder_prefix,
        sID = sID,
        pID = pID
    )

    # Define checkpoint filename.
    # We implement three checkpoint files, an old, current and new one,
    # that get cycled through to ensure a checkpoint file always exists in the off
    # chance that the simulation is killed while a checkpoint is getting written to file.
    # Additionally, each simulation that is running in parallel with MPI will have their own
    # checkpoints written to file.
    datafolder = simulation_info.datafolder
    sID        = simulation_info.sID
    pID        = simulation_info.pID
    checkpoint_name_old          = @sprintf "checkpoint_sID%d_pID%d_old.jld2" sID pID
    checkpoint_filename_old      = joinpath(datafolder, checkpoint_name_old)
    checkpoint_name_current      = @sprintf "checkpoint_sID%d_pID%d_current.jld2" sID pID
    checkpoint_filename_current  = joinpath(datafolder, checkpoint_name_current)
    checkpoint_name_new          = @sprintf "checkpoint_sID%d_pID%d_new.jld2" sID pID
    checkpoint_filename_new      = joinpath(datafolder, checkpoint_name_new)

    ######################################################
    ### DEFINE SOME RELEVANT DQMC SIMULATION PARAMETERS ##
    ######################################################

    # Set the discretization in imaginary time for the DQMC simulation.
    Δτ = 0.05

    # length of imaginary time axis
    Lτ = round(Int, β/Δτ)

    # This flag indicates whether or not to use the checkboard approximation to
    # represent the exponentiated hopping matrix exp(-Δτ⋅K)
    checkerboard = false

    # Whether the propagator matrices should be represented using the
    # symmetric form B = exp(-Δτ⋅K/2)⋅exp(-Δτ⋅V)⋅exp(-Δτ⋅K/2)
    # or the asymetric form B = exp(-Δτ⋅V)⋅exp(-Δτ⋅K)
    symmetric = false

    # Set the initial period in imaginary time slices with which the Green's function matrices
    # will be recomputed using a numerically stable procedure.
    n_stab = 10

    # Specify the maximum allowed error in any element of the Green's function matrix that is
    # corrected by performing numerical stabiliziation.
    δG_max = 1e-6

    # Number of fermionic time-steps in HMC update.
    Nt = 10

    # Fermionic time-step used in HMC update.
    Δt = (π/(2*Ω))/Nt

    # Regularization parameter used in HMC udpate.
    reg = 0.0

    # Initialize variables to keep track of the largest numerical error in the
    # Green's function matrices corrected by numerical stabalization.
    δG = 0.0
    δθ = 0.0

    #######################
    ### DEFINE THE MODEL ##
    #######################

    # Initialize an instance of the type UnitCell.
    unit_cell = lu.UnitCell(lattice_vecs = [[1.0, 0.0],
                                            [0.0, 1.0]],
                            basis_vecs   = [[0.0, 0.0]])

    # Initialize an instance of the type Lattice.
    lattice = lu.Lattice(
        L = [L, L],
        periodic = [true, true]
    )

    # Number of sites in the lattice.
    N = L^2

    # Initialize an instance of the ModelGeometry type.
    model_geometry = ModelGeometry(unit_cell, lattice)

    # Define the nearest-neighbor bond in +x direction
    bond_px = lu.Bond(orbitals = (1,1), displacement = [1,0])

    # Add this bond to the model, by adding it to the ModelGeometry type.
    bond_px_id = add_bond!(model_geometry, bond_px)

    # Define the nearest-neighbor bond in +y direction
    bond_py = lu.Bond(orbitals = (1,1), displacement = [0,1])

    # Add this bond to the model, by adding it to the ModelGeometry type.
    bond_py_id = add_bond!(model_geometry, bond_py)

    # Define the nearest-neighbor bond in -x direction
    bond_nx = lu.Bond(orbitals = (1,1), displacement = [-1,0])

    # Add this bond to the model, by adding it to the ModelGeometry type.
    bond_nx_id = add_bond!(model_geometry, bond_nx)

    # Define the nearest-neighbor bond in -y direction
    bond_ny = lu.Bond(orbitals = (1,1), displacement = [0,-1])

    # Add this bond to the model, by adding it to the ModelGeometry type.
    bond_ny_id = add_bond!(model_geometry, bond_ny)

    # Define nearest-neighbor hopping amplitude, setting the energy scale for the system.
    t = 1.0

    # Define the tight-binding model
    tight_binding_model = TightBindingModel(
        model_geometry = model_geometry,
        t_bonds = [bond_px, bond_py], # defines hopping
        t_mean = [t, t], # defines corresponding hopping amplitude
        μ = 0.0, # set chemical potential
        ϵ_mean = [0.] # set the (mean) on-site energy
    )

    # Initialize a null electron-phonon model.
    electron_phonon_model = ElectronPhononModel(
        model_geometry = model_geometry,
        tight_binding_model = tight_binding_model
    )

    # Define a dispersionless electron-phonon mode for x bond.
    phonon_x = PhononMode(orbital = 1, Ω_mean = Ω)

    # Add the phonon mode definition to the electron-phonon model.
    phonon_x_id = add_phonon_mode!(
        electron_phonon_model = electron_phonon_model,
        phonon_mode = phonon_x
    )

    # Define a dispersionless electron-phonon mode for y bond.
    phonon_y = PhononMode(orbital = 1, Ω_mean = Ω)

    # Add the phonon mode definition to the electron-phonon model.
    phonon_y_id = add_phonon_mode!(
        electron_phonon_model = electron_phonon_model,
        phonon_mode = phonon_y
    )

    # Define first holstein coupling for phonon mode living on the x bond.
    holstein_px_coupling = HolsteinCoupling(
    	model_geometry = model_geometry,
    	phonon_mode = phonon_x_id,
    	bond = bond_px,
    	α_mean = α,
        shifted = false
    )

    # Add the Holstein coupling definition to the model.
    holstein_px_coupling_id = add_holstein_coupling!(
    	electron_phonon_model = electron_phonon_model,
    	holstein_coupling = holstein_px_coupling,
    	model_geometry = model_geometry
    )

    # Define second holstein coupling for phonon mode living on the x bond.
    holstein_nx_coupling = HolsteinCoupling(
    	model_geometry = model_geometry,
    	phonon_mode = phonon_x_id,
    	bond = lu.Bond(orbitals = (1,1), displacement = [0,0]),
    	α_mean = -α,
        shifted = false
    )

    # Add the Holstein coupling definition to the model.
    holstein_nx_coupling_id = add_holstein_coupling!(
    	electron_phonon_model = electron_phonon_model,
    	holstein_coupling = holstein_nx_coupling,
    	model_geometry = model_geometry
    )

    # Define first holstein coupling for phonon mode living on the y bond.
    holstein_py_coupling = HolsteinCoupling(
    	model_geometry = model_geometry,
    	phonon_mode = phonon_y_id,
    	bond = bond_py,
    	α_mean = α,
        shifted = false
    )

    # Add the Holstein coupling definition to the model.
    holstein_py_coupling_id = add_holstein_coupling!(
    	electron_phonon_model = electron_phonon_model,
    	holstein_coupling = holstein_py_coupling,
    	model_geometry = model_geometry
    )

    # Define second holstein coupling for phonon mode living on the y bond.
    holstein_ny_coupling = HolsteinCoupling(
    	model_geometry = model_geometry,
    	phonon_mode = phonon_y_id,
    	bond = lu.Bond(orbitals = (1,1), displacement = [0,0]),
    	α_mean = -α,
        shifted = false
    )

    # Add the Holstein coupling definition to the model.
    holstein_ny_coupling_id = add_holstein_coupling!(
    	electron_phonon_model = electron_phonon_model,
    	holstein_coupling = holstein_ny_coupling,
    	model_geometry = model_geometry
    )

    # Initialize the accumulator for local observables
    local_acc = _LocalAcc(L)

    #######################################################
    ### BRANCHING BEHAVIOR BASED ON WHETHER STARTING NEW ##
    ### SIMULAIOTN OR RESUMING PREVIOUS SIMULATION.      ##
    #######################################################

    # Synchronize all the MPI processes.
    MPI.Barrier(comm)

    # If starting a new simulation.
    if !simulation_info.resuming

        # Initialize a random number generator that will be used throughout the simulation.
        seed = abs(rand(Int))
        rng = Xoshiro(seed)

        # Initialize the directory the data will be written to.
        initialize_datafolder(simulation_info)

        # Write the model summary to file.
        model_summary(
            simulation_info = simulation_info,
            β = β, Δτ = Δτ,
            model_geometry = model_geometry,
            tight_binding_model = tight_binding_model,
            interactions = (electron_phonon_model,)
        )

        # Calculate the bins size.
        bin_size = div(N_updates, N_bins)

        # Initialize a dictionary to store additional information about the simulation.
        additional_info = Dict(
            "dG_max" => δG_max,
            "N_burnin" => N_burnin,
            "N_updates" => N_updates,
            "N_bins" => N_bins,
            "bin_size" => bin_size,
            "hmc_acceptance_rate" => 0.0,
            "reflection_acceptance_rate" => 0.0,
            "swap_acceptance_rate" => 0.0,
            "n_stab" => n_stab,
            "symmetric" => symmetric,
            "checkerboard" => checkerboard,
            "seed" => seed,
            "cdw_start" => cdw_start
        )

        #########################################
        ### INITIALIZE FINITE MODEL PARAMETERS ##
        #########################################


        # Initialize tight-binding parameters.
        tight_binding_parameters = TightBindingParameters(
            tight_binding_model = tight_binding_model,
            model_geometry = model_geometry,
            rng = rng
        )

        # Initialize electron-phonon parameters.
        electron_phonon_parameters = ElectronPhononParameters(
            β = β, Δτ = Δτ,
            electron_phonon_model = electron_phonon_model,
            tight_binding_parameters = tight_binding_parameters,
            model_geometry = model_geometry,
            rng = rng
        )

        # start the simulation from a cdw state
        if cdw_start
            X = reshape(electron_phonon_parameters.x, (L, L, 2, Lτ))
            for y in 1:L, x in 1:L
                X_xy = (randn(rng) + 2*(-1.0)^(x+y))/sqrt(2*Ω)
                @. X[x, y, 1, :] = X_xy
                Y_xy = (randn(rng) + 2*(-1.0)^(x+y))/sqrt(2*Ω)
                @. X[x, y, 2, :] = Y_xy
            end
        end

        ##############################
        ### INITIALIZE MEASUREMENTS ##
        ##############################

        # Initialize the container that measurements will be accumulated into.
        measurement_container = initialize_measurement_container(model_geometry, β, Δτ)

        # Initialize the tight-binding model related measurements, like the hopping energy.
        initialize_measurements!(measurement_container, tight_binding_model)

        # Initialize the electron-phonon related measurements.
        initialize_measurements!(measurement_container, electron_phonon_model)

        # Initialize the single-particle electron Green's function measurement.
        initialize_correlation_measurements!(
            measurement_container = measurement_container,
            model_geometry = model_geometry,
            correlation = "greens",
            time_displaced = true,
            pairs = [(1, 1)]
        )

        # Initialize phonon Green's function measurement.
        initialize_correlation_measurements!(
            measurement_container = measurement_container,
            model_geometry = model_geometry,
            correlation = "phonon_greens",
            time_displaced = false,
            integrated = true,
            pairs = [
                (phonon_x_id, phonon_x_id),
                (phonon_y_id, phonon_y_id)
            ]
        )

        # Initialize density correlation function measurement.
        initialize_correlation_measurements!(
            measurement_container = measurement_container,
            model_geometry = model_geometry,
            correlation = "density",
            time_displaced = false,
            integrated = true,
            pairs = [(1, 1)]
        )

        # Initialize the pair correlation function measurement.
        initialize_correlation_measurements!(
            measurement_container = measurement_container,
            model_geometry = model_geometry,
            correlation = "pair",
            time_displaced = false,
            integrated = true,
            pairs = [
                (1, 1), (bond_px_id, bond_px_id), (bond_py_id, bond_py_id)
            ]
        )

        # Initialize the spin-z correlation function measurement.
        initialize_correlation_measurements!(
            measurement_container = measurement_container,
            model_geometry = model_geometry,
            correlation = "spin_z",
            time_displaced = false,
            integrated = true,
            pairs = [(1, 1)]
        )

        # Initialize the d-wave pair susceptibility measurement.
        initialize_composite_correlation_measurement!(
            measurement_container = measurement_container,
            model_geometry = model_geometry,
            name = "d-wave",
            correlation = "pair",
            ids = [bond_px_id, bond_nx_id, bond_py_id, bond_ny_id],
            coefficients = [0.5, 0.5, -0.5, -0.5],
            time_displaced = false,
            integrated = true
        )

        # Initialize the extended s-wave pair susceptibility measurement.
        initialize_composite_correlation_measurement!(
            measurement_container = measurement_container,
            model_geometry = model_geometry,
            name = "ext-s-wave",
            correlation = "pair",
            ids = [bond_px_id, bond_nx_id, bond_py_id, bond_ny_id],
            coefficients = [0.5, 0.5, 0.5, 0.5],
            time_displaced = false,
            integrated = true
        )

        # Initialize the sub-directories to which the various measurements will be written.
        initialize_measurement_directories(
            simulation_info = simulation_info,
            measurement_container = measurement_container
        )

        #############################
        ### WRITE FIRST CHECKPOINT ##
        #############################

        # Calculate the bin size.
        bin_size = div(N_updates, N_bins)

        # Calculate the number of thermalization/burnin bins.
        # This determines the number times the simulations checkpoints
        # during the initial thermalziation/burnin period.
        N_bins_burnin = div(N_burnin, bin_size)

        # Initialize variable to keep track of the current burnin bin.
        n_bin_burnin = 1

        # Initialize variable to keep track of the current bin.
        n_bin = 1

        # Write an initial checkpoint to file.
        JLD2.jldsave(
            checkpoint_filename_current;
            rng, additional_info,
            N_burnin, N_updates, N_bins, bin_size,
            N_bins_burnin, n_bin_burnin, n_bin,
            measurement_container,
            model_geometry,
            tight_binding_parameters,
            electron_phonon_parameters,
            dG = δG,
            dtheta = δθ,
            n_stab = n_stab
        )

    # If resuming simulation from previous checkpoint.
    else

        # Initialize checkpoint to nothing before it is loaded.
        checkpoint = nothing

        # Try loading in the new checkpoint.
        if isfile(checkpoint_filename_new)
            try
                # Load the new checkpoint.
                checkpoint = JLD2.load(checkpoint_filename_new)
            catch
                nothing
            end
        end

        # Try loading in the current checkpoint.
        if isfile(checkpoint_filename_current) && isnothing(checkpoint)
            try
                # Load the current checkpoint.
                checkpoint = JLD2.load(checkpoint_filename_current)
            catch
                nothing
            end
        end

        # Try loading in the current checkpoint.
        if isfile(checkpoint_filename_old) && isnothing(checkpoint)
            try
                # Load the old checkpoint.
                checkpoint = JLD2.load(checkpoint_filename_old)
            catch
                nothing
            end
        end

        # Throw an error if no checkpoint was succesfully loaded.
        if isnothing(checkpoint)
            error("Failed to load checkpoint successfully!")
        end

        # Unpack the contents of the checkpoint.
        rng                        = checkpoint["rng"]
        additional_info            = checkpoint["additional_info"]
        N_burnin                   = checkpoint["N_burnin"]
        N_updates                  = checkpoint["N_updates"]
        N_bins                     = checkpoint["N_bins"]
        bin_size                   = checkpoint["bin_size"]
        N_bins_burnin              = checkpoint["N_bins_burnin"]
        n_bin_burnin               = checkpoint["n_bin_burnin"]
        n_bin                      = checkpoint["n_bin"]
        model_geometry             = checkpoint["model_geometry"]
        measurement_container      = checkpoint["measurement_container"]
        tight_binding_parameters   = checkpoint["tight_binding_parameters"]
        electron_phonon_parameters = checkpoint["electron_phonon_parameters"]
        δG                         = checkpoint["dG"]
        δθ                         = checkpoint["dtheta"]
        n_stab                     = checkpoint["n_stab"]
    end

    # Synchronize all the MPI processes.
    MPI.Barrier(comm)

    #############################
    ### SET-UP DQMC SIMULATION ##
    #############################

    # Allocate fermion path integral type.
    fermion_path_integral = FermionPathIntegral(tight_binding_parameters = tight_binding_parameters, β = β, Δτ = Δτ)

    # Initialize the fermion path integral type with respect to electron-phonon interaction.
    initialize!(fermion_path_integral, electron_phonon_parameters)

    # Allocate and initialize propagators for each imaginary time slice.
    B = initialize_propagators(fermion_path_integral, symmetric=symmetric, checkerboard=checkerboard)

    # Initialize fermion greens calculator.
    fermion_greens_calculator = dqmcf.FermionGreensCalculator(B, β, Δτ, n_stab)

    # Initialize alternate fermion greens calculator required for performing various global updates.
    fermion_greens_calculator_alt = dqmcf.FermionGreensCalculator(fermion_greens_calculator)

    # Allocate equal-time Green's function matrix.
    G = zeros(eltype(B[1]), size(B[1]))

    # Initialize equal-time Green's function matrix
    logdetG, sgndetG = dqmcf.calculate_equaltime_greens!(G, fermion_greens_calculator)

    # Allocate matrices for various time-displaced Green's function matrices.
    G_ττ = similar(G) # G(τ,τ)
    G_τ0 = similar(G) # G(τ,0)
    G_0τ = similar(G) # G(0,τ)

    # Initialize variables to keep track of the largest numerical error in the
    # Green's function matrices corrected by numerical stabalization.
    δG = zero(typeof(logdetG))
    δθ = zero(typeof(sgndetG))

    # Initialize Hamitlonian/Hybrid monte carlo (HMC) updater.
    hmc_updater = EFAHMCUpdater(
        electron_phonon_parameters = electron_phonon_parameters,
        G = G, Nt = Nt, Δt = Δt, reg = reg
    )

    ####################################
    ### BURNIN/THERMALIZATION UPDATES ##
    ####################################

    # Iterate over burnin/thermalization bins.
    for bin in n_bin_burnin:N_bins_burnin

        # Iterate over updates in current bin.
        for n in 1:bin_size

            # Perform a reflection update.
            (accepted, logdetG, sgndetG) = reflection_update!(
                G, logdetG, sgndetG, electron_phonon_parameters,
                fermion_path_integral = fermion_path_integral,
                fermion_greens_calculator = fermion_greens_calculator,
                fermion_greens_calculator_alt = fermion_greens_calculator_alt,
                B = B, rng = rng, phonon_types = (phonon_x_id, phonon_y_id)
            )

            # Record whether the reflection update was accepted or rejected.
            additional_info["reflection_acceptance_rate"] += accepted

            # perform swap update
            (accepted, logdetG, sgndetG) = swap_update!(
                G, logdetG, sgndetG, electron_phonon_parameters,
                fermion_path_integral = fermion_path_integral,
                fermion_greens_calculator = fermion_greens_calculator,
                fermion_greens_calculator_alt = fermion_greens_calculator_alt,
                B = B, rng = rng,
                phonon_type_pairs = (
                    (phonon_x_id, phonon_x_id),
                    (phonon_y_id, phonon_y_id),
                    (phonon_x_id, phonon_y_id),
                )
            )

            # Record whether the swap update was accepted or rejected.
            additional_info["swap_acceptance_rate"] += accepted

            # Perform an HMC update.
            (accepted, logdetG, sgndetG, δG, δθ) = hmc_update!(
                G, logdetG, sgndetG, electron_phonon_parameters, hmc_updater,
                fermion_path_integral = fermion_path_integral,
                fermion_greens_calculator = fermion_greens_calculator,
                fermion_greens_calculator_alt = fermion_greens_calculator_alt,
                B = B, δG_max = δG_max, δG = δG, δθ = δθ, rng = rng,
                update_stabilization_frequency = false
            )

            # Record whether the HMC update was accepted or rejected.
            additional_info["hmc_acceptance_rate"] += accepted
        end

        # Write the new checkpoint to file.
        JLD2.jldsave(
            checkpoint_filename_new;
            rng, additional_info,
            N_burnin, N_updates, N_bins,
            bin_size, N_bins_burnin,
            n_bin_burnin = bin + 1,
            n_bin = 1,
            measurement_container,
            model_geometry,
            tight_binding_parameters,
            electron_phonon_parameters,
            dG = δG, dtheta = δθ, n_stab = n_stab
        )
        # Make the current checkpoint the old checkpoint.
        mv(checkpoint_filename_current, checkpoint_filename_old, force = true)
        # Make the new checkpoint the current checkpoint.
        mv(checkpoint_filename_new, checkpoint_filename_current, force = true)
    end

    ################################
    ### START MAKING MEAUSREMENTS ##
    ################################

    # Iterate over the number of bin, i.e. the number of time measurements will be dumped to file.
    for bin in n_bin:N_bins

        # Reset the measurement container.
        _reset!(local_acc)

        # Iterate over the number of updates and measurements performed in the current bin.
        for n in 1:bin_size

            # Perform a reflection update.
            (accepted, logdetG, sgndetG) = reflection_update!(
                G, logdetG, sgndetG, electron_phonon_parameters,
                fermion_path_integral = fermion_path_integral,
                fermion_greens_calculator = fermion_greens_calculator,
                fermion_greens_calculator_alt = fermion_greens_calculator_alt,
                B = B, rng = rng, phonon_types = (phonon_x_id, phonon_y_id)
            )

            # Record whether the reflection update was accepted or rejected.
            additional_info["reflection_acceptance_rate"] += accepted

            # perform swap update
            (accepted, logdetG, sgndetG) = swap_update!(
                G, logdetG, sgndetG, electron_phonon_parameters,
                fermion_path_integral = fermion_path_integral,
                fermion_greens_calculator = fermion_greens_calculator,
                fermion_greens_calculator_alt = fermion_greens_calculator_alt,
                B = B, rng = rng,
                phonon_type_pairs = (
                    (phonon_x_id, phonon_x_id),
                    (phonon_y_id, phonon_y_id),
                    (phonon_x_id, phonon_y_id),
                )
            )

            # Record whether the swap update was accepted or rejected.
            additional_info["swap_acceptance_rate"] += accepted

            # Perform an HMC update.
            (accepted, logdetG, sgndetG, δG, δθ) = hmc_update!(
                G, logdetG, sgndetG, electron_phonon_parameters, hmc_updater,
                fermion_path_integral = fermion_path_integral,
                fermion_greens_calculator = fermion_greens_calculator,
                fermion_greens_calculator_alt = fermion_greens_calculator_alt,
                B = B, δG_max = δG_max, δG = δG, δθ = δθ, rng = rng,
                update_stabilization_frequency = false
            )

            # Record whether the HMC update was accepted or rejected.
            additional_info["hmc_acceptance_rate"] += accepted

            # Make measurements.
            (logdetG, sgndetG, δG, δθ) = make_measurements!(
                measurement_container,
                logdetG, sgndetG, G, G_ττ, G_τ0, G_0τ,
                fermion_path_integral = fermion_path_integral,
                fermion_greens_calculator = fermion_greens_calculator,
                B = B, δG_max = δG_max, δG = δG, δθ = δθ,
                model_geometry = model_geometry, tight_binding_parameters = tight_binding_parameters,
                coupling_parameters = (electron_phonon_parameters,)
            )
            
            # Accumulate local real-space observables for this sweep
            _accumulate!(local_acc, G, electron_phonon_parameters; Lτ=Lτ)

        end

        # Write the average measurements for the current bin to file.
        write_measurements!(
            measurement_container = measurement_container,
            simulation_info = simulation_info,
            model_geometry = model_geometry,
            bin = bin,
            bin_size = bin_size,
            Δτ = Δτ
        )

        # Write the local real-space observables for this bin to file
        _finalize_and_write!(local_acc, simulation_info.datafolder; bin=bin, pID=pID)

        # Write the new checkpoint to file.
        JLD2.jldsave(
            checkpoint_filename_new;
            rng, additional_info,
            N_burnin, N_updates, N_bins,
            bin_size, N_bins_burnin,
            n_bin_burnin = N_bins_burnin+1,
            n_bin = bin + 1,
            measurement_container,
            model_geometry,
            tight_binding_parameters,
            electron_phonon_parameters,
            dG = δG, dtheta = δθ, n_stab = n_stab
        )
        # Make the current checkpoint the old checkpoint.
        mv(checkpoint_filename_current, checkpoint_filename_old, force = true)
        # Make the new checkpoint the current checkpoint.
        mv(checkpoint_filename_new, checkpoint_filename_current, force = true)
    end

    # Calculate acceptance rate for HMC updates.
    additional_info["hmc_acceptance_rate"] /= (N_updates + N_burnin)

    # Calculate acceptance rate for reflection updates.
    additional_info["reflection_acceptance_rate"] /= (N_updates + N_burnin)

    # Calculate acceptance rate for reflection updates.
    additional_info["swap_acceptance_rate"] /= (N_updates + N_burnin)

    # Record the maximum numerical error corrected by numerical stablization.
    additional_info["dG"] = δG

    # Write simulation summary TOML file.
    save_simulation_info(simulation_info, additional_info)

    #################################
    ### PROCESS SIMULATION RESULTS ##
    #################################

    # Synchronize all the MPI processes.
    MPI.Barrier(comm)

    # Process results
    process_measurements(comm, simulation_info.datafolder, N_bins, time_displaced = true)

    # compress JLD2 binary files
    # compress_jld2_bins(comm, folder = simulation_info.datafolder)

    return nothing
end

# Example Terminal Command To Run Script On A Single Process:
# > julia bond_holstein_square.jl 0 1.0 0.4 4.0 4 5000 10000 50 0

# Example Terminal Command To Run Script With 8 Walker in Parallel with MPI:
# > mpiexecjl -n 8 bond_julia holstein_square.jl 0 1.0 0.4 4.0 4 5000 10000 50 0

# Only excute if script is run directly from the command line.
if abspath(PROGRAM_FILE) == @__FILE__

    # Initialize the MPI communicator.
    comm = MPI.COMM_WORLD

    # Initialize MPI
    MPI.Init()

    # Read in the command line arguments.

    # specify simulation ID
    sID = parse(Int, ARGS[1])

    # specify phonon frequency
    Ω = parse(Float64, ARGS[2])

    # electron-phonon coupling
    α = parse(Float64, ARGS[3])

    # specify inverse temperature β = 1/T
    β = parse(Float64, ARGS[4])

    # specify system size, total number of size is N = L^2
    L = parse(Int, ARGS[5])

    # specify number of burnin updates performed to theramlize system.
    N_burnin = parse(Int, ARGS[6])

    # specify number of updates and measurements made once the system is theramlized.
    N_updates = parse(Int, ARGS[7])

    # specify the number of time measurements will be written to file
    N_bins = parse(Int, ARGS[8])

    # whether to start the simulation from a cdw state
    cdw_start = parse(Bool, ARGS[9])

    # ensure the total number of measurements made is divisible by the number of time
    # measurements are written to file
    @assert N_updates % N_bins == 0

    # Run the simulation.
    run_bond_holstein_square_simulation(comm, sID, Ω, α, β, L, N_burnin, N_updates, N_bins, cdw_start)

    # Finalize MPI (not strictly required).
    MPI.Finalize()
end