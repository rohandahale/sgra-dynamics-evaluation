"""
Julia Version 1.10.4
Commit 48d4fd48430 (2024-06-04 10:41 UTC)
Build Info:
  Official https://julialang.org/ release
Platform Info:
  OS: Linux (x86_64-linux-gnu)
  CPU: 112 × AMD EPYC 7B13
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-15.0.7 (ORCJIT, znver3)
Threads: 1 default, 0 interactive, 1 GC (on 112 virtual cores)

  [c7e460c6] ArgParse v1.2.0
  [336ed68f] CSV v0.10.14
⌃ [13f3f980] CairoMakie v0.11.11
  [863f3e99] Comonicon v1.0.8
⌃ [99d987ce] Comrade v0.9.4
  [a93c6f00] DataFrames v1.6.1
  [8bb1440f] DelimitedFiles v1.9.1
  [c27321d9] Glob v1.3.1
⌃ [3e6eede4] OptimizationBBO v0.2.1
  [bd407f91] OptimizationCMAEvolutionStrategy v0.2.1
  [3aafef2f] OptimizationMetaheuristics v0.2.0
⌃ [4096cdfb] VIDA v0.11.7
  [9a3f8284] Random
"""

using Pkg; Pkg.activate(@__DIR__)
using Distributed
@everywhere using Pkg
@everywhere Pkg.activate(@__DIR__)
using ArgParse
using DelimitedFiles
using DataFrames
using CSV
using Comrade
using VIDA
using CSV
using DataFrames
using VIDA
using OptimizationMetaheuristics: OptimizationMetaheuristics, ECA, Options

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--imfiles"
            help = "The file containing the absolute paths to all images"
            arg_type = String
            required = true
        "--outname"
            help = "output csv file with extractions"
            arg_type = String
            required = true
        "--stride"
             help = "Checkpointing stride, i.e. number of steps."
             arg_type = Int
             default = 8
        "--fevals"
             help = "The number of evaluations allowed when extracting params"
             arg_type = Int
             default = 20000
        "--blur"
             help = "Blur images before extracting fitted model parameters"
             arg_type = Float64
             default = 0.0
        "--restart"
            help = "Tells the sampler to read in the old frames and restart the run."
            action = :store_true
        "--regrid"
            help = "To not regrid the images before extracting"
            action = :store_true
        "--order"
            help = "The order of the ring parameters"
            arg_type = Int
            default = 4
    end
    return parse_args(s)
end


addprocs(8)

@everywhere begin
    using ArgParse
    using Distributed
    using DelimitedFiles
    using DataFrames
    using CSV
    using Comrade
    using VIDA
    using CSV
    using DataFrames
    using VIDA
    using OptimizationMetaheuristics: OptimizationMetaheuristics, ECA, Options
end


function loaddir(file)
    open(file) do io
        return readlines(io)
    end
end


@everywhere  function lpmodes(img::IntensityMap{<:StokesParams}, modes::NTuple{<:Any, Int}; rmin=0, rmax=μas2rad(60.0))
    I = stokes(img, :I)
    lp = linearpol.(img)
    g = imagegrid(img)
    r = splat(hypot).(values.(g))
    ang(p) = atan(-p.X, p.Y)
    θ = ang.(g)

    mask = (r .<= rmax).&(r .>= rmin)
    flux = sum(I[mask])
    betas = map(modes) do m
        pbasis = cis.(m.*θ)
        prod = lp.*pbasis
        coef = sum(prod[mask])
        return coef/flux
    end
    return betas
end

@everywhere function cpmodes(img::IntensityMap{<:StokesParams}, modes::NTuple{<:Any, Int}; rmin=0, rmax=μas2rad(60.0))
    I = stokes(img, :I)
    lp = stokes(img, :V)
    g = imagegrid(img)
    r = splat(hypot).(values.(g))
    ang(p) = atan(-p.X, p.Y)
    θ = ang.(g)

    mask = (r .<= rmax).&(r .>= rmin)
    flux = sum(I[mask])
    betas = map(modes) do m
        pbasis = cis.(m.*θ)
        prod = lp.*pbasis
        coef = sum(prod[mask])
        return coef/flux
    end
    return betas
end

@everywhere mnet(img::AbstractArray{<:StokesParams}) = abs(sum(linearpol, img))/sum(stokes(img, :I))

@everywhere vnet(img::AbstractArray{<:StokesParams}) = sum(stokes(img, :V))/sum(stokes(img, :I))

@everywhere mavg(img::AbstractArray{<:StokesParams}) = sum(abs.(linearpol.(img)))/sum(stokes(img, :I))

@everywhere vavg(img::AbstractArray{<:StokesParams}) = sum(abs.(stokes(img, :V)))/sum(stokes(img, :I))


@everywhere function center_ring(img::IntensityMap{<:Real}; order=1, maxiters=10_000)
    xopt, θopt = find_ring_center(img; order, maxiters)
    return θopt, xopt, shifted(img, -xopt.x0, -xopt.y0)
end

@everywhere function center_ring(img::IntensityMap{<:StokesParams}; order=1, maxiters=10_000)
    xopt, θopt = find_ring_center(stokes(img, :I); order, maxiters)
    return θopt, xopt, shifted(img, -xopt.x0, -xopt.y0)
end

@everywhere function find_ring_center(img::IntensityMap{<:Real}; order=1, maxiters=10_000)
    bh = LeastSquares(max.(img, 0.0))

    x0, y0 = centroid(img)

    ring(x) = modify(RingTemplate(RadialGaussian(x.σ/x.r0), AzimuthalCosine(x.s, x.ξ)),
                     Stretch(x.r0, x.r0), Shift(x.x0, x.y0))+
              modify(Gaussian(), Stretch(x.σg), Shift(x.xg, x.yg), Renormalize(x.fg)) +
              x.f0*Constant(fieldofview(img).X)
    lower = (r0 = μas2rad(15.0), σ = μas2rad(1.0),
             s = ntuple(_->0.001, order),
             ξ = ntuple(_->0.0, order),
             x0 = -μas2rad(15.0), y0 = -μas2rad(15.0),
             σg = μas2rad(30.0),
             xg = -fieldofview(img).X/4,
             yg = -fieldofview(img).Y/4,
             fg = 1e-6, f0 = 1e-6
             )
    upper = (r0 = μas2rad(40.0), σ = μas2rad(15.0),
             s = ntuple(_->0.999, order),
             ξ = ntuple(_->2π, order),
             x0 = μas2rad(15.0), y0 = μas2rad(15.0),
             σg = fieldofview(img).X/2,
             xg = fieldofview(img).X/4,
             yg = fieldofview(img).Y/4,
             fg = 20.0, f0=10.0
             )
    p0 = (r0 = μas2rad(20.0), σ = μas2rad(4.0),
             s = ntuple(_->0.01, order),
             ξ = ntuple(_->1π, order),
            x0 = x0, y0 = y0,
          σg = μas2rad(40.0), xg = 0.0, yg = 0.0, fg = 0.2, f0=1e-3)
    prob = VIDAProblem(bh, ring, lower, upper)
    xopt, θopt, _ = vida(prob, ECA(;options=Options(f_calls_limit = maxiters, f_tol = 1e-5)); init_params=p0)
    return xopt, θopt
end

function match_center_and_res(target::IntensityMap{<:StokesParams}, input::IntensityMap{<:StokesParams})
    target_I = stokes(target, :I)
    input_I = stokes(input, :I)
    _, xopt = match_center_and_res(target_I, input_I)
    return Comrade.smooth(shifted(input, xopt.x, xopt.y), xopt.σ), xopt
end

function match_center_and_res(target::IntensityMap, input::IntensityMap)
    cache = create_cache(FFTAlg(), input)
    f = let cimg=ContinuousImage(input, cache)
        function f(x)
            return smoothed(shifted(cimg, x.x, x.y), x.σ)
        end
    end
    div = NxCorr(target)
    lower = (x=-μas2rad(20.0), y=-μas2rad(20.0), σ=μas2rad(1.0))
    upper = (x=μas2rad(20.0), y=μas2rad(20.0), σ=μas2rad(30.0))
    p0 = (x=0.0, y=0.0, σ=μas2rad(10.0))

    prob = VIDAProblem(div, f, lower, upper)
    xopt, θopt, divmin = vida(prob, ECA(;options=Options(f_calls_limit = 2000, f_tol = 1e-5)); init_params=p0, maxiters=800)
    return Comrade.smooth(shifted(input, xopt.x, xopt.y), xopt.σ), xopt
end

@everywhere function summary_ringparams(img::IntensityMap{<:StokesParams};
                            lpmode=(2,), cpmode=(1,),
                            order=1, maxiters=1000)
    xopt = summary_ringparams(stokes(img, :I); order, maxiters)
    simg = shifted(img, -xopt.x0, -xopt.y0)
    m_net = mnet(simg)
    v_net = vnet(simg)

    m_avg = mavg(simg)
    v_avg = vavg(simg)

    βlp = lpmodes(simg, lpmode)
    βcp = cpmodes(simg, cpmode)

    n_relp = Tuple((Symbol("re_betalp", "_$n") for n in lpmode))
    real_betalp = NamedTuple{n_relp}(map(real, βlp))
    n_imlp = Tuple((Symbol("im_betalp", "_$n") for n in lpmode))
    imag_betalp = NamedTuple{n_imlp}(map(imag, βlp))

    n_recp = Tuple((Symbol("re_betacp", "_$n") for n in cpmode))
    real_betacp = NamedTuple{n_recp}(map(real, βcp))
    n_imcp = Tuple((Symbol("im_betacp", "_$n") for n in cpmode))
    imag_betacp = NamedTuple{n_imcp}(map(imag, βcp))

    return merge(xopt, (;m_net, m_avg), real_betalp, imag_betalp, (;v_net, v_avg), real_betacp, imag_betacp)
end

@everywhere function summary_ringparams(img::IntensityMap{<:Real};
                            lpmodes=(2,), cpmodes=(1,),
                            order=1, maxiters=1000)
    rimg = regrid(img, imagepixels(μas2rad(120.0), μas2rad(120.0), 32, 32))
    _, xopt, _ = center_ring(rimg; order, maxiters)
    return _flatten_tuple(xopt)
end

@everywhere  function _flatten_tuple(nt::NamedTuple)
    names = keys(nt)
    vals = values(nt)
    mapreduce(merge, zip(names, vals)) do (name, value)
        return _flatten_tuple(name, value)
    end
end

@everywhere  function _flatten_tuple(name::Symbol, a)
    return NamedTuple{(name,)}((a,))
end

@everywhere  function _flatten_tuple(name::Symbol, t::NTuple{N}) where {N}
    names = Tuple((Symbol("$(name)_$i") for i in 1:N))
    return NamedTuple{names}(t)
end

function main()
    parsed_args = parse_commandline()

    imfiles = parsed_args["imfiles"]
    outname = parsed_args["outname"]
    stride = parsed_args["stride"]
    blur = parsed_args["blur"]
    restart = parsed_args["restart"]
    regrid = parsed_args["regrid"]
    order = parsed_args["order"]
    fevals = parsed_args["fevals"]

    @info "Image files path: $(imfiles)"
    @info "Outputting results to $(outname)"
    @info "Using a $(order) order ring model"

    imfs = loaddir(imfiles)
    @info "Loaded $(length(imfs)) files"

    g = imagepixels(μas2rad(200.0), μas2rad(200.0), 64, 64)

    # Flip this because by default we want to regrid
    regrid = !regrid

    @info "Regridding image : $(regrid)"
    @info "Blurring kernel: $(blur) μas"

    @info "Starting to extract summary statistics"
    if restart
        df = CSV.read(outname, DataFrame)
    else
        df = DataFrame()
    end
    startindex = nrow(df)+1
    indexpart = Iterators.partition(startindex:length(imfs), stride)
    for ii in indexpart
        @info "Extracting from $(ii[begin]) to $(ii[end])"
        res = pmap(imfs[ii]) do f

            t = @elapsed begin
                img = center_image(load_image(f; polarization=true))
                if regrid
                    rimg = Comrade.regrid(img, g)
                else
                    rimg = img
                end

                if blur > 0.0
                    rimg = Comrade.smooth(rimg, μas2rad(blur)/(2*sqrt(2*log(2))))
                end
                stats = summary_ringparams(rimg; maxiters=fevals, order)
            end
            println("This took $t seconds")

            return stats
        end
        dftmp = DataFrame(res)
        dftmp.file = imfs[ii]
        df = vcat(df, dftmp)
        CSV.write(outname, df)
    end
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end