using VOFML_19
using OffsetArrays
using Test

@test size(VOFML_19.coefs[1]) == (10, 80)
@test size(VOFML_19.coefs[2]) == (80, 40)
@test size(VOFML_19.coefs[3]) == (40, 20)
@test size(VOFML_19.coefs[4]) == (20, 10)
@test size(VOFML_19.coefs[5]) == (10, 1)

@test size(VOFML_19.intercepts[1]) == (80, 1)
@test size(VOFML_19.intercepts[2]) == (40, 1)
@test size(VOFML_19.intercepts[3]) == (20, 1)
@test size(VOFML_19.intercepts[4]) == (10, 1)
@test size(VOFML_19.intercepts[5]) == (1, 1)

@test typeof(VOFML_19.neural_network(rand(10))) == Float64

α = OffsetArray(rand(3, 3), -1:1, -1:1)
@test VOFML_19.upsidedown(α)[1, 1] == α[1, -1]

# Centered vertical interface
α = OffsetArray(zeros(3, 3), -1:1, -1:1)
α[-1, -1:1] .= 1.0
α[0, -1:1] .= 0.5
α[1, -1:1] .= 0.0

@test VOFML_19.flux(α, 0.2) ≈ 0.0
@test VOFML_19.flux(α, 0.6) ≈ 0.18936623983604
@test VOFML_19.flux(α, 1.0) ≈ 0.5
@test VOFML_19.flux(VOFML_19.upsidedown(α), 0.6) ≈ 0.1893662398360
@test VOFML_19.flux(VOFML_19.invert(α), 0.6) ≈ 0.8106337601639

# Centered horizontal interface
α = OffsetArray(zeros(3, 3), -1:1, -1:1)
α[-1:1, -1] .= 1.0
α[-1:1, 0] .= 0.5
α[-1:1, 1] .= 0.0

@test VOFML_19.flux(α, 0.2) ≈ 0.5
@test VOFML_19.flux(α, 0.6) ≈ 0.5
@test VOFML_19.flux(α, 1.0) ≈ 0.5
@test VOFML_19.flux(VOFML_19.upsidedown(α), 0.6) ≈ 0.5
@test VOFML_19.flux(VOFML_19.invert(α), 0.6) ≈ 0.5

# Top right corner
α = OffsetArray(zeros(3, 3), -1:1, -1:1)
α[-1, -1:1] .= 0.0
α[-1:1, -1] .= 0.0
α[0, 0] = 0.25
α[1, 0] = 0.5
α[0, 1] = 0.5
α[1, 1] = 1.0

@test VOFML_19.flux(α, 0.2) ≈ 0.4994985817253
@test VOFML_19.flux(α, 0.6) ≈ 0.4010428879650
@test VOFML_19.flux(α, 1.0) ≈ 0.25
