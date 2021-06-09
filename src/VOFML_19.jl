module VOFML_19

using DelimitedFiles

const coefs = [readdlm("data/" * file, ',') for file in ["W1.csv", "W2.csv", "W3.csv", "W4.csv", "W5.csv"]]
const intercepts = [readdlm("data/" * file, ',') for file in ["b1.csv", "b2.csv", "b3.csv", "b4.csv", "b5.csv"]]

# NEURAL NETWORK
relu(x) = max(0.0, x)

function neural_network(x)
    for i_layer in 1:length(coefs)
        x = transpose(coefs[i_layer]) * x + intercepts[i_layer]
        x = relu.(x)
    end
    return x[1]
end

# PRE-PROCESSING
# Conventions:
# The values of α on a 3×3 stencil are stored in a 3×3 array indexed
# with {-1, 0, 1} using the notation α[x, y] (and NOT α[row, col]).
# Examples:
# α[0, 0] = center of stencil
# α[-1, 0] = west of stencil
# α[1, 1] = north east of stencil
# α[1, -1] = south east of stencil
# etc...
#
# We derive the flux on the right of the center cell, that is between the [0, 0] cell and the [1, 0] cell.
# Rotate the stencil for the fluxes at the other interfaces.
#
# β is the Courant number: u Δt / Δx

flatten(α, β) = [β, α[1, 1], α[0, 1], α[-1, 1], α[1, 0], α[0, 0], α[-1, 0], α[1, -1], α[0, -1], α[-1, -1]]
raw_flux(α, β) = neural_network(flatten(α, β))

# POST-PROCESSING
# Ensure the output is within [0, 1]
fix(α_flux) = max(0.0, min(1.0, α_flux))

# Ensure the positivity of the remaining volume fraction
# (see section 6.2 of Després and Jourdren 2020)
positivity(α_flux, α0, β) = min(α0/β, max(1 - (1 - α0)/β, α_flux))

# Symmetries
# (see section 6.1 of Després and Jourdren 2020)
function symmetric_flux(α, β)
    return (raw_flux(α, β)
            + raw_flux(upsidedown(α), β)
            + 1.0 - raw_flux(invert(upsidedown(α)), β)
            + 1.0 - raw_flux(invert(α), β)
           )/4
end

invert(α) = map(αi -> 1 - αi, α)
upsidedown(α) = reverse(α, dims=2)


# MAIN FUNCTION
flux(α, β) = fix(positivity(symmetric_flux(α, β), α[0, 0], β))

end
