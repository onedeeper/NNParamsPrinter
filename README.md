# NNParamsPrinter.jl

`NNParamsPrinter.jl` is a simple package for printing neural network parameters in a readable format.

## Installation

```julia
using Pkg
Pkg.add("NNParamsPrinter")
```

## Usage

```julia
using Lux
using NNParamsPrinter
using StableRNGs

U = Chain(
    Dense(1, 3, rbf),
    Dense(3, 3, rbf),
    Dense(3, 3, rbf),
    Dense(3, 1)
)

nn_init_params, snn = Lux.setup(rng, U);
print_weights_and_biases(nn_init_params)
```