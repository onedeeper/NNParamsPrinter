# NNParamsPrinter.jl

`NNParamsPrinter.jl` is a simple package for printing Lux neural network parameters in a readable format.

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
printWeightsBiases(U, nn_init_params)
```
### Tested and Supported:
- [x] Conv
- [x] BatchNorm
- [x] MaxPool
- [x] Dropout
- [x] FlattenLayer
- [x] Dense
- [x] LSTMCell
- [x] RNNCell
- [x] GRUCell
- [ ] AdaptiveMaxPool
- [ ] AdaptiveMeanPool
- [ ] AlphaDropout
- [ ] Bilinear
- [ ] BranchLayer
- [ ] Chain
- [ ] ConvTranspose
- [ ] CrossCor
- [ ] Embedding
- [ ] GlobalMaxPool
- [ ] GlobalMeanPool
- [ ] GroupNorm
- [ ] InstanceNorm
- [ ] LayerNorm
- [ ] Maxout
- [ ] MeanPool
- [ ] NoOpLayer
- [ ] PairwiseFusion
- [ ] Parallel
- [ ] Recurrence
- [ ] ReshapeLayer
- [ ] Scale
- [ ] SelectDim
- [ ] SkipConnection
- [ ] StatefulRecurrentCell
- [ ] Upsample
- [ ] VariationalHiddenDropout
- [ ] WeightNorm
- [ ] WrappedFunction
