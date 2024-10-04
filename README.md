# NNParamsPrinter.jl

`NNParamsPrinter.jl` is a simple package for printing neural network parameters in a readable format.
Currently only supported for **Lux**. 

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
printWeightsBiases(U, nn_init_params, print_values = true)

Layer 1 :  Dense(1 => 3, rbf) : 
        weights (shape: (3, 1)):
                Float32[0.70705664; -1.2754807; 1.0824884;;]
        bias (shape: (3,)):
                Float32[0.18551421, 0.4326675, 0.024800062]
Layer 2 :  Dense(3 => 3, rbf) : 
        weights (shape: (3, 3)):
                Float32[-0.7039337 0.65329766 0.73901486; -0.8899543 0.31681418 -0.7408178; -0.4000666 0.48276234 -0.5379133]
        bias (shape: (3,)):
                Float32[0.42945153, 0.3800699, -0.18156144]
Layer 3 :  Dense(3 => 3, rbf) : 
        weights (shape: (3, 3)):
                Float32[-0.022448301 -0.13235784 -0.32542706; 0.59363365 -0.1478889 0.9222369; -0.6551368 0.78240037 -0.21426916]
        bias (shape: (3,)):
                Float32[-0.18754774, 0.27221495, 0.07845076]
Layer 4 :  Dense(3 => 1) : 
        weights (shape: (1, 3)):
                Float32[-0.8237293 -0.6526296 0.12768888]
        bias (shape: (1,)):
                Float32[0.1146311]
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
