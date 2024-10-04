using Test
using NNParamsPrinter
using Lux
using StableRNGs

@testset "printWeightsBiases tests" begin
    rbf(x) = exp(-x^2)
    # Define the network U
    U = Chain(
        # Convolutional Block 1
        Conv((3, 3), 1 => 8, rbf),
        BatchNorm(8),
        MaxPool((2, 2)),

        # Convolutional Block 2
        Conv((3, 3), 8 => 16, rbf),
        BatchNorm(16),
        Dropout(0.25),

        # Convolutional Block 3
        Conv((3, 3), 16 => 16, rbf),
        BatchNorm(16),

        # Convolutional Block 4
        Conv((1, 1), 16 => 1),
        FlattenLayer(),

        # Fully Connected Layers
        Dense(256 => 64, rbf),
        Dropout(0.5),
        Dense(64 => 32, rbf),

        # LSTM and RNN Blocks
        LSTMCell(32 => 16),
        RNNCell(32 => 16),
        GRUCell(16 => 3),
        Dense(3 => 1)  # Adjusted input size to match GRUCell output
    )

    # Initialize the random number generator
    rng = rng = StableRNG(1234)

    # Define a mock input that matches the expected input siz
    # Set up the network parameters and stte
    ps, st = Lux.setup(rng, U)

    # Test the print_weights_and_biases function
    @test printWeightsBiases(U,ps) === nothing
end
