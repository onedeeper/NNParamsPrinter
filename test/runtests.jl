
using Test
using NNParamsPrinter

@testset "print_weights_and_biases tests" begin
    # Mocking a small neural network parameter set
    mock_params = (
        (weight = [1.0, 2.0, 3.0], bias = [0.1, 0.2, 0.3]),
        (weight = [4.0, 5.0, 6.0], bias = [0.4, 0.5, 0.6])
    )

    # We can test if the function runs without errors
    @test print_weights_and_biases(mock_params) === nothing
end
