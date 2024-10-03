module NNParamsPrinter

# Export the function to make it accessible when the package is imported
export print_weights_and_biases

"""
    print_weights_and_biases(nn_params)

    Pretty print the weights and biases of the neural network.
"""
function print_weights_and_biases(nn_params)
    for (layer, i) in enumerate(nn_params)
        println("Layer $layer : \n\tweights (shape: $(size(i.weight))):\n\t\t$(i.weight)\n\tbias (shape: $(size(i.bias))):\n\t\t$(i.bias)")
    end
end

end # module NNParamsPrinter
