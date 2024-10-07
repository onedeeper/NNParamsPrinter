module NNParamsPrinter

# Export the function to make it accessible when the package is imported
export printWeightsBiases
"""
    printWeightsBiases(net, nn_params; print_values = false)

Prints out all the weights and biases of a neural network.

# Arguments:
- `net`: The neural network model to get the abstract layer names
- `nn_params`: parameters of the neural network
- `print_values`: If `true`, print the values of the weights and biases. Default is `false`, just print the shapes.
"""
function printWeightsBiases(net, nn_params; print_values = false)
    labelled_layers = net.layers
    for (layer, layer_params) in enumerate(nn_params)
        layer_type = labelled_layers[layer]
        try
            weights = get(layer_params, :weight,nothing)
            bias = get(layer_params, :bias,nothing)
            if print_values
                println("Layer $layer :  $(layer_type) : \n\tweights (shape: $(size(weights))):\n\t\t$(weights)\n\tbias (shape: $(size(bias))):\n\t\t$(bias)")
            else
                println("Layer $layer : $(layer_type) : \n\tweights (shape: $(size(weights))):\n\tbias (shape: $(size(bias))):")
            end
        catch
            layer_name = string(typeof(layer_type))
            result = match(r"^[^{]*", layer_name)
            println("Layer $layer : $(layer_type)")
            if result.match  == "LSTMCell" || result.match == "GRUCell" || result.match == "RNNCell"
                for cell in keys(layer_params)
                    if print_values
                        println("\t$cell (shape: $(size(layer_params[cell]))):\n\t\t$(layer_params[cell])")
                    else
                        println("\t$cell (shape: $(size(layer_params[cell]))):")
                    end
                end
            end
        end
    end
end

end
