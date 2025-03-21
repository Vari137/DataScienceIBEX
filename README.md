# DataScienceIBEX

# TODO

1. Implementar descarga de datos con yf en lugar de datos de ates

# AÑADE LAS PROMPTS MÁS IMPORTANTES QUE USAS

1. Cotizaciónes en formato OCHL (Open Close High Low)

2. Entrenar modelo predictivo de la siguiente vela en base de los últimos N periodos

3. in files datos_entrenamiento there are the stock values in columns of date, opening, close, max and min value during a day, in datos_validacion there are more for more advanced dates. Create a model that using the last 10 days values (opening close max and min) predicts those same values of day 11, then, plot the models predictions for the data in datos_validacion. You can use Adam, nn.sequential, a Relu activation and MSE as error.


4. awesome, using the data from the model (and just the data from the model) tell me the cumulative returns I would have had if I had bought where  the function has a minimum (the value for n-1>n<n+1) and sells at maximums, that is, you have the model, say I use that little algorithm of buying low and selling high, which returns I would have compared to what really happened?

5. using a RELU with ADAM two layers MSE neural network usign torch 100 epochs, download data from the ibex 2010-2024 use 2010-2020 as training 2020-2024 as validation, input parameters are  open close high and low from the last 10 days and output should be the expected open close high low the next day, use 40;64 64:32 32:4 as values for the neural network. Plot everything (model training validation comparision)
