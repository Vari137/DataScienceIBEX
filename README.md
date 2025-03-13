# DataScienceIBEX

# TODO

# AÑADE LAS PROMPTS QUE USAS

1. Cotizaciónes en formato OCHL (Open Close High Low)

2. Entrenar modelo predictivo de la siguiente vela en base de los últimos N periodos

3. in files datos_entrenamiento there are the stock values in columns of date, opening, close, max and min value during a day, in datos_validacion there are more for more advanced dates. Create a model that using the last 10 days values (opening close max and min) predicts those same values of day 11, then, plot the models predictions for the data in datos_validacion. You can use Adam, nn.sequential, a Relu activation and MSE as error