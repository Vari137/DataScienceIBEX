# DataScienceIBEX

# Recopilación prompts más relevantes

1. Cambia las cotizaciones a formato OCHL (Open Close High Low)

2. Entrenar modelo predictivo de la siguiente vela en base de los últimos N periodos

3. in files datos_entrenamiento there are the stock values in columns of date, opening, close, max and min value during a day, in datos_validacion there are more for more advanced dates. Create a model that using the last 10 days values (opening close max and min) predicts those same values of day 11, then, plot the models predictions for the data in datos_validacion. You can use Adam, nn.sequential, a Relu activation and MSE as error.


4. Using the data from the model (and just the data from the model) tell me the cumulative returns I would have had if I had bought where  the function has a minimum (the value for n-1>n<n+1) and sells at maximums, that is, you have the model, say I use that little algorithm of buying low and selling high, which returns I would have compared to what really happened?

5. Using a RELU with ADAM two layers MSE neural network usign torch 100 epochs, download data from the ibex 2010-2024 use 2010-2020 as training 2020-2024 as validation, input parameters are  open close high and low from the last 10 days and output should be the expected open close high low the next day, use 40;64 64:32 32:4 as values for the neural network. Plot everything (model training validation comparision)

6. create a website that runs the code in predictions_yf with the ticket you choose (via a menu that lets you choose between ibex, sp500 and the top 10 stocks in the sp500)and shows you the plot in a neat way, make sure it has a good interface for changing and that it saves the files that are already generated. It should also let you provide a custom ticket (the string in the ticket variable), first check if that has been calculated (if it has the plots will already be in this folder in .png files starting with the ticket name)