Este software está diseñado para tratar de predecir el comportamiento de Stocks empleando una red neuronal y calcular el beneficio esperado del modelo. Se trata de un estudio preliminar que emplea redes neuronales clásicas. La implementación de otros modelos más apropiados (como LSTM) se ha considerado, sin embargo, se ha preferido desarrollar un frontend para que el proyecto sea más visual en lugar de tratar de mejorar el funcionamiento del mismo. 

## Modo de empleo

1. Correr server.py para encender el servidor
2. Ir a localhost:500 que es donde se hostea (no lo hemos subido a la web).
3. Seleccionar el Ticket que se quiere ver
4. Según si ya está guardado o no, el software bien muestra los resultados o los calcula en el momento. Para poder visualizar ambos comportamientos de forma sencilla los datos del IBEX, SP500 y Meta ya se encuentran calculados mientras que el resto se hacen en el momento.

## Componentes del Proyecto

### 1. Interfaz Web (website.html)
- Interfaz limpia y receptiva para la selección de acciones
- Actualizaciones de progreso en tiempo real
- Carga dinámica de gráficos de predicción

### 2. Servidor (server.py)
- Servidor web Flask para gestionar peticiones
- Sistema de seguimiento de progreso
- Procesamiento asíncrono de predicciones
- Sirve gráficos generados y maneja peticiones concurrentes

### 3. Motor de Predicción (predictions_yf.py)
- Descarga datos históricos de Yahoo Finance (2010-2024)
- Implementa una Red Neuronal con:
  - Entrada: 10 días de datos OHLC (Apertura, Máximo, Mínimo, Cierre)
  - Arquitectura: 40->64->32->4 neuronas con activación ReLU
  - Entrenamiento: optimizador ADAM, pérdida MSE, 100 épocas
  - División: 2010-2020 entrenamiento, 2020-2024 validación

### 4. Visualizaciones Generadas
- Gráficos comparativos de entrenamiento/validación
- Predicciones de precios con señales de compra/venta
- Análisis de retornos acumulados
- Todos los gráficos se guardan automáticamente en el directorio 'plots'

## Technical Details

### Neural Network Architecture
- Input Layer: 40 neurons (10 days × 4 features)
- Hidden Layer 1: 64 neurons with ReLU
- Hidden Layer 2: 32 neurons with ReLU
- Output Layer: 4 neurons (next day's OHLC)

### Data Processing
- Real-time data fetching from Yahoo Finance
- Automatic data normalization
- Sequence-based dataset creation
- Sliding window approach for predictions

### Trading Signals
- Automatically identifies local minima (buy) and maxima (sell)
- Calculates potential returns based on signal strategy
- Visualizes entry and exit points on price charts

## Performance Metrics
- MSE (Mean Squared Error) for prediction accuracy
- Cumulative returns calculation
- Real-time training progress monitoring