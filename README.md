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