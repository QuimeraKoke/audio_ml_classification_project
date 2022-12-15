# Trabajo de investigación 
# Estudio del Rendimiento de RNNs en clasificación de audio


## Introducción

En la actualidad, una de la familia de redes más utilizadas para el procesamiento de secuencia son las redes neuronales recurrentes (RNNs). Una de las particularidades de este tipo de red es que procesa secuencialmente la información de la secuencia, sin embargo, esto puede derivar en la pérdida de información del principio de una secuencia hacia más adelante. Estas redes son ampliamente usadas en tareas de procesamiento de lenguaje natural y también para el procesamiento de audio.

En este proyecto en particular se hace uso de RNNs para la clasificación de audio haciendo uso de características precalculadas del audio original correspondientes a MFCC, delta y delta-delta. El dataset usado corresponde a Speech Commands, este tiene muchos ejemplos de audios de 35 comandos de voz y se busca ocupar las RNNs para predecir la clase correspondiente.

El objetivo principal del proyecto es estudiar el desempeño de distintas arquitecturas de RNNs para la clasificación de audio. En una etapa inicial el objetivo es observar el desempeño obtenido por 3 arquitecturas tradicionales, que serían la RNN Vanilla, Gated Recurrent Unit (GRU) y Long Short Term Memory (LSTM), para obtener el mejor resultado variando la ventana de muestreo. Posteriormente, en una segunda etapa, se busca estudiar el desempeño en la tarea de arquitecturas que consten de hacer variaciones sobre la estructura de las celdas de una implementación personalizada de LSTM entregadas por el tutor del proyecto.


## Metodología

### Primera etapa

En esta etapa nuestro principal objetivo es comparar las tres principales variantes de una red RNN, las cuales son RNN Vanilla, Gated Recurrent Unit (GRU) y Long Short Term Memory en la tarea de clasificación de audio, es por esto que procesamos el audio con la FFT para obtener sus 13 frecuencias principales variando la resolución de la ventana para aplicar la FFT. Obteniendo así 4 datasets procesados con un overlap de un 50%, un sample rate de 16.000 Hz y con una ventana de 20ms, 40ms, 80ms y 120ms respectivamente.

Para revisar está experiencia y los resultados  elaboramos un laboratorio en donde se pueden hacer las pruebas pertinentes en el archivo Resultados_1.ipynb

### Segunda etapa

En esta etapa nuestro principal objetivo es comparar las 5 variaciones de una arquitectura LSTM entregada por el cuerpo docente. Así se implementaron las variantes:
- Peephole LSTM
- Coupled Gate LSTM
- No Forget Gate LSTM
- No Input Gate LSTM
- No Output Gate LSTM

 Para revisar está experiencia y los resultados elaboramos un laboratorio en donde se pueden hacer las pruebas pertinentes en el archivo Resultados_2.ipynb


## Resultados

### Primera etapa

|                    | 20 ms |          |                 | 40 ms |          |              | 80 ms |          |              | 120 ms |          |              |
|--------------------|-------|----------|-----------------|-------|----------|--------------|-------|----------|--------------|--------|----------|--------------|
| Modelo             | Loss  | Accuracy | Tiempo epoch(s) | Loss  | Accuracy | Tiempo epoch | Loss  | Accuracy | Tiempo epoch | Loss   | Accuracy | Tiempo epoch |
| RNN                | 3.035 | 11.76%   | 2.0             | 1.344 | 63.24%   | 1            | 1.078 | 77.57%   | 1            | 1.135  | 77.18%   | 1            |
| GRU                | 0.856 | 81.73%   | 1.0             | 0.93  | 82.10%   | 1            | 0.86  | 82.31%   | 1            | 0.979  | 82.02%   | 1            |
| LSTM               | 0.949 | 76.86%   | 1.00            | 1.076 | 77.04%   | 1            | 1.053 | 78.29%   | 1            | 0.981  | 80.86%   | 1            |
| RNN bidireccional  | 3.048 | 11.68%   | 1.00            | 1.166 | 70.12%   | 1            | 1.046 | 78.20%   | 0            | 1.108  | 78.70%   | 0            |
| GRU bidireccional  | 0.839 | 80.50%   | 1.00            | 0.886 | 79.82%   | 1            | 0.891 | 79.95%   | 0            | 0.913  | 80.96%   | 0            |
| LSTM bidireccional | 1.002 | 73.09%   | 1.00            | 0.996 | 76.66%   | 1            | 0.985 | 77.87%   | 0            | 0.836  | 80.69%   | 0            |


### Segunda etapa

|                        | 40 ms |          |                 |
|------------------------|-------|----------|-----------------|
| Modelo                 | Loss  | Accuracy | Tiempo epoch(s) |
| CustomLSTM             | 0.379 | 92.42%   | 14              |
| CustomPeepholeLSTM     | 0.165 | 97.55%   | 18              |
| CustomCoupledGateLSTM  | 0.266 | 95.31%   | 12              |
| CustomNoForgetGateLSTM | 0.246 | 95.20%   | 12              |
| CustomNoInputGateLSTM  | 1.673 | 51.95%   | 11              |
| CustomNoOutputGateLSTM | 0.338 | 92.93%   | 12              |



