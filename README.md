# Reconocimiento_facial

            Red Neuronal Artificial para el reconocimiento facial

Entrenar una red neuronal convolucional para que pueda reconocer el
rostro del usuario.

Para el entrenamiento de la red, se usó la técnica Transfer-Learning, que
consiste en entrenar una red de forma general, en este caso para el
reconocimiento facial, se entrenó primero para el reconocimiento de rostros, y
luego, para distinguir entre el usuario y otra persona (un desconocido).

Para la primera red, se usó la base de datos CelebA que cuenta con 202,599
imágenes etiquetadas en 40 atributos diferentes, y luego se entrenó la
segunda sobre esta red pre – entrenada ahora con las fotos del usuario para
poder distinguirlo de entre otras personas.

El problema principal que se encontró es que, por falta de un equipo adecuado,
fue imposible entrenar la primera red con todas sus imágenes y etiquetas, así
que se tuvo que usar una cantidad menor de imágenes, así como de etiquetas.

Hubo 3 versiones diferentes del entrenamiento de la primera red:
  1. Reconocimiento_facial_v1.py
      Es una red de 3 capas convolucionales de 32, 32, y 16 filtros, y dos
      capas planas, una interna de 64 neuronas, y una de salida de 40
      neuronas, la entrada fue de los 202599 datos y salida de 40 etiquetas
      sin datos de prueba.
      Esta fue la que dio peores resultados, no logrando aprender a lo largo
      de 30 épocas, con una precisión menor del 10%, la función de costo no
      pudo ser minimizada, dando resultados cada vez más altos. Las etiquetas
      están en la forma de -1 y 1, por lo que se usó la función de activación
      "tanh" y el optimizador "RMSprop".

  2. Reconocimiento_facial_v2.py
      Es una red de 3 capas convolucionales de 32, 64, y 128 filtros, y dos
      capas densas, una oculta de 64 neuronas, y una de salida con
      20 neuronas.
      Es la que dio mejores resultados, llegando a una eficiencia de 87.92%
      sin sufrir sobreajuste a lo largo de 30 épocas. Para esta versión,
      los datos usados, fueron 50000 imágenes y 20 etiquetas (las más
      importantes para reconocer un rostro), a los datos de los atributos,
      se les cambió el formato para que estuvieran en valores 0 y 1
      (cambiando los -1 originales a 0). Los datos fueron separados en dos
      conjuntos, uno de entrenamiento y otro de pruebas para probar la
      precisión de la red en condiciones reales.  La función de costo fue
      cambiada por "binary_crossentropy", la función de activación de la
      última capa fue "sigmoid" la compilación de la red fue con el
      optimizador "RMSprop".

      La evolucion del entrenamiento se puede observar en el archivo
      "redes_entrenadas/3.6.txt"
      La red entrenada es
      "redes_entrenadas/reconocimiento_facial_v2.8.h5"

  3. Reconocimiento_facial_v3.py
      Es una red de 3 capas convolucionales de 32, 64, y 128 filtros, y dos
      capas densas, una oculta de 64 neuronas, y una de salida con 20
      neuronas.
      Esta versión es una copia de la versión dos, pero aumentando los datos
      a 70000 y usando las 40 etiquetas obteniendo resultados satisfactorios
      del 80% de precisión, pero con resultados de entrenamiento extraños,
      lo que se observó en las gráficas es que la red no lograba
      aprender, minimizando la función de costo en el entrenamiento, pero
      incrementándola en las validaciones.

La segunda red.
  1. Reconocimiento_facial_Usuario.py
      Es una red que carga el modelo pre - entrenado con capas convolucionales
      para reconocer rostros, y aumentando dos capas planas, densas, ocultas
      de 128 y 64 neuronas cada una, y una de salida de una neurona.
      Los datos de alimentación fueron tomados de CelebA, y de las fotos
      proporcionadas por el usuario.

      Esta red fue entrenada por 20 épocas, obteniendo unos resultados mucho
      mejor de lo esperado, obteniendo un 100% de precisión sin que fuera
      notorio un sobre ajuste.

      La red entrenada es
      "redes_entrenadas/reconocimiento_facial_Usuario_v4.h5"

La técnica Transfer-Learning es una muy poderosa de aprendizaje, en
este caso, el entrenamiento del reconocimiento del usuario entre otras personas,
fue increíblemente rápido, y preciso, mucho más que si se hubiera entrenado la
red desde un inicio con las fotos del usuario y las de otras personas, en otras
palabras, es mejor entrenar una red que pueda reconocer ciertas características
y luego entrenarla para que reconozca características específicas. El mayor
problema del uso de esta técnica es el tiempo de entrenamiento de la primer
red, y que, para tener mejores resultados, es necesario tener un equipo adecuado
que soporte manejar una cantidad muy grande de datos, así como de atributos.
