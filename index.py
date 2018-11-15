# librerias estandar que maneja subprocesos y archivos de csv esto ya viene con la instalacion de python 
from subprocess import Popen, PIPE
import csv

# Esta instruccion abre el archivo aleatorios.csv 
# el operador de contexto with se usa en la instruccion open para que al terminar el bloque python se encarga de cerrar el archivo y liberar la memoria
with open('aleatorios.csv', newline='') as csvfile:
    # a poblacion se le asigna con la libreria csv una estructura de datos que se llama diccionario funciona de manera similar a un json 
    # las claves que tomara corresponden a la primer columna y el delimitador que va a tomar es la coma 
    poblacion = csv.DictReader(csvfile, delimiter=',')
    # al igual que con el primer archivo abrimos el archivo poblacion.csv en modo de escritura
    with open('poblacion.csv', 'w', newline='') as archivoPoblacion:
        # instanciamos un objeto writer delimitado por comas 
        poblacionWriter = csv.writer(archivoPoblacion, delimiter=',')
        # escribimos las cabeceras en la primer linea del archivo
        poblacionWriter.writerow(["capas","neuronas","epocas","momentum","learning_rate", "instancias_correctamente_clasificadas"])
        index = 0
        # con un ciclo for recorremos cada fila del archivo donde se encuentran los numeros aleatorios
        for row in poblacion:            
            # debido a que el proceso tarda mucho simplemente ponemos un mensaje en pantalla para ver cuantos registros llevamos procesados
            print("{0} registros procesados".format(index))
            # debido a que el momentum y el learning rate vienen en numeros enteros hay que convertirlos a decimal como estamos leyendo un archivo de texto inicialmente es un strin por lo cual hay que parsear el dato a un entero y dividirlo entre 100
            learning_rate =  float(row["learning_rate"]) / 100
            momentum = float(row["momentum"]) / 100
            # como weka recibe las capas ocultas separadas por comas hay que concatenar una coma por cada capa indicando el numero de neuronas
            # python te permite hacer ciclos for en una sola linea llamados list comprehension y depositando el resultado en una lista una ves que se devolvio la lista
            # le decimos a python que los concatene con una coma             
            hidden_layers_string = ",".join([ row["neuronas"] for index in range(0, int(row["capas"]) ) ])
            # a continuacion al comando de java que ejecuta la red neuronal usamos algo que se llama template string 
            # y con la funcion format le indicamos que variable tiene que concatenar en ese lugar indicado por numeros 
            # 1 learning rate 2 momentum 3 epocas 4 capas ocultas y neuronas separadas por comas
            # despues le hacemos un split para que separado por espacios nos cree un arreglo el cual nos va a funcionar para llamar al subproceso que va a ejecutar eso en el sistema operativo y nos va a dar el resultado 
            multilayerPerceptron = "java -cp weka.jar weka.classifiers.functions.MultilayerPerceptron -L {0} -M {1} -N {2} -V 0 -S 0 -E 20 -H {3} -t physhing.arff".format(learning_rate, momentum, row["epocas"],hidden_layers_string).split(" ")
            # Popen o ( Program Open ) como lo sugiere el nombre abre un programa externo y le pasamos por parametros el comando de java 
            # y le indicamos como segundo parametro que debe de abrir una tuberia para la salida estandar 
            # el concepto de tuberia no es de python es de sistemas operativos 
            result = Popen(multilayerPerceptron, stdout= PIPE)
            # a la variable output obtenemos la salida estandar y la variable read nos la trae como un stream de bytes 
            # para poder manipular la salida como un string la dedcodificamos en formato UTF-8 
            # la salida que nos va a dar es el archivo que sale en la GUI de weka 
            output = result.stdout.read().decode("utf-8")
            # como ya tenemos el resultado de weka ahora lo que queremos es buscar el porcentaje de instancias clasificadas 
            # asi que hacemos un ciclo for separando la cadena de weka por saltos de linea y le pedimos que si encuentra la cadena "Correctly Classified Instances"
            # nos devuelva el resultado, actualmente nos devuelve 2 resultados el primero esta relacionado al porcentaje de error y el segundo a la validacion cuzada
            # que es el que nos interesa por eso le decimos que queremos el resultado del indice 1 y con split le decimos que lo separe por estacios en blanco para poder obtener el numero             
            wekaResult = [ item for item in output.split("\n") if "Correctly Classified Instances" in item ][1].split()            
            # por ultimo le decimos al objeto que va a escribir el resultado en el archivo que lo escriba de la siguiente manera 
            # capas, neuronas, epocas, momentum, learning_rate y porcentaje de clasificacion como se daran cuenta tiene un indice -2 
            # esto es por que python te deja leer los arreglos de derecha a izquierda colocando indices negativos este corresponde al numerito que tiene el porcentaje de clasificacion 
            poblacionWriter.writerow([row["capas"], row["neuronas"], row["epocas"], row["momentum"], row["learning_rate"], wekaResult[-2]])
            # por ultimo incrementamos el contador de las instancias que llevamos procesadas 
            index+=1
