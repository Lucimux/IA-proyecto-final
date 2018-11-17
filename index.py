# librerias estandar para manejar archivos csv, subprocesos, tiempo de procesador, y muliprocesamiento 
from subprocess import Popen, PIPE
from multiprocessing import Queue
from time import time
import multiprocessing
import os
import csv
# cola de procesos
queue = Queue()
# cola resultados de la clasificacion
finalResult = Queue()

# funcion para procesar cada subconjunto de datos
def process_chunk(chunk):
    # obtener un pid para cada proceso y encolarlo
    queue.put(os.getpid())        
    for row in chunk:
        print("Procesando {0}".format(row))
        """ 
            debido a que el momentum y el learning rate vienen en numeros enteros hay que convertirlos a decimal como estamos 
            leyendo un archivo de texto inicialmente es un strin por lo cual hay que parsear el dato a un entero 
            y dividirlo entre 100   
        """
        learning_rate =  float(row["learning_rate"]) / 100
        momentum = float(row["momentum"]) / 100
        """ 
            como weka recibe las capas ocultas separadas por comas hay que concatenar una coma por cada capa indicando el numero de neuronas
            python te permite hacer ciclos for en una sola linea llamados list comprehension y depositando el resultado en una 
            lista una ves que se devolvio la lista le decimos a python que los concatene con una coma             
        """            
        """ 
            a continuacion al comando de java que ejecuta la red neuronal usamos algo que se llama template string 
            y con la funcion format le indicamos que variable tiene que concatenar en ese lugar indicado por numeros 
            1 learning rate 2 momentum 3 epocas 4 capas ocultas y neuronas separadas por comas
            despues le hacemos un split para que separado por espacios nos cree un arreglo el cual nos va a funcionar 
            para llamar al subproceso que va a ejecutar eso en el sistema operativo y nos va a dar el resultado 
        """
        hidden_layers_string = ",".join([ row["neuronas"] for index in range(0, int(row["capas"]))])
        multilayerPerceptron = "java -cp weka.jar weka.classifiers.functions.MultilayerPerceptron -L {0} -M {1} -N {2} -V 0 -S 0 -E 20 -H {3} -t physhing.arff".format(learning_rate, momentum, row["epocas"],hidden_layers_string).split(" ")
        """ 
            Popen o ( Program Open ) como lo sugiere el nombre abre un programa externo y le pasamos por parametros el 
            comando de java y le indicamos como segundo parametro que debe de abrir una tuberia para la salida estandar 
            el concepto de tuberia no es de python es de sistemas operativos 
        """             
        result = Popen(multilayerPerceptron,stdout= PIPE)
        """ 
            a la variable output obtenemos la salida estandar y la variable read nos la trae como un stream de bytes 
            para poder manipular la salida como un string la dedcodificamos en formato UTF-8 
            la salida que nos va a dar es el archivo que sale en la GUI de weka 
        """
        output = result.stdout.read().decode("utf-8")
        """
            como ya tenemos el resultado de weka ahora lo que queremos es buscar el porcentaje de instancias clasificadas 
            asi que hacemos un ciclo for separando la cadena de weka por saltos de linea y le pedimos que si encuentra la cadena "Correctly Classified Instances"
            nos devuelva el resultado, actualmente nos devuelve 2 resultados el primero esta relacionado al porcentaje de error y el segundo a la validacion cuzada
            que es el que nos interesa por eso le decimos que queremos el resultado del indice 1 y con split le decimos que lo separe por estacios en blanco para poder obtener el numero             
        """
        wekaResult = [ item for item in output.split("\n") if "Correctly Classified Instances" in item ][1].split()  
        print("Procesado por weka")          
        # metemos en una cola el resultado obtenido        
        finalResult.put([row["capas"], row["neuronas"], row["epocas"], row["momentum"], row["learning_rate"], wekaResult[-2]])     
    

if __name__ == "__main__":
    # medimos el tiempo en segundos cuando empezo a ejecutarse el programa
    start_time = time()
    with open('aleatorios.csv') as csvfile:        
        # abrimos el archivo con los numeros aleatorios delimitado por comas en una estrucura de diccionario de python
        reader = csv.DictReader(csvfile, delimiter=',')
        # convertimos el resultado en una lista 
        sample = list(reader)
        # obtenemos el tamano de cada bloque a procesar
        chunks = round(len(sample) / 4)  
        # en un list comprehension hacemos 4 subconjuntos de datos 
        sampleChunks = [ sample[i:i+chunks] for i in range(0, len(sample), chunks) ]
        # creamos una lista vacia donde depositaremos los procesos 
        processes = []
        # iteramos sobre los 4 subconjuntos de datos 
        for chunk in sampleChunks:
            # creamos un proceso por cada subconjunto de datos
            thread = multiprocessing.Process(target=process_chunk, args=[chunk])
            # lo anadimos a la lista
            processes.append(thread)
            # comenzamos la ejecucion del hilo
            thread.start()
        # sincronizamos los 4 procesos
        for one_process in processes:
            one_process.join()
    # volvemos a medir el tiempo y le restamos el tiempo inicial para medir el tiempo total de la ejecucion
    elapsed_time = (time() - start_time) / 60
    print("Done!")
    print("Elapsed time: %0.10f minutes." % elapsed_time)
    # mandamos a escribir en un nuevo archivo 
    with open("resultado.csv", "w") as writeFile:
        resultWriter = csv.writer(writeFile, delimiter=',')
        # escribimos las cabeceras en la primer linea del archivo
        resultWriter.writerow(["capas","neuronas","epocas","momentum","learning_rate", "instancias_correctamente_clasificadas"])
        # escribimos los resultados de la calsificacion en una nueva linea        
        while not finalResult.empty():
            resultWriter.writerow(finalResult.get())