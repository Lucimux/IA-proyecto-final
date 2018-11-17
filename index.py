# librerias estandar que maneja subprocesos y archivos de csv esto ya viene con la instalacion de python 
from subprocess import Popen, PIPE
from multiprocessing import Pool
import csv

def process_line(row):
    print("Procesando {0}".format(row))
    learning_rate =  float(row["learning_rate"]) / 100
    momentum = float(row["momentum"]) / 100
    hidden_layers_string = ",".join([ row["neuronas"] for index in range(0, int(row["capas"]))])
    multilayerPerceptron = "java -cp weka.jar weka.classifiers.functions.MultilayerPerceptron -L {0} -M {1} -N {2} -V 0 -S 0 -E 20 -H {3} -t physhing.arff".format(learning_rate, momentum, row["epocas"],hidden_layers_string).split(" ")
    result = Popen(multilayerPerceptron, stdout= PIPE)
    output = result.stdout.read().decode("utf-8")
    wekaResult = [ item for item in output.split("\n") if "Correctly Classified Instances" in item ][1].split()  
    print("Procesado por weka")          
    return [row["capas"], row["neuronas"], row["epocas"], row["momentum"], row["learning_rate"], wekaResult[-2]]

if __name__ == "__main__":
    pool = Pool(4)
    with open('aleatorios.csv') as csvfile:
        # chunk the work into batches of 4 lines at a time
        poblacion = csv.DictReader(csvfile, delimiter=',')       
        results = pool.map(process_line, poblacion, 4)
    
    with open('poblacion.csv', 'w', newline='') as archivoPoblacion:
        poblacionWriter = csv.writer(archivoPoblacion, delimiter=',')
        poblacionWriter.writerow(["capas","neuronas","epocas","momentum","learning_rate", "instancias_correctamente_clasificadas"])
        for result in results:            
            poblacionWriter.writerow(result)
