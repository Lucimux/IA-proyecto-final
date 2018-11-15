from subprocess import Popen, PIPE
import csv

with open('aleatorios.csv', newline='') as csvfile:
    poblacion = csv.DictReader(csvfile, delimiter=',')
    with open('poblacion.csv', 'w', newline='') as archivoPoblacion:
        poblacionWriter = csv.writer(archivoPoblacion, delimiter=',')
        poblacionWriter.writerow(["capas","neuronas","epocas","momentum","learning_rate", "instancias_correctamente_clasificadas"])
        index = 0
        for row in poblacion:            
            print("{0} registros procesados".format(index))
            # 1 learning rate 2 momentum 3 epocas 4 capas ocultas y neuronas separadas por comas
            learning_rate =  float(row["learning_rate"]) / 100
            momentum = float(row["momentum"]) / 100
            hidden_layers_string = ",".join([ row["neuronas"] for index in range(0, int(row["capas"]) ) ])
            multilayerPerceptron = "java -cp weka.jar weka.classifiers.functions.MultilayerPerceptron -L {0} -M {1} -N {2} -V 0 -S 0 -E 20 -H {3} -t physhing.arff".format(learning_rate, momentum, row["epocas"],hidden_layers_string).split(" ")
            result = Popen(multilayerPerceptron, stdout= PIPE)
            output = result.stdout.read().decode("utf-8")
            wekaResult = [ item for item in output.split("\n") if "Correctly Classified Instances" in item ][1].split()            
            poblacionWriter.writerow([row["capas"], row["neuronas"], row["epocas"], row["momentum"], row["learning_rate"], wekaResult[-2]])
            index+=1