from subprocess import Popen, PIPE
import matplotlib.pyplot as plt
import csv 

class geneticAlgorithm():
    
    def __init__(self):
        pass

    def fitness(self, wekaConfig):
        print("Procesando {0}".format(wekaConfig))
        learning_rate =  float(wekaConfig[4]) / 100
        momentum = float(wekaConfig[3]) / 100
        hidden_layers_string = ",".join([ str(wekaConfig[0]) for index in range(0, wekaConfig[1])])
        multilayerPerceptron = "java -cp weka.jar weka.classifiers.functions.MultilayerPerceptron -L {0} -M {1} -N {2} -V 0 -S 0 -E 20 -H {3} -t physhing.arff".format(learning_rate, momentum, wekaConfig[2],hidden_layers_string).split(" ")
        result = Popen(multilayerPerceptron,stdout= PIPE)
        output = result.stdout.read().decode("utf-8")
        wekaResult = [ item for item in output.split("\n") if "Correctly Classified Instances" in item ][1].split()  
        print("Procesado por weka")          
        return wekaResult[-2]
    

    def sortData(self): 
        with open("resultado.csv", 'r', newline='') as f_input:
            csv_input = csv.reader(f_input)
            next(csv_input)
            sortedData = sorted(csv_input, key=lambda row: (row[-1]), reverse=True)
        return sortedData
        
    def showGraph(self, data):
        x = []
        y = []
        for index in range(0, len(data)):
            y.append(index)
            x.append(data[index][-1])
    
        plt.plot(x,y, label='Resultados')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Porcentaje de instancias correctamente clasificadas')
        plt.legend()
        plt.show()

    def convertToBinaryRepresentation(self):
        data = self.sortData()
        binaryRepresentation = []
        for row in data[:50]:
            layers_to_bin = '{0:02b}'.format(int(row[0]))
            neurons_to_bin = '{0:04b}'.format(int(row[1]))
            epochs_to_bin = '{0:012b}'.format(int(row[2]))
            momentum_to_bin = '{0:06b}'.format(int(row[3]))
            learning_date_to_bin = '{0:06b}'.format(int(row[4]))
            binaryRepresentation.append([layers_to_bin, neurons_to_bin, epochs_to_bin, momentum_to_bin, learning_date_to_bin])        
        return binaryRepresentation    
    
    def generateDecentens(self):
        binaryData = self.convertToBinaryRepresentation()
        dataPairs = [ binaryData[i:i+2] for i in range(0, 50, 2) ]
        decendents = [ self.crossGenes(row[0], row[1]) for row in dataPairs ]    
        result = []
        for binaryString in decendents:        
            layers_to_dec = int(binaryString[0], 2)            
            if layers_to_dec < 1:
                layers_to_dec = 1
            neurons_to_dec = int(binaryString[1], 2)
            epochs_to_dec = int(binaryString[2], 2)            
            if epochs_to_dec < 100:
                epochs_to_dec = 100
            momentum_to_dec = int(binaryString[3], 2) 
            learning_to_dec = int(binaryString[4], 2) 
            clasified_instances = self.fitness([layers_to_dec, neurons_to_dec, epochs_to_dec, momentum_to_dec, learning_to_dec])
            result.append([layers_to_dec, neurons_to_dec, epochs_to_dec, momentum_to_dec, learning_to_dec, clasified_instances])
        return result

    def crossGenes(self, x, y):
        return [
            self.stepOperator(x[0], y[0]),
            self.crossOperator(x[1], y[1]),
            self.crossOperator(x[2], y[2]),
            self.stepOperator(x[3], y[3]),
            self.stepOperator(x[4], y[4])
        ]
                        
    def stepOperator(self, x, y):
        half = round(len(x)/2)
        return str(x[:half] + y[half:])
        
    def crossOperator(self, x, y):
        half = round(len(x)/2)
        xHalf = x[:half]
        yHalf = y[half:]    
        newChild = "".join([ str(xHalf[index]) + str(yHalf[index]) for index in range(0, half) ])
        return newChild
                
if __name__ == "__main__":
    genetic = geneticAlgorithm()
    
    print("Resultado de la generacion")
    for row in genetic.generateDecentens():
        print(row)
    
    # genetic.showGraph(genetic.sortData())
    
    
    