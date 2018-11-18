import matplotlib.pyplot as plt
import csv 

class geneticAlgorithm():
    
    def __init__(self):
        pass

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
        for row in data:
            layers_to_bin = '{0:02b}'.format(int(row[0]))
            neurons_to_bin = '{0:04b}'.format(int(row[1]))
            epochs_to_bin = '{0:012b}'.format(int(row[2]))
            momentum_to_bin = '{0:06b}'.format(int(row[3]))
            learning_date_to_bin = '{0:06b}'.format(int(row[4]))
            binaryRepresentation.append([layers_to_bin, neurons_to_bin, epochs_to_bin, momentum_to_bin, learning_date_to_bin])        
        return binaryRepresentation    

if __name__ == "__main__":
    genetic = geneticAlgorithm()
    for binary in genetic.convertToBinaryRepresentation():
        print(binary)
    genetic.showGraph(genetic.sortData())
    
    
    