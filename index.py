from subprocess import Popen, PIPE
multilayerPerceptron = "java -cp weka.jar weka.classifiers.functions.MultilayerPerceptron -L 0.01 -M 0.2 -N 500 -V 0 -S 0 -E 20 -H 10,10 -t physhing.arff".split(" ")
result = Popen(multilayerPerceptron, stdout= PIPE)
output = result.stdout.read().decode("utf-8")
wekaResult = [ item for item in output.split("\n") if "Correctly Classified Instances" in item ][1].split()
print(wekaResult[-2])