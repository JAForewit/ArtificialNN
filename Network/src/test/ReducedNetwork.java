package test;

import java.util.Arrays;

/*
This class represents an approximate 3N reduction in total pointers used
where N is the total number of neurons. The effect is minimal because the
pointers needed is O(N^2).
 */
public class ReducedNetwork {

    public final double MIN_BIAS = -0.5;
    public final double MAX_BIAS = 0.7;
    public final double MIN_WEIGHT = -1;
    public final double MAX_WEIGHT = 1;

    //Indexes to help navigate the network array
    public final int OUTPUT = 0;
    public final int BIAS = 1;
    public final int ERROR = 2;
    public final int DERIV = 3;
    public final int WEIGHTS = 0;
    public final int INFO = 1;

    // network [layer] [neuron] [0:[weights], 1:[info]]
    // info[0] = output
    // info[1] = bias
    // info[2] = error
    // info[3] = outputDerivative
    private double[][][/*2*/][] network;
    private double[] input;

    public ReducedNetwork(int... layers) {
        input = new double[layers[0]];
        network = new double[layers.length - 1][][][];

        for (int layer = 0; layer < layers.length - 1; layer++) {
            network[layer] = new double[layers[layer + 1]][ERROR][];

            for (int neuron = 0; neuron < layers[layer + 1]; neuron++) {
                network[layer][neuron][WEIGHTS]
                        = createRandomArray(layers[layer], MIN_WEIGHT, MAX_WEIGHT);
                network[layer][neuron][INFO] = new double[4];
                network[layer][neuron][INFO][BIAS] = randomValue(MIN_BIAS, MAX_BIAS);
            }
        }


    }

    public void train (double[] input, double[] target, double rate, int iterations) {
        if (input.length != input.length || target.length != network[network.length-1].length) return;

        this.input = input;

        for (int i = 0; i < iterations; i++) {
            feedForward();
            backpropError(target);
            updateWeights(rate);
        }
    }

    public double[] calculate(double[] input) {
        this.input = input;
        feedForward();

        double[] output = new double[network[network.length - 1].length];
        for (int neuron = 0; neuron < network[network.length - 1].length; neuron ++) {
            output[neuron] = network[network.length-1][neuron][INFO][OUTPUT];
        }

        return  output;
    }

    //Stores the output errors for each neuron given a target output
    private void backpropError (double[] target) {

        //Calculate error for output layer
        for (int neuron = 0; neuron < network[network.length - 1].length; neuron++) {
            network[network.length - 1][neuron][INFO][ERROR] = (network[network.length-1][neuron][INFO][OUTPUT] - target[neuron])
                    * network[network.length - 1][neuron][INFO][DERIV];
        }

        //Calculate error for hidden neurons
        for (int layer = network.length - 2; layer >= 0; layer --) {
            for (int neuron = 0; neuron < network[layer].length; neuron++) {
                double sum = 0;

                for (int nextNeuron = 0; nextNeuron < network[layer + 1].length; nextNeuron++) {
                    sum += network[layer + 1][nextNeuron][WEIGHTS][neuron] * network[layer + 1][nextNeuron][INFO][ERROR];
                }

                network[layer][neuron][INFO][ERROR] = sum * network[layer][neuron][INFO][DERIV];
            }
        }
    }

    private void updateWeights (double rate) {
        for (int layer = 0; layer < network.length; layer++) {
            for (int neuron = 0; neuron < network[layer].length; neuron++) {

                //Adjusting bias
                double delta = -rate * network[layer][neuron][INFO][ERROR];
                network[layer][neuron][INFO][BIAS] += delta;

                //Adjusting weights
                if (layer == 0) {
                    for (int prevNeuron = 0; prevNeuron < input.length; prevNeuron++) {
                        network[layer][neuron][WEIGHTS][prevNeuron] += delta * input[prevNeuron];
                    }
                } else {
                    for (int prevNeuron = 0; prevNeuron < network[layer - 1].length; prevNeuron++) {
                        network[layer][neuron][WEIGHTS][prevNeuron] += delta * network[layer - 1][prevNeuron][INFO][OUTPUT];
                    }
                }
            }
        }
    }

    private void feedForward() {
        if (input.length != input.length) return;

        //First Hidden layer
        for (int neuron = 0; neuron < network[0].length; neuron++) {
            double sum = network[0][neuron][INFO][BIAS];

            for (int prevNeuron = 0; prevNeuron < input.length; prevNeuron++) {
                sum += (input[prevNeuron] * network[0][neuron][WEIGHTS][prevNeuron]);
            }

            network[0][neuron][INFO][OUTPUT] = sigmoid(sum); //output
            network[0][neuron][INFO][DERIV] = network[0][neuron][INFO][OUTPUT] * (1 - network[0][neuron][INFO][OUTPUT]);
        }

        //All other layers
        for (int layer = 1; layer < network.length; layer++) {
            for (int neuron = 0; neuron < network[layer].length; neuron++) {
                double sum = network[layer][neuron][INFO][BIAS]; //bias

                for (int prevNeuron = 0; prevNeuron < network[layer-1].length; prevNeuron++) {
                    sum += (network[layer-1][prevNeuron][INFO][OUTPUT] * network[layer][neuron][WEIGHTS][prevNeuron]);
                }

                network[layer][neuron][INFO][OUTPUT] = sigmoid(sum); //output
                network[layer][neuron][INFO][DERIV] = network[layer][neuron][INFO][OUTPUT] * (1 - network[layer][neuron][INFO][OUTPUT]);
            }
        }
    }

    private double sigmoid(double x) {
        return 1d / (1 + Math.exp(-x));
    }

    public void printNet () {

        System.out.println("Input Layer: " + Arrays.toString(this.input));

        for (int layer = 0; layer < this.network.length; layer++) {
            System.out.println("Layer " + layer + ":");

            for (int neuron = 0; neuron < this.network[layer].length; neuron++) {
                System.out.println("  Neuron " + neuron + ":");
                System.out.println("    Weights: "
                        + Arrays.toString(this.network[layer][neuron][WEIGHTS]));
                System.out.println("    Output: "
                        + this.network[layer][neuron][INFO][OUTPUT]);
                System.out.println("    Bias: "
                        + this.network[layer][neuron][INFO][BIAS]);
                System.out.println("    error: "
                        + this.network[layer][neuron][INFO][ERROR]);
                System.out.println("    outputDerivative: "
                        + this.network[layer][neuron][INFO][DERIV]);

            }
        }
    }

    private static double randomValue (double min, double max) {
        return (Math.random() * (max - min)) + min;
    }

    private static double[] createRandomArray(int size, double min, double max) {

        double[] array = new double[size];
        for (int i = 0; i < size; i++) {
            array[i] = (Math.random() * (max - min)) + min;
        }

        return array;
    }

    public static void main(String[] args) {

        ReducedNetwork net = new ReducedNetwork(3, 2, 2);
        double[] input = {0.1,0.2,0.3};
        double[] target = {1, 0};
        double rate = 0.3;


        net.train(input, target,rate,1000);

        System.out.println(Arrays.toString(net.calculate(input)));
    }
}
