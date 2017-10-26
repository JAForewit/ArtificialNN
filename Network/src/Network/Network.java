package Network;

import parser.*;
import java.util.Arrays;
import java.util.Random;


public class Network {

    public final int INPUT_SIZE;
    public final int OUTPUT_SIZE;
    public final int[] LAYER_SIZES;
    public final int NETWORK_SIZE;
    public final double MIN_BIAS = -0.5;
    public final double MAX_BIAS = 0.7;
    public final double MIN_WEIGHT = -1;
    public final double MAX_WEIGHT = 1;

        private double[][] output;              //output[layer][neuron]
        private double[][] bias;                //bias[layer][neuron]
        private double[][][] weights;           //weights[layer][neuron][prevNeuron]
        private double[][] errors;              //errors[layer][neuron]
        private double[][] outputDerivative;    //outputDerivative[layer][neuron]


    public Network (int... layers) {

        LAYER_SIZES = layers;
        NETWORK_SIZE = LAYER_SIZES.length;
        INPUT_SIZE = LAYER_SIZES[0];
        OUTPUT_SIZE = LAYER_SIZES[NETWORK_SIZE - 1];

        output = new double[NETWORK_SIZE][];
        bias = new double[NETWORK_SIZE][];
        errors = new double[NETWORK_SIZE][];
        outputDerivative = new double[NETWORK_SIZE][];
        weights = new double[NETWORK_SIZE][][];


        output[0] = new double[LAYER_SIZES[0]];

        for (int layer = 1; layer < NETWORK_SIZE; layer++) {

            output[layer] = new double[LAYER_SIZES[layer]];

            bias[layer] = createRandomArray(LAYER_SIZES[layer], MIN_BIAS, MAX_BIAS);

            errors[layer] = new double[LAYER_SIZES[layer]];

            outputDerivative[layer] = new double[LAYER_SIZES[layer]];

            weights[layer] = createRandomArray(LAYER_SIZES[layer],
                    LAYER_SIZES[layer - 1], MIN_WEIGHT, MAX_WEIGHT);
        }
    }

    public double[] calculateOutput(double... input) {
        if (input.length != INPUT_SIZE) return null;
        output[0] = input;

        for (int layer = 1; layer < NETWORK_SIZE; layer++) {
            for (int neuron = 0; neuron < LAYER_SIZES[layer]; neuron++) {

                double sum = bias[layer][neuron];

                for(int prevNeuron = 0; prevNeuron < LAYER_SIZES[layer-1]; prevNeuron++) {
                    sum += output[layer-1][prevNeuron] * weights[layer][neuron][prevNeuron];
                }

                output[layer][neuron] = sigmoid(sum);

                //Sets up the derivative used in the backpropogation algorithm
                outputDerivative[layer][neuron] = output[layer][neuron] * (1 - output[layer][neuron]);
            }
        }

        return output[NETWORK_SIZE-1];
    }

    public void train (double[] input, double[] target, double rate, int iterations) {
        if (target.length != OUTPUT_SIZE) return;

        for (int i = 0; i < iterations; i++) {
            calculateOutput(input);
            backpropError(target);
            updateWeights(rate);
        }
    }

    public void trainBatch(double[][] inputs, double[][] targets, double rate, int iterations) {
        if (inputs.length != targets.length) return;
        for (int i = 0; i < inputs.length; i++) {
            if (inputs[i].length != INPUT_SIZE || targets[i].length != OUTPUT_SIZE) return;
        }

        for (int j = 0; j < iterations; j++) {
            for (int i = 0; i < inputs.length; i++) {
                calculateOutput(inputs[i]);
                backpropError(targets[i]);
                updateWeights(rate);
            }
        }
    }

    public double[] getLastOutput () {
        return output[NETWORK_SIZE-1];
    }

    //********** Relies on the parser package ********
    public void saveNetwork(String fileName) throws Exception {
        Parser p = new Parser();
        p.create(fileName);
        Node root = p.getContent();
        Node netw = new Node("Network");
        Node ly = new Node("Layers");
        netw.addAttribute(new Attribute("sizes", Arrays.toString(this.LAYER_SIZES)));
        netw.addChild(ly);
        root.addChild(netw);
        for (int layer = 1; layer < this.NETWORK_SIZE; layer++) {

            Node c = new Node("" + layer);
            ly.addChild(c);
            Node w = new Node("weights");
            Node b = new Node("biases");
            c.addChild(w);
            c.addChild(b);

            b.addAttribute("values", Arrays.toString(this.bias[layer]));

            for (int we = 0; we < this.weights[layer].length; we++) {

                w.addAttribute("" + we, Arrays.toString(weights[layer][we]));
            }
        }
        p.close();
    }

    //********** Relies on the parser package ********
    public static Network loadNetwork(String fileName) throws Exception {

        Parser p = new Parser();

        p.load(fileName);
        String sizes = p.getValue(new String[] { "Network" }, "sizes");
        int[] si = ParserTools.parseIntArray(sizes);
        Network ne = new Network(si);

        for (int i = 1; i < ne.NETWORK_SIZE; i++) {
            String biases = p.getValue(new String[] { "Network", "Layers", new String(i + ""), "biases" }, "values");
            double[] bias = ParserTools.parseDoubleArray(biases);
            ne.bias[i] = bias;

            for(int n = 0; n < ne.LAYER_SIZES[i]; n++){

                String current = p.getValue(new String[] { "Network", "Layers", new String(i + ""), "weights" }, ""+n);
                double[] val = ParserTools.parseDoubleArray(current);

                ne.weights[i][n] = val;
            }
        }
        p.close();
        return ne;
    }

    //Returns the Mean Squared Error between the target and output given a set of inputs
    public double MSE (double[] input, double[] target) {
        if (input.length != INPUT_SIZE || target.length != OUTPUT_SIZE) return 0;

        calculateOutput(input);

        double sum = 0;
        for (int i = 0; i < target.length; i++) {
            sum += (target[i] - output[NETWORK_SIZE - 1][i]) * (target[i] - output[NETWORK_SIZE - 1][i]);
        }
        return sum / (2d * target.length);
    }


        private void backpropError (double[] target) {

            //Calculate error for output layer
            for (int neuron = 0; neuron < LAYER_SIZES[NETWORK_SIZE - 1]; neuron++) {
                errors[NETWORK_SIZE-1][neuron] = (output[NETWORK_SIZE-1][neuron] - target[neuron])
                        * outputDerivative[NETWORK_SIZE - 1][neuron];
            }

            //Calculate error for hidden neurons
            for (int layer = NETWORK_SIZE - 2; layer > 0; layer --) {
                for (int neuron = 0; neuron < LAYER_SIZES[layer]; neuron++) {
                    double sum = 0;

                    for (int nextNeuron = 0; nextNeuron < LAYER_SIZES[layer + 1]; nextNeuron++) {
                        sum += weights[layer + 1][nextNeuron][neuron] * errors[layer + 1][nextNeuron];
                    }

                    errors[layer][neuron] = sum * outputDerivative[layer][neuron];
                }
            }
        }

        private void updateWeights (double rate) {
            for (int layer = 1; layer < NETWORK_SIZE; layer++) {
                for (int neuron = 0; neuron < LAYER_SIZES[layer]; neuron++) {

                    //Adjusting bias
                    double delta = -rate * errors[layer][neuron];
                    bias[layer][neuron] += delta;

                    //Adjusting weights
                    for (int prevNeuron = 0; prevNeuron < LAYER_SIZES[layer - 1]; prevNeuron++) {
                        weights[layer][neuron][prevNeuron] += delta * output[layer -1][prevNeuron];
                    }
                }
            }
        }

        private double sigmoid(double x) {
            return 1d / (1 + Math.exp(-x));
        }

        private static double[] createRandomArray (int size, double min, double max) {
            Random rand = new Random();

            double[] array = new double[size];
            for (int i = 0; i < size; i++) {
                array[i] = (Math.random() * (max - min)) + min;
            }

            return array;
        }

        private static double[][] createRandomArray (int size1, int size2, double min, double max) {
            Random rand = new Random();

            double[][] array = new double[size1][size2];
            for (int i = 0; i < size1; i++) {
                for (int j = 0; j < size2; j++) {
                    array[i][j] = (Math.random() * (max - min)) + min;

                }
            }

            return array;
        }

}


