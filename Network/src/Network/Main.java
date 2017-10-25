package Network;

import java.util.Arrays;

public class Main {
    public static void main(String[] args) {

        Network net = new Network(3, 3, 2);

        double rate = 0.3;

        double[] input = new double[]{10, -5, 2};
        double[] target = new double[]{1, 0};

        double[] input2 = new double[]{10, 5, 3};
        double[] target2 = new double[]{0, 1};

        net.train(input, target, rate, 1000);
        net.train(input2, target2, rate, 1000);

        System.out.println("Train individually: ");

        System.out.println(Arrays.toString(net.calculateOutput(input)));
        System.out.printf("Error: %.3f \n", net.MSE(input, target));

        System.out.println(Arrays.toString(net.calculateOutput(input2)));
        System.out.printf("Error: %.3f \n", net.MSE(input2, target2));




        Network net2 = new Network(3, 3, 2, 2);

        double[][] inputs = {input, input2};
        double[][] targets = {target, target2};

        net2.trainBatch(inputs,targets,rate,1000000);

        System.out.println("\nTrain as a batch: ");

        System.out.println(Arrays.toString(net2.calculateOutput(inputs[0])));
        System.out.printf("Error: %.3f \n", net2.MSE(inputs[0], targets[0]));

        System.out.println(Arrays.toString(net2.calculateOutput(inputs[1])));
        System.out.printf("Error: %.3f \n", net2.MSE(inputs[1], targets[1]));
    }
}
