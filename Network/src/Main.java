
import java.util.Arrays;
import Network.*;
import TrainSet.*;

public class Main {
    public static void main(String[] args) {

        Network net = new Network(3, 3, 2);
        TrainSet trainSet = new TrainSet(3,2);

        double[] input = new double[]{10, -5, 2};
        double[] target = new double[]{1, 0};
        double[] input2 = new double[]{2, -1, 4};
        double[] target2 = new double[]{0, 1};

        trainSet.addData(input,target);
        trainSet.addData(input2,target2);

        net.train(trainSet,100000,2,0.3);

        System.out.println("Results: ");
        System.out.println(Arrays.toString(net.calculateOutput(input)));
        System.out.println("Error: " + net.MSE(input, target));
        System.out.println(Arrays.toString(net.calculateOutput(input2)));
        System.out.println("Error: " + net.MSE(input2, target2));
    }
}
