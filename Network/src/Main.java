
import java.util.Arrays;
import Network.*;
import TrainSet.*;

public class Main {
    public static void main(String[] args) {

        Network net = new Network(3, 3, 2);

        TrainSet trainSet = new TrainSet(net.INPUT_SIZE,net.OUTPUT_SIZE);
        double[] input1 = {10, -5, 2}, target1 = {1, 0};
        double[] input2 = {3, 6, -4}, target2 = {0, 1};
        trainSet.addData(input1, target1);
        trainSet.addData(input2, target2);

        net.train(trainSet,10000,2,0.3);

        System.out.println("Result");
        System.out.println(Arrays.toString(net.calculateOutput(input1)));
        System.out.println("MSE: " + net.MSE(input2, target2));
        System.out.println("Result");
        System.out.println(Arrays.toString(net.calculateOutput(input1)));
        System.out.println("MSE: " + net.MSE(input2, target2));
    }
}
