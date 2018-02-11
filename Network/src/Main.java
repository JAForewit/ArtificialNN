
import java.util.Arrays;
import Network.*;
import TrainSet.*;

public class Main {
    public static void main(String[] args) {

        Network net = new Network(10, 10, 10);

        TrainSet trainSet = new TrainSet(net.INPUT_SIZE,net.OUTPUT_SIZE);
        double[] input1 = {10,-5,2,5,5,4,4,8,8,-2};
        double[] target1 = {1,0,1,0,1,0,1,0,1,0};
        double[] input2 = {6,-4,5,2,2,8,8,4,4,2};
        double[] target2 = {0,1,0,1,0,1,0,1,0,1};
        trainSet.addData(input1, target1);
        trainSet.addData(input2, target2);

        long time = System.nanoTime();
        net.train(trainSet,10000,2,0.3);
        time = System.nanoTime() - time;

        System.out.println("Result: " + Arrays.toString(net.calculateOutput(input1)));
        System.out.println("MSE: " + net.MSE(input1, target1));
        System.out.println("Result: " + Arrays.toString(net.calculateOutput(input2)));
        System.out.println("MSE: " + net.MSE(input2, target2));
        System.out.println("training time (ms): " + time/1000000);
    }
}
