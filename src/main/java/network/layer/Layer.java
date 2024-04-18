package network.layer;

import org.apache.commons.math3.linear.RealMatrix;

public interface Layer {

    RealMatrix propogateBackward(RealMatrix inputVector);

    RealMatrix propogateForward(RealMatrix inputVector);

    void correctWeights(double learnRate);

    void setPrevious(Layer layer);

    int getSize();
}
