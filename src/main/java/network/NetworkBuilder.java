package network;

import network.layer.Layer;

import java.util.Deque;
import java.util.LinkedList;

public class NetworkBuilder {

    private Deque<Layer> layers;

    private NetworkBuilder() {
        this.layers = new LinkedList<>();
    }

    public static NetworkBuilder builder() {
        return new NetworkBuilder();
    }

    public NetworkBuilder append(Layer layer) {
        layers.add(layer);
        return this;
    }

    public Network build() {
        return new Perceptron(layers);
    }
}
