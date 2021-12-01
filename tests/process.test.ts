import { NeuralNet } from '../src/classes/NeuralNet';
import { squaredError } from "../src/util/error/squaredError";
import { sigmoidPrime, sigmoid } from "../src/util/activation/sigmoid";



const netOptions = {
    inputLayerNodes: 2,
    hiddenLayerNodes: 2,
    hiddenLayers: 1,
    outputLayerNodes: 2,
    learningRate: 0.5,
    weights: [
        [
            [0.15, 0.25],
            [0.20, 0.30]
        ],
        [
            [0.40, 0.50],
            [0.45, 0.55]
        ]
    ],
    biases: [
        [0.35, 0.35],
        [0.6, 0.6]
    ],
    activationFunction: sigmoid,
    activationFunctionPrime: sigmoidPrime,
    errorFunction: squaredError
};
test("Processing sample input", () => {
    const net = new NeuralNet(netOptions);
    expect(net.process([0.05, 0.10])).toEqual([0.7513650695523157, 0.7729284653214625]);
});

test("Error Evaluation", () => {
    const net = new NeuralNet(netOptions);
    expect(net.getError([0.05, 0.10], [0.01, 0.99])).toBe(0.2983711087600027);
});

test("Backpropagation", () => {
    const net = new NeuralNet(netOptions);
    net.backpropagate([0.05, 0.10], [0.01, 0.99]);
    expect(net.getError([0.05, 0.10], [0.01, 0.99])).toBe(0.2910277743228536);
});