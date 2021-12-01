import { sigmoid, sigmoidPrime } from '../util/activation/sigmoid';
import { squaredError } from '../util/error/squaredError';

type Weights = number[][][];
type Biases = number[][];
type Nodes = number[][];
type Gradient = number[][][];
type ActivationFunction = (x: number) => number;
type ActivationFunctionPrime = (x: number) => number;
type ErrorFunction = (outputs: number[], targets: number[]) => number;

type NeuralNetOptions = {
    learningRate?: number;
    inputLayerNodes: number;
    hiddenLayerNodes: number;
    hiddenLayers: number;
    outputLayerNodes: number;
    activationFunction?: ActivationFunction;
    activationFunctionPrime?: ActivationFunctionPrime;
    errorFunction?: ErrorFunction;
    weights?: Weights;
    biases?: Biases;
}

export class NeuralNet {
    private inputLayerNodes: number;
    private hiddenLayerNodes: number;
    private hiddenLayers: number;
    private outputLayerNodes: number;

    private weights: Weights;
    private biases: Biases;
    private activationFunction: ActivationFunction;
    private activationFunctionPrime: ActivationFunctionPrime;
    private errorFunction: ErrorFunction;

    private learningRate: number;

    constructor(options: NeuralNetOptions) {

        //assign properties
        this.inputLayerNodes = options.inputLayerNodes;
        this.hiddenLayerNodes = options.hiddenLayerNodes;
        this.hiddenLayers = options.hiddenLayers;
        this.outputLayerNodes = options.outputLayerNodes;
        this.activationFunction = options.activationFunction || sigmoid;
        this.activationFunctionPrime = options.activationFunctionPrime || sigmoidPrime;
        this.errorFunction = options.errorFunction || squaredError;
        this.learningRate = options.learningRate || 0.1;

        //generate weights and biases
        this.weights = options.weights || [
            this.generateRandomWeights(this.inputLayerNodes, this.hiddenLayerNodes),
            ...new Array(this.hiddenLayers - 1).fill(0).map(() => this.generateRandomWeights(this.hiddenLayerNodes, this.hiddenLayerNodes)),
            this.generateRandomWeights(this.hiddenLayerNodes, this.outputLayerNodes)
        ];

        this.biases = options.biases || [
            ...new Array(this.hiddenLayers).fill(0).map(() => this.generateRandomBiases(this.hiddenLayerNodes)),
            this.generateRandomBiases(this.outputLayerNodes),
        ];
    }

    //generates a array of random weights for a set input size and output size
    private generateRandomWeights(inputSize: number, outputSize: number): number[][] {
        return new Array(inputSize).fill(0).map(() => new Array(outputSize).fill(0).map(() => Math.random()));
    }

    //generates a array of empty weights for a set input size and output size
    private generateEmptyWeights(inputSize: number, outputSize: number): number[][] {
        return new Array(inputSize).fill(0).map(() => new Array(outputSize).fill(0));
    }

    //generates an array of random biases for a set number of nodes
    private generateRandomBiases(nodes: number): number[] {
        return new Array(nodes).fill(0).map(() => Math.random());
    }

    //processes one neuron
    private processNeuron(weights: number[], inputs: number[], bias: number): number {
        return this.activationFunction(weights.map((weight, index) => weight * inputs[index]).reduce((a, b) => a + b) + bias);
    }

    //forward pass
    private forwardPass(inputs: number[]): Nodes {

        if (inputs.length !== this.inputLayerNodes) {
            throw new Error(`Inputs do not match input layer nodes. ${inputs.length} !== ${this.inputLayerNodes}`);
        }

        //initialize nodes
        const nodes: Nodes = [
            new Array(this.inputLayerNodes).fill(0),
            ...new Array(this.hiddenLayers).fill(0).map(() => new Array(this.hiddenLayerNodes).fill(0)),
            new Array(this.outputLayerNodes).fill(0)
        ];

        //set input nodes
        nodes[0] = inputs;

        //process hidden layers
        for (let i = 1; i <= this.hiddenLayers + 1; i++) {

            //get the weights for this pair of layers
            const weights = this.weights[i - 1];

            //get the biases for this layer
            const biases = this.biases[i - 1];

            //process each neuron
            for (let j = 0; j < nodes[i].length; j++) {
                nodes[i][j] = this.processNeuron(weights.map(n => n[j]), nodes[i - 1], biases[j]);
            }
        }

        return nodes;
    }

    //processes based on inputs
    process(inputs: number[]): number[] {

        const nodes = this.forwardPass(inputs);

        //return output layer
        return nodes[nodes.length - 1];
    }

    //gets the error
    getError(inputs: number[], targetOutput: number[]): number {
        return this.errorFunction(this.process(inputs), targetOutput);
    }

    //calculate gradient
    getGradient(inputs: number[], targetOutput: number[]): Gradient {

        //forward pass
        const nodes = this.forwardPass(inputs);

        //get the output of the net
        const netOutput = nodes[nodes.length - 1];

        //setup gradients
        const gradient: number[][][] = [
            this.generateEmptyWeights(this.inputLayerNodes, this.hiddenLayerNodes),
            ...new Array(this.hiddenLayers - 1).fill(0).map(() => this.generateEmptyWeights(this.hiddenLayerNodes, this.hiddenLayerNodes)),
            this.generateEmptyWeights(this.hiddenLayerNodes, this.outputLayerNodes)
        ];

        //setup partial product nodes
        const partialNodes: Nodes = [
            new Array(this.inputLayerNodes).fill(0),
            ...new Array(this.hiddenLayers).fill(0).map(() => new Array(this.hiddenLayerNodes).fill(0)),
            new Array(this.outputLayerNodes).fill(0)
        ];

        //get the gradient for the last layer of weights

        //get partial product of the last layer
        partialNodes[partialNodes.length - 1] = netOutput.map((output, index) => (output - targetOutput[index]) * this.activationFunctionPrime(output));

        //get the gradient for the last layer of weights
        for (let i = 0; i < this.hiddenLayerNodes; i++) {
            for (let j = 0; j < this.outputLayerNodes; j++) {
                gradient[gradient.length - 1][i][j] = partialNodes[partialNodes.length - 1][j] * nodes[nodes.length - 2][i];
            }
        }

        //get the gradient for the hidden layers
        for (let i = this.hiddenLayers; i > 1; i--) {
            for (let j = 0; j < this.hiddenLayerNodes; j++) {

                //get partial product
                partialNodes[i][j] = this.weights[i][j].map((weight, index) => weight * partialNodes[i + 1][index]).reduce((a, b) => a + b);

                for (let k = 0; k < this.hiddenLayerNodes; k++) {
                    gradient[i - 1][j][k] = partialNodes[i][j] * this.activationFunctionPrime(nodes[i][k]) * nodes[i - 1][k];
                }
            }
        }

        //get the gradient for the input layer
        for (let j = 0; j < this.hiddenLayerNodes; j++) {

            //get partial product
            partialNodes[1][j] = this.weights[1][j].map((weight, index) => weight * partialNodes[2][index]).reduce((a, b) => a + b);

            for (let k = 0; k < this.inputLayerNodes; k++) {
                gradient[0][k][j] = partialNodes[1][j] * this.activationFunctionPrime(nodes[1][k]) * nodes[0][k];
            }
        }

        return gradient;
    }

    //apply gradient
    applyGradient(gradient: Gradient): void {
        for (let i = 0; i < this.weights.length; i++) {
            for (let j = 0; j < this.weights[i].length; j++) {
                for (let k = 0; k < this.weights[i][j].length; k++) {
                    this.weights[i][j][k] -= this.learningRate * gradient[i][j][k];
                }
            }
        }
    }

    //calculate gradient and apply
    backpropagate(inputs: number[], targetOutput: number[]): void {

        //get gradient
        const gradient = this.getGradient(inputs, targetOutput);

        //update weights
        this.applyGradient(gradient);
    }

}