import { NeuralNet } from '../../classes/NeuralNet';

export type TrainingSetEntry = {
    input: number[],
    output: number[]
}

export type TrainingSet = TrainingSetEntry[];

export function trainNet(net: NeuralNet, trainingSet: TrainingSet, epochs: number): NeuralNet {
    for (let i = 0; i < epochs; i++) {
        for (const entry of trainingSet) {
            net.backpropagate(entry.input, entry.output);
        }
    }
    return net;
}