//squared error
export function squaredError(true_set: number[], predicted_set: number[]): number {

    //if the predicted set is not the same length as the true set, throw an error
    if (true_set.length !== predicted_set.length) throw "The predicted set must be the same length as the true set";

    return true_set.map((value, index) => (predicted_set[index] - value) ** 2).reduce((a, b) => a + b) * (1 / true_set.length);
}