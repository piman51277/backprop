//sigmoid activation function
export function sigmoid(n: number): number {
    return 1 / (1 + Math.exp(-n));
}

//derivative, takes output of sigmoid
export function sigmoidPrime(output: number) {
    return output * (1 - output);
}