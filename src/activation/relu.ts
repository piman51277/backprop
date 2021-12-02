//relu activation function
export function relu(n: number): number {
    return n > 0 ? n : 0;
}

//derivative, takes output of relu
export function reluPrime(n: number): number {
    return n > 0 ? 1 : 0;
}