import {relu, reluPrime} from '../../src/util/activation/relu';

test("relu x = 0", () => {
    expect(relu(0)).toBe(0);
});

test("relu x < 0", () => {
    expect(relu(-1)).toBe(0);
});

test("relu x > 0", () => {
    expect(relu(1)).toBe(1);
});

test("reluPrime x = 0",()=>{
    expect(reluPrime(0)).toBe(0);
});

test("reluPrime x < 0",()=>{
    expect(reluPrime(-1)).toBe(0);
});

test("reluPrime x > 0",()=>{
    expect(reluPrime(1)).toBe(1);
});