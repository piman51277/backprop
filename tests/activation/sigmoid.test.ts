import {sigmoid, sigmoidPrime} from '../../src/activation/sigmoid';


test("sigmoid x = 0",()=>{
    expect(sigmoid(0)).toBe(0.5);
});

test("sigmoid x = 1",()=>{
    expect(sigmoid(1)).toBe(0.7310585786300049);
});

test("sigmoid x = -1",()=>{
    expect(sigmoid(-1)).toBe(0.2689414213699951);
});

test("sigmoidPrime x = 0",()=>{
    expect(sigmoidPrime(0.5)).toBe(0.25);
});

test("sigmoidPrime x = 1",()=>{
    expect(sigmoidPrime(0.7310585786300049)).toBe(0.19661193324148185);
});

test("sigmoidPrime x = -1",()=>{
    expect(sigmoidPrime(0.2689414213699951)).toBe(0.19661193324148185);
});