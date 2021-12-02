import {squaredError} from "../../src/error/squaredError";

test("squaredError",()=>{
    expect(squaredError([1,2,3,4,5],[1,2,3,4,5])).toBe(0);
    expect(squaredError([1,2,3,4,5],[1,2,3,4,6])).toBe(0.2);
    expect(squaredError([1,2,3,4,5],[1,2,3,4,4])).toBe(0.2);
});