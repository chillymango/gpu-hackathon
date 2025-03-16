struct Matrix {
    size: vec2<u32>,  // Matrix dimensions (rows, cols)
    numbers: array<f32>,  // Flattened matrix data
};

@group(0) @binding(0) var<storage, read> a: Matrix;
@group(0) @binding(1) var<storage, read> b: Matrix;
@group(0) @binding(2) var<storage, read_write> result: Matrix;

@compute @workgroup_size(8, 8)  // Defines 8x8 workgroup threads
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let rowsA = a.size.x;
    let colsA = a.size.y;
    let colsB = b.size.y;

    let row = id.x;
    let col = id.y;

    if (row >= rowsA || col >= colsB) {
        return; // Out-of-bounds check
    }

    var sum: f32 = 0.0;
    for (var i: u32 = 0; i < colsA; i = i + 1) {
        let indexA = row * colsA + i;
        let indexB = i * colsB + col;
        sum = sum + a.numbers[indexA] * b.numbers[indexB];
    }

    let indexResult = row * colsB + col;
    result.numbers[indexResult] = sum;
}
