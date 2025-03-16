struct Matrix {
  data: array<f32>,
};

@group(0) @binding(0)
var<storage, read> matrixA: Matrix;

@group(0) @binding(1)
var<storage, read> matrixB: Matrix;

@group(0) @binding(2)
var<storage, read_write> matrixC: Matrix;

@group(0) @binding(3)
var<uniform> dims: u32;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
  let row: u32 = GlobalInvocationID.x;
  let col: u32 = GlobalInvocationID.y;
  let N: u32 = dims;

  if (row < N && col < N) {
    var sum: f32 = 0.0;
    for (var k: u32 = 0u; k < N; k = k + 1u) {
      let a: f32 = matrixA.data[row * N + k];
      let b: f32 = matrixB.data[k * N + col];
      sum = sum + a * b;
    }
    matrixC.data[row * N + col] = sum;
  }
}
