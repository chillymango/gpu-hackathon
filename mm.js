async function createMatrixMultiplicationPipeline(device) {
    const shaderModule = device.createShaderModule({
        code: await (await fetch('matrix-multiply.wgsl')).text()
    });

    const pipeline = device.createComputePipeline({
        layout: "auto",
        compute: {
            module: shaderModule,
            entryPoint: "main",
        },
    });

    return pipeline;
}

async function multiplyMatrices(device, queue, matrixA, matrixB) {
    const rowsA = matrixA.length;
    const colsA = matrixA[0].length;
    const rowsB = matrixB.length;
    const colsB = matrixB[0].length;

    if (colsA !== rowsB) {
        throw new Error("Matrix dimensions do not match for multiplication.");
    }

    // Flatten matrices
    const flatA = new Float32Array(rowsA * colsA);
    const flatB = new Float32Array(rowsB * colsB);
    const flatResult =
