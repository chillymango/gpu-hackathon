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
    const flatResult = new Float32Array(rowsA * colsB);

    matrixA.flat().forEach((val, i) => (flatA[i] = val));
    matrixB.flat().forEach((val, i) => (flatB[i] = val));

    // Create buffers
    const bufferA = device.createBuffer({
        size: flatA.byteLength + 8, // Extra space for size
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
    });

    const bufferB = device.createBuffer({
        size: flatB.byteLength + 8,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
    });

    const bufferResult = device.createBuffer({
        size: flatResult.byteLength + 8,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    // Write matrix dimensions at the start
    new Uint32Array(bufferA.getMappedRange()).set([rowsA, colsA]);
    new Float32Array(bufferA.getMappedRange(8)).set(flatA);
    bufferA.unmap();

    new Uint32Array(bufferB.getMappedRange()).set([rowsB, colsB]);
    new Float32Array(bufferB.getMappedRange(8)).set(flatB);
    bufferB.unmap();

    // Create compute pipeline
    const pipeline = await createMatrixMultiplicationPipeline(device);

    // Create bind group
    const bindGroupLayout = pipeline.getBindGroupLayout(0);
    const bindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
            { binding: 0, resource: { buffer: bufferA } },
            { binding: 1, resource: { buffer: bufferB } },
            { binding: 2, resource: { buffer: bufferResult } },
        ],
    });

    // Dispatch workgroups
    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(rowsA / 8), Math.ceil(colsB / 8));
    passEncoder.end();

    // Copy results to a readable buffer
    const stagingBuffer = device.createBuffer({
        size: flatResult.byteLength + 8,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    commandEncoder.copyBufferToBuffer(bufferResult, 0, stagingBuffer, 0, flatResult.byteLength + 8);
    queue.submit([commandEncoder.finish()]);

    // Read back results
    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const resultArray = new Float32Array(stagingBuffer.getMappedRange(8));
    stagingBuffer.unmap();

    // Convert back to 2D array
    const resultMatrix = [];
    for (let i = 0; i < rowsA; i++) {
        resultMatrix.push(resultArray.slice(i * colsB, (i + 1) * colsB));
    }

    return resultMatrix;
}

// Initialize WebGPU
async function main() {
    if (!navigator.gpu) {
        console.error("WebGPU is not supported on this device.");
        return;
    }

    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();
    const queue = device.queue;

    // Define test matrices
    const matrixA = [
        [1, 2, 3],
        [4, 5, 6]
    ];
    const matrixB = [
        [7, 8],
        [9, 10],
        [11, 12]
    ];

    console.log("Multiplying matrices...");
    const result = await multiplyMatrices(device, queue, matrixA, matrixB);
    console.log("Result:", result);
}

main();
