const canvas = document.getElementById("gpuCanvas");

if (!navigator.gpu) {
    console.error("WebGPU not supported on this browser.");
} else {
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();
    console.log("WebGPU is working!", device);
}
