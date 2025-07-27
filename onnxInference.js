// ONNX Inference Module
// Handles model loading, inference, and result parsing

class ONNXInferenceEngine {
  constructor() {
    this.session = null;
    this.inferenceCanvas = null;
    this.inferenceContext = null;
    this.isWebGPUSupported = false;
    this.modelPath = './best.onnx';
    
    // Performance monitoring
    this.performanceStats = {
      totalInferences: 0,
      totalTime: 0,
      averageTime: 0
    };
  }

  // Check WebGPU support
  async checkWebGPUSupport() {
    if (!navigator.gpu) {
      console.warn("WebGPU not supported in this browser");
      return false;
    }
    
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) {
        console.warn("WebGPU adapter not available");
        return false;
      }
      console.log("WebGPU is supported and available");
      return true;
    } catch (error) {
      console.warn("WebGPU check failed:", error);
      return false;
    }
  }

  // Initialize ONNX model with WebGPU acceleration
  async initializeModel(modelPath = null) {
    try {
      if (modelPath) this.modelPath = modelPath;
      
      this.isWebGPUSupported = await this.checkWebGPUSupport();
      
      let sessionOptions = {};
      
      if (this.isWebGPUSupported) {
        sessionOptions = {
          executionProviders: ['webgpu'],
          graphOptimizationLevel: 'all',
          enableCpuMemArena: false,
          enableMemPattern: false,
          executionMode: 'sequential'
        };
        console.log("Initializing ONNX model with WebGPU acceleration...");
      } else {
        sessionOptions = {
          executionProviders: ['wasm'],
          graphOptimizationLevel: 'all'
        };
        console.log("Initializing ONNX model with CPU (WASM) fallback...");
      }

      this.session = await ort.InferenceSession.create(this.modelPath, sessionOptions);
      
      if (this.isWebGPUSupported) {
        console.log("✅ ONNX model loaded successfully with WebGPU acceleration");
      } else {
        console.log("⚠️ ONNX model loaded with CPU fallback (WebGPU not available)");
        console.log("Note: Some nodes may be assigned to CPU for optimization - this is normal behavior");
      }

      // Create canvas for inference processing
      this.inferenceCanvas = document.createElement('canvas');
      this.inferenceContext = this.inferenceCanvas.getContext('2d');
      
      console.log("ONNX inference engine initialized");
      return true;
    } catch (error) {
      console.error("Failed to load ONNX model:", error);
      
      // Fallback to CPU if WebGPU fails
      try {
        console.log("Attempting CPU fallback...");
        this.session = await ort.InferenceSession.create(this.modelPath, {
          executionProviders: ['wasm']
        });
        console.log("✅ ONNX model loaded with CPU fallback");
        return true;
      } catch (fallbackError) {
        console.error("CPU fallback also failed:", fallbackError);
        return false;
      }
    }
  }

  // Preprocess video frame for ONNX inference
  preprocessFrame(videoElement, targetWidth = 640, targetHeight = 640) {
    if (!this.inferenceCanvas || !this.inferenceContext) return null;
    
    this.inferenceCanvas.width = targetWidth;
    this.inferenceCanvas.height = targetHeight;
    
    // Draw video frame to canvas
    this.inferenceContext.drawImage(videoElement, 0, 0, targetWidth, targetHeight);
    
    // Get image data
    const imageData = this.inferenceContext.getImageData(0, 0, targetWidth, targetHeight);
    const pixels = imageData.data;
    
    // Convert to RGB and normalize to [0, 1]
    const inputTensor = new Float32Array(3 * targetWidth * targetHeight);
    
    for (let i = 0; i < pixels.length; i += 4) {
      const pixelIndex = i / 4;
      const r = pixels[i] / 255.0;
      const g = pixels[i + 1] / 255.0;
      const b = pixels[i + 2] / 255.0;
      
      // CHW format (channels first)
      inputTensor[pixelIndex] = r; // R channel
      inputTensor[targetWidth * targetHeight + pixelIndex] = g; // G channel
      inputTensor[2 * targetWidth * targetHeight + pixelIndex] = b; // B channel
    }
    
    return inputTensor;
  }

  // Parse detection results - Updated for model format [1, 300, 6]
  parseDetectionResults(results) {
    const output = results.output0;
    if (!output) {
      console.log("Available output keys:", Object.keys(results));
      return [];
    }

    const outputData = output.data;
    const outputShape = output.dims; // [1, 300, 6] for your model
    console.log("Output shape:", outputShape);
    console.log("Output data length:", outputData.length);

    const detections = [];
    const numDetections = outputShape[1]; // 300
    const numFeatures = outputShape[2]; // 6 (x1, y1, x2, y2, confidence, class_id)

    console.log(`Processing ${numDetections} detections with ${numFeatures} features each`);

    // Your model format: [x1, y1, x2, y2, confidence, class_id]
    for (let i = 0; i < numDetections; i++) {
      const startIdx = i * numFeatures;

      // バウンディングボックス座標 (x1, y1, x2, y2形式)
      const x1 = outputData[startIdx + 0];
      const y1 = outputData[startIdx + 1];
      const x2 = outputData[startIdx + 2];
      const y2 = outputData[startIdx + 3];
      const confidence = outputData[startIdx + 4];
      const classId = Math.round(outputData[startIdx + 5]);

      // 信頼度閾値でフィルタ
      if (confidence > 0.3) { // 0.3に下げて検出しやすく
        // x1,y1,x2,y2 から width, height を計算
        const width = x2 - x1;
        const height = y2 - y1;

        const detection = {
          bbox: {
            x: x1,
            y: y1,
            width: width,
            height: height,
          },
          confidence: confidence,
          classId: classId,
          x1: x1,
          y1: y1,
          x2: x2,
          y2: y2,
        };

        detections.push(detection);
        console.log(`Valid detection ${detections.length-1}: x1=${x1.toFixed(1)}, y1=${y1.toFixed(1)}, x2=${x2.toFixed(1)}, y2=${y2.toFixed(1)}, conf=${confidence.toFixed(3)}, class=${classId}`);
      }
    }

    return detections;
  }

  // Run ONNX inference on video frame
  async runInference(videoElement, side = "unknown") {
    if (!this.session || !videoElement || videoElement.videoWidth === 0) return null;

    try {
      const startTime = performance.now();
      
      const inputTensor = this.preprocessFrame(videoElement);
      if (!inputTensor) return null;

      const tensor = new ort.Tensor('float32', inputTensor, [1, 3, 640, 640]);
      const feeds = { images: tensor };

      const results = await this.session.run(feeds);
      
      const endTime = performance.now();
      const inferenceTime = endTime - startTime;
      
      // Update performance stats
      this.performanceStats.totalInferences++;
      this.performanceStats.totalTime += inferenceTime;
      this.performanceStats.averageTime = this.performanceStats.totalTime / this.performanceStats.totalInferences;

      // 詳細な推論結果をコンソール出力
      console.log(`=== ${side.toUpperCase()} SIDE INFERENCE RESULTS ===`);
      console.log("Raw results keys:", Object.keys(results));
      
      Object.keys(results).forEach(key => {
        const result = results[key];
        console.log(`${key}:`, {
          dims: result.dims,
          type: result.type,
          dataLength: result.data.length,
          sampleData: result.data.slice(0, 10) // 最初の10個のデータを表示
        });
      });

      // 検出結果をパース
      const detections = this.parseDetectionResults(results);
      console.log(`Found ${detections.length} detections with confidence > 0.3`);
      detections.forEach((detection, idx) => {
        console.log(`Detection ${idx}:`, {
          classId: detection.classId,
          confidence: detection.confidence.toFixed(3),
          bbox: {
            x: detection.bbox.x.toFixed(1),
            y: detection.bbox.y.toFixed(1),
            width: detection.bbox.width.toFixed(1),
            height: detection.bbox.height.toFixed(1),
          },
        });
      });
      console.log("=====================================");

      return { 
        rawResults: results, 
        detections: detections,
        inferenceTime: inferenceTime,
        performanceStats: { ...this.performanceStats }
      };
    } catch (error) {
      console.error(`ONNX inference failed on ${side} side:`, error);
      return null;
    }
  }

  // Get performance statistics
  getPerformanceStats() {
    return { ...this.performanceStats };
  }

  // Reset performance statistics
  resetPerformanceStats() {
    this.performanceStats = {
      totalInferences: 0,
      totalTime: 0,
      averageTime: 0
    };
  }

  // Check if WebGPU is being used
  isUsingWebGPU() {
    return this.isWebGPUSupported && this.session;
  }

  // Dispose of resources
  dispose() {
    if (this.session) {
      this.session.release();
      this.session = null;
    }
    this.inferenceCanvas = null;
    this.inferenceContext = null;
    console.log("ONNX inference engine disposed");
  }
}

// Export the class for use in other modules
export { ONNXInferenceEngine };