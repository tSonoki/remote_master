// ONNX Inference Module
// Handles model loading, inference, and result parsing

class ONNXInferenceEngine {
  constructor() {
    this.session = null;
    this.inferenceCanvas = null;
    this.inferenceContext = null;
    this.isWebGPUSupported = false;
    this.modelPath = './best.onnx';
    this.lastInferenceTime = 0;
    this.minInferenceInterval = 100; // 最小推論間隔（ms）
    
    // 検出結果の安定化フィルタ
    this.detectionHistory = []; // 過去の検出結果を保存
    this.maxHistoryLength = 5; // 保存する履歴の最大長
    this.confidenceThreshold = 0.3; // 最小信頼度閾値
    
    // Performance monitoring
    this.performanceStats = {
      totalInferences: 0,
      totalTime: 0,
      averageTime: 0,
      skippedFrames: 0,
      errorCount: 0
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

  // Parse detection results with stability filtering
  parseDetectionResults(results) {
    const output = results.output0;
    if (!output) {
      console.log("Available output keys:", Object.keys(results));
      return this.stabilizeDetections([]);
    }

    const outputData = output.data;
    const outputShape = output.dims; // [1, 300, 6] for your model
    
    const rawDetections = [];
    const numDetections = outputShape[1]; // 300
    const numFeatures = outputShape[2]; // 6 (x1, y1, x2, y2, confidence, class_id)

    // Your model format: [x1, y1, x2, y2, confidence, class_id]
    for (let i = 0; i < numDetections; i++) {
      const startIdx = i * numFeatures;

      const x1 = outputData[startIdx + 0];
      const y1 = outputData[startIdx + 1];
      const x2 = outputData[startIdx + 2];
      const y2 = outputData[startIdx + 3];
      const confidence = outputData[startIdx + 4];
      const classId = Math.round(outputData[startIdx + 5]);

      // 動的信頼度閾値でフィルタ（基本的な妥当性チェック付き）
      if (confidence > this.confidenceThreshold && 
          x1 >= 0 && y1 >= 0 && x2 > x1 && y2 > y1 && 
          x2 <= 640 && y2 <= 640) {
        
        const width = x2 - x1;
        const height = y2 - y1;
        
        // 最小サイズフィルタ（ノイズ除去）
        if (width > 10 && height > 10) {
          const detection = {
            bbox: { x: x1, y: y1, width: width, height: height },
            confidence: confidence,
            classId: classId,
            x1: x1, y1: y1, x2: x2, y2: y2,
            timestamp: Date.now()
          };
          rawDetections.push(detection);
        }
      }
    }

    // 検出結果の安定化フィルタを適用
    return this.stabilizeDetections(rawDetections);
  }

  // 検出結果の安定化フィルタ
  stabilizeDetections(currentDetections) {
    // 履歴に現在の検出結果を追加
    this.detectionHistory.push({
      timestamp: Date.now(),
      detections: currentDetections
    });

    // 履歴の長さを制限
    if (this.detectionHistory.length > this.maxHistoryLength) {
      this.detectionHistory.shift();
    }

    // 履歴が不十分な場合は現在の結果をそのまま返す
    if (this.detectionHistory.length < 2) {
      return currentDetections;
    }

    // 安定した検出結果をフィルタリング
    const stableDetections = [];
    
    for (const detection of currentDetections) {
      let stabilityCount = 0;
      
      // 過去の履歴で類似の検出があるかチェック
      for (const historyEntry of this.detectionHistory.slice(-3)) { // 直近3フレームをチェック
        for (const histDetection of historyEntry.detections) {
          if (this.isSimilarDetection(detection, histDetection)) {
            stabilityCount++;
            break;
          }
        }
      }
      
      // 少なくとも2回以上検出されているか、高い信頼度の場合は採用
      if (stabilityCount >= 2 || detection.confidence > 0.7) {
        stableDetections.push(detection);
      }
    }

    return stableDetections;
  }

  // 2つの検出結果が類似しているかチェック
  isSimilarDetection(det1, det2, threshold = 50) {
    if (det1.classId !== det2.classId) return false;
    
    // バウンディングボックスの中心点距離で判定
    const center1 = {
      x: det1.x1 + (det1.x2 - det1.x1) / 2,
      y: det1.y1 + (det1.y2 - det1.y1) / 2
    };
    const center2 = {
      x: det2.x1 + (det2.x2 - det2.x1) / 2,
      y: det2.y1 + (det2.y2 - det2.y1) / 2
    };
    
    const distance = Math.sqrt(
      Math.pow(center1.x - center2.x, 2) + 
      Math.pow(center1.y - center2.y, 2)
    );
    
    return distance < threshold;
  }

  // Run ONNX inference on video frame with frame skipping and error handling
  async runInference(videoElement, side = "unknown") {
    if (!this.session || !videoElement || videoElement.videoWidth === 0) {
      return null;
    }

    // フレームスキッピング - 最小間隔をチェック
    const currentTime = performance.now();
    if (currentTime - this.lastInferenceTime < this.minInferenceInterval) {
      this.performanceStats.skippedFrames++;
      return null;
    }
    this.lastInferenceTime = currentTime;

    // ビデオフレームの有効性チェック
    if (videoElement.readyState < 2) { // HAVE_CURRENT_DATA
      console.warn(`Video not ready for inference (readyState: ${videoElement.readyState})`);
      return null;
    }

    try {
      const startTime = performance.now();
      
      const inputTensor = this.preprocessFrame(videoElement);
      if (!inputTensor) {
        console.warn("Failed to preprocess frame");
        return null;
      }

      // テンソル作成時のエラーハンドリング
      let tensor, feeds;
      try {
        tensor = new ort.Tensor('float32', inputTensor, [1, 3, 640, 640]);
        feeds = { images: tensor };
      } catch (tensorError) {
        console.error(`Tensor creation failed: ${tensorError.message}`);
        return null;
      }

      // 推論実行（タイムアウト付き）
      const results = await Promise.race([
        this.session.run(feeds),
        new Promise((_, reject) => 
          setTimeout(() => reject(new Error('Inference timeout')), 5000)
        )
      ]);
      
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
      console.error(`ONNX inference failed on ${side} side:`, {
        message: error.message,
        stack: error.stack?.split('\n').slice(0, 3).join('\n'),
        timestamp: new Date().toISOString()
      });
      
      // エラーカウンターを追加
      this.performanceStats.errorCount = (this.performanceStats.errorCount || 0) + 1;
      
      // 連続エラーが多い場合は推論間隔を延長
      if (this.performanceStats.errorCount > 5) {
        this.minInferenceInterval = Math.min(1000, this.minInferenceInterval * 1.5);
        console.warn(`High error rate detected. Increasing inference interval to ${this.minInferenceInterval}ms`);
      }
      
      return null;
    }
  }

  // Get performance statistics
  getPerformanceStats() {
    return { 
      ...this.performanceStats,
      skipRate: this.performanceStats.skippedFrames / (this.performanceStats.totalInferences + this.performanceStats.skippedFrames) * 100
    };
  }

  // 推論間隔を動的調整
  adjustInferenceInterval(averageTime) {
    if (averageTime > 100) {
      this.minInferenceInterval = Math.min(500, this.minInferenceInterval + 50);
    } else if (averageTime < 50) {
      this.minInferenceInterval = Math.max(50, this.minInferenceInterval - 10);
    }
  }

  // Reset performance statistics
  resetPerformanceStats() {
    this.performanceStats = {
      totalInferences: 0,
      totalTime: 0,
      averageTime: 0,
      skippedFrames: 0,
      errorCount: 0
    };
    this.lastInferenceTime = 0;
    this.detectionHistory = [];
    this.minInferenceInterval = 100; // デフォルト値にリセット
    console.log("ONNX performance stats and detection history reset");
  }
  
  // Set confidence threshold dynamically
  setConfidenceThreshold(threshold) {
    this.confidenceThreshold = Math.max(0.1, Math.min(0.9, threshold));
    console.log(`Confidence threshold set to ${this.confidenceThreshold}`);
  }
  
  // Get detection statistics
  getDetectionStats() {
    const recentHistory = this.detectionHistory.slice(-10);
    const totalDetections = recentHistory.reduce((sum, entry) => sum + entry.detections.length, 0);
    const avgDetectionsPerFrame = recentHistory.length > 0 ? totalDetections / recentHistory.length : 0;
    
    return {
      historyLength: this.detectionHistory.length,
      avgDetectionsPerFrame: avgDetectionsPerFrame.toFixed(2),
      confidenceThreshold: this.confidenceThreshold,
      minInferenceInterval: this.minInferenceInterval
    };
  }

  // Check if WebGPU is being used
  isUsingWebGPU() {
    return this.isWebGPUSupported && this.session;
  }

  // メモリクリーンアップ
  cleanupMemory() {
    if (this.inferenceCanvas) {
      this.inferenceContext = null;
      this.inferenceCanvas.width = 1;
      this.inferenceCanvas.height = 1;
    }
    
    // ガベージコレクションの実行を試行
    if (window.gc) {
      window.gc();
    }
  }

  // Dispose of resources
  dispose() {
    if (this.session) {
      this.session.release();
      this.session = null;
    }
    this.cleanupMemory();
    this.inferenceCanvas = null;
    this.inferenceContext = null;
    console.log("ONNX inference engine disposed");
  }
}

// Export the class for use in other modules
export { ONNXInferenceEngine };