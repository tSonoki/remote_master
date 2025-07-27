import { ONNXInferenceEngine } from './onnxInference.js';

const signalingSocket = new WebSocket("ws://localhost:8080");
const autorunSocket = new WebSocket("ws://127.0.0.1:8081");
let peerConnection;
let remoteDataChannel = null;
let iceCandidateQueue = []; // リモートSDP設定前のICE候補を保存するキュー
let stream = null;
const streamCaptureBtn = document.getElementById("stream-capture");
const cameraSelect = document.getElementById("camera-select");

// ONNX Inference Engine
let onnxEngine = null;
let detectionCanvas = null;
let detectionContext = null;
let largeDetectionCanvas = null;
let largeDetectionContext = null;
let isInferenceEnabled = true; // 推論の有効/無効状態

// Initialize ONNX model with WebGPU acceleration
async function initONNXModel() {
  try {
    onnxEngine = new ONNXInferenceEngine();
    const success = await onnxEngine.initializeModel();
    
    if (success) {
      // Get detection canvas overlay (small one)
      detectionCanvas = document.getElementById("detection-canvas");
      if (detectionCanvas) {
        detectionContext = detectionCanvas.getContext("2d");
      }

      // Get large detection canvas
      largeDetectionCanvas = document.getElementById("large-detection-canvas");
      if (largeDetectionCanvas) {
        largeDetectionContext = largeDetectionCanvas.getContext("2d");
        console.log("ONNX inference engine initialized on answer side");
        return true;
      } else {
        console.error("Large detection canvas not found in DOM");
        return false;
      }
    } else {
      console.error("Failed to initialize ONNX inference engine");
      return false;
    }
  } catch (error) {
    console.error("Error initializing ONNX inference engine:", error);
    return false;
  }
}

// Stream video to large canvas
function streamToCanvas(videoElement, canvas, context) {
  if (!canvas || !context || !videoElement || videoElement.videoWidth === 0)
    return;

  // Draw video frame to large canvas (1920x1080)
  context.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
}

// Draw bounding boxes on large canvas (1920x1080)
function drawBoundingBoxesOnLargeCanvas(detections) {
  if (!largeDetectionCanvas || !largeDetectionContext) {
    console.error("Large canvas not available");
    return;
  }

  // Note: We don't clear the canvas here because we want to keep the video stream

  if (detections.length === 0) {
    return;
  }

  // Scale factors from 640x640 model input to 1920x1080 canvas
  const scaleX = 1920 / 640;
  const scaleY = 1080 / 640;

  // COCO class names
  const classNames = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
  ];

  detections.forEach((detection, idx) => {
    const bbox = detection.bbox;

    // Scale coordinates from 640x640 to 1920x1080
    const x = bbox.x * scaleX;
    const y = bbox.y * scaleY;
    const width = bbox.width * scaleX;
    const height = bbox.height * scaleY;

    // Choose color for this detection
    const hue = (idx * 137.5) % 360;
    const boxColor = `hsl(${hue}, 100%, 50%)`;

    // Draw bounding box
    largeDetectionContext.strokeStyle = boxColor;
    largeDetectionContext.lineWidth = 6; // Thicker lines for large canvas
    largeDetectionContext.strokeRect(x, y, width, height);

    // Prepare label text
    const className =
      classNames[detection.classId] || `Class ${detection.classId}`;
    const confidence = (detection.confidence * 100).toFixed(1);
    const label = `${className} ${confidence}%`;

    // Set font and measure text (larger for large canvas)
    largeDetectionContext.font = "bold 32px Arial";
    const textMetrics = largeDetectionContext.measureText(label);
    const textWidth = textMetrics.width;
    const textHeight = 40;

    // Calculate label position
    const labelX = Math.max(0, x);
    const labelY = Math.max(textHeight, y);

    // Draw label background
    largeDetectionContext.fillStyle = boxColor;
    largeDetectionContext.fillRect(
      labelX,
      labelY - textHeight,
      textWidth + 16,
      textHeight + 8
    );

    // Draw label text
    largeDetectionContext.fillStyle = "white";
    largeDetectionContext.fillText(label, labelX + 8, labelY - 8);
  });
}

// Run ONNX inference using the inference engine
async function runInference(videoElement) {
  if (!onnxEngine || !videoElement || videoElement.videoWidth === 0) return null;
  
  return await onnxEngine.runInference(videoElement, "answer");
}

// Performance monitoring variables
let inferenceCount = 0;
let totalInferenceTime = 0;
let dynamicInterval = 33; // Start with 33ms (30 FPS)

// Start continuous inference on video stream with adaptive FPS
function startVideoInference(videoElement) {
  async function inferenceLoop() {
    if (videoElement && videoElement.videoWidth > 0 && largeDetectionCanvas && largeDetectionContext) {
      const startTime = performance.now();

      // Stream video to large canvas first
      streamToCanvas(videoElement, largeDetectionCanvas, largeDetectionContext);

      // Only run inference if enabled
      if (isInferenceEnabled) {
        const results = await runInference(videoElement);
        if (results && results.detections) {
          // Draw bounding boxes on large canvas over the video stream
          drawBoundingBoxesOnLargeCanvas(results.detections);

          // Send results through WebRTC data channel
          if (remoteDataChannel && remoteDataChannel.readyState === "open") {
            remoteDataChannel.send(
              JSON.stringify({
                type: "inferenceResults",
                payload: {
                  timestamp: Date.now(),
                  detectionsCount: results.detections.length,
                  detections: results.detections,
                  fps: (1000 / dynamicInterval).toFixed(1),
                },
              })
            );
          }
        }
      }

      const endTime = performance.now();
      const inferenceTime = endTime - startTime;

      // Update performance metrics
      inferenceCount++;
      totalInferenceTime += inferenceTime;

      if (inferenceCount % 30 === 0) {
        // Log every 30 inferences
        const avgInferenceTime = totalInferenceTime / 30;

        // Adaptive interval adjustment for WebGPU
        if (avgInferenceTime < 20) {
          // Very fast inference (WebGPU working well)
          dynamicInterval = Math.max(16, dynamicInterval - 2); // Up to 60 FPS
        } else if (avgInferenceTime < 40) {
          // Good inference speed
          dynamicInterval = Math.max(33, dynamicInterval - 1); // Up to 30 FPS
        } else if (avgInferenceTime > 80) {
          // Slow inference (fallback to CPU)
          dynamicInterval = Math.min(200, dynamicInterval + 10); // Down to 5 FPS
        }

        totalInferenceTime = 0;
        inferenceCount = 0;
      }
    }

    // Schedule next inference
    setTimeout(inferenceLoop, dynamicInterval);
  }

  // Start the inference loop
  inferenceLoop();
}

let mr1000aReceiveInfo = {
  lat: 0,
  lon: 0,
  gnssQuality: 0,
  gnssSpeed: 0,
  heading: 0,
  headingError: 0,
  lateralError: 0,
  steerAngle: 0,
  realSteerAngle: 0,
  stopStatus: 0,
};

signalingSocket.onmessage = async (event) => {
  const { type, payload } = JSON.parse(event.data);

  if (type === "offer") {
    peerConnection = new RTCPeerConnection({
      iceServers: [{ urls: "stun:10.100.0.35:3478" }],
    });

    await peerConnection.setRemoteDescription(
      new RTCSessionDescription({ type: "offer", sdp: payload.sdp })
    );

    stream.getTracks().forEach((track) => {
      peerConnection.addTrack(track, stream);
    });

    peerConnection.ondatachannel = (event) => {
      remoteDataChannel = event.channel;

      remoteDataChannel.onopen = () => {
        console.log("DataChannel is open");
        setInterval(() => {
          if (remoteDataChannel != null) {
            remoteDataChannel.send(
              JSON.stringify({
                type: "outputAutorunInfo",
                payload: mr1000aReceiveInfo,
              })
            );
          }
        }, 33); // 約30Hz
      };

      remoteDataChannel.onmessage = (event) => {
        const data = JSON.parse(event.data);
        switch (data.type) {
          case "inputAutorunInfo":
            if (autorunSocket != null) {
              const remoteDrivingData = data;
              autorunSocket.send(
                JSON.stringify({
                  type: "inputAutorunInfo",
                  payload: { inputInfo: remoteDrivingData },
                })
              );
            }
            break;
          case "videoQualityChange":
            console.log("Received video quality change request:", data.payload); // 受信ログを追加
            const videoSender = peerConnection.getSenders().find(
              (sender) => sender.track && sender.track.kind === "video"
            );
            if (videoSender) {
              const params = videoSender.getParameters();
              if (params.encodings && params.encodings.length > 0) {
                // data.payload.bitrate を参照するように修正
                params.encodings[0].maxBitrate = data.payload.bitrate;
                videoSender.setParameters(params)
                  .then(() => {
                    console.log(`Video bitrate successfully changed to: ${data.payload.bitrate}`);
                  })
                  .catch(e => {
                    console.error("Failed to set video bitrate:", e);
                  });
              }
            }
            break;
          case "offerInferenceResults":
            console.log(
              "Received inference results from offer side:",
              data.payload
            );
            break;
        }
      };
    };

    peerConnection.onicecandidate = (event) => {
      if (event.candidate) {
        signalingSocket.send(
          JSON.stringify({
            type: "ice-answer",
            payload: { candidate: event.candidate },
          })
        );
      }
    };

    while (iceCandidateQueue.length > 0) {
      const candidate = iceCandidateQueue.shift();
      await peerConnection.addIceCandidate(candidate);
    }

    const answer = await peerConnection.createAnswer();
    await peerConnection.setLocalDescription(answer);

    signalingSocket.send(
      JSON.stringify({ type: "answer", payload: { sdp: answer.sdp } })
    );
  } else if (type === "ice-offer") {
    const candidate = new RTCIceCandidate(payload.candidate);

    if (peerConnection && peerConnection.remoteDescription) {
      await peerConnection.addIceCandidate(candidate);
    } else {
      iceCandidateQueue.push(candidate);
    }
  }
};

signalingSocket.onopen = () => {
  signalingSocket.send(
    JSON.stringify({ type: "register-answer", payload: { id: "answer" } })
  );
};

autorunSocket.onopen = () => {
  autorunSocket.send(JSON.stringify({ type: "remote-control" }));
};

autorunSocket.onmessage = (event) => {
  const nodeData = JSON.parse(event.data);
  if (nodeData.type === "autorun-output-data") {
    const { outputAutorunInfo } = nodeData.payload;
    mr1000aReceiveInfo = outputAutorunInfo;
  }
};

function populateCameras() {
  if (!("mediaDevices" in navigator)) return;
  navigator.mediaDevices.enumerateDevices().then((mediaDevices) => {
    while (cameraSelect.options.length > 0) {
      cameraSelect.remove(0);
    }
    const defaultOption = document.createElement("option");
    defaultOption.id = "default";
    defaultOption.textContent = "Default Camera";
    cameraSelect.appendChild(defaultOption);

    const videoInputDevices = mediaDevices.filter(
      (mediaDevice) => mediaDevice.kind === "videoinput"
    );
    if (videoInputDevices.length > 0) {
      cameraSelect.disabled = false;
    }
    videoInputDevices.forEach((videoInputDevice, index) => {
      if (!videoInputDevice.deviceId) {
        return;
      }
      const option = document.createElement("option");
      option.id = videoInputDevice.deviceId;
      option.textContent = videoInputDevice.label || `Camera ${index + 1}`;
      option.selected = deviceId == option.id;
      cameraSelect.appendChild(option);
    });
  });
}

window.addEventListener("DOMContentLoaded", populateCameras);
if ("mediaDevices" in navigator) {
  navigator.mediaDevices.addEventListener("devicechange", populateCameras);
}

let deviceId = "default";
cameraSelect.onchange = (_) => {
  deviceId = cameraSelect.selectedOptions[0].id;
};

streamCaptureBtn.addEventListener("click", async () => {
  stream = await navigator.mediaDevices.getUserMedia({
    video: {
      width: Number(document.getElementById("video-width").value),
      height: Number(document.getElementById("video-height").value),
      frameRate: Number(document.getElementById("video-rate").value),
      deviceId: String(deviceId),
    },
  });
  const videoElement = document.getElementById("local-video");
  videoElement.srcObject = stream;

  videoElement.addEventListener("loadeddata", async () => {
    await new Promise(resolve => setTimeout(resolve, 100));
    
    const modelInitialized = await initONNXModel();
    if (modelInitialized && largeDetectionCanvas && largeDetectionContext) {
      startVideoInference(videoElement);
    }
  });

  const inferenceOnRadio = document.getElementById("inference-on");
  const inferenceOffRadio = document.getElementById("inference-off");
  
  inferenceOnRadio.addEventListener("change", () => {
    if (inferenceOnRadio.checked) {
      isInferenceEnabled = true;
    }
  });
  
  inferenceOffRadio.addEventListener("change", () => {
    if (inferenceOffRadio.checked) {
      isInferenceEnabled = false;
    }
  });
});

// ===== ここから遅延ログ機能 (詳細版) =====
const delayLogs = [];

setInterval(async function debugRTCStats() {
  if (!peerConnection) return;
  const stats = await peerConnection.getStats();

  let logRow = {
    timestamp: new Date().toLocaleTimeString('ja-JP'),
    roundTripTime: null,
    targetBitrate: null, // targetBitrate を追加
    totalPacketSendDelay: null,
    '[totalPacketSendDelay/packetsSent_in_ms]': null,
    totalEncodeTime: null,
    '[totalEncodeTime/framesEncoded_in_ms]': null,
    packetsSent: null,
    bytesSent: null,
    framesEncoded: null,
    '[framesEncoded/s]': null,
    frameWidth: null,
    frameHeight: null,
    framesPerSecond: null,
  };

  let prevFramesEncoded = null;
  let prevTimestamp = null;

  stats.forEach((report) => {
    if (report.type === "candidate-pair" && report.state === "succeeded") {
      logRow.roundTripTime = report.currentRoundTripTime ?? report.roundTripTime ?? null;
    }

    if (report.type === "outbound-rtp" && report.kind === "video") {
      logRow.targetBitrate = report.targetBitrate ?? null; // targetBitrate を取得
      logRow.packetsSent = report.packetsSent ?? null;
      logRow.bytesSent = report.bytesSent ?? null;
      logRow.totalEncodeTime = report.totalEncodeTime ?? null;
      logRow.framesEncoded = report.framesEncoded ?? null;
      logRow.totalPacketSendDelay = report.totalPacketSendDelay ?? null;

      if (report.totalEncodeTime !== undefined && report.framesEncoded > 0) {
        logRow['[totalEncodeTime/framesEncoded_in_ms]'] =
          ((report.totalEncodeTime / report.framesEncoded) * 1000).toFixed(3);
      }

      if (report.totalPacketSendDelay !== undefined && report.packetsSent > 0) {
        logRow['[totalPacketSendDelay/packetsSent_in_ms]'] =
          ((report.totalPacketSendDelay / report.packetsSent) * 1000).toFixed(6);
      }

      // framesEncoded/s の計算
      if (prevFramesEncoded !== null && prevTimestamp !== null && report.framesEncoded) {
        const timeDiff = (report.timestamp - prevTimestamp) / 1000; // in seconds
        const frameDiff = report.framesEncoded - prevFramesEncoded;
        if (timeDiff > 0) {
            logRow['[framesEncoded/s]'] = (frameDiff / timeDiff).toFixed(2);
        }
      }
      prevFramesEncoded = report.framesEncoded;
      prevTimestamp = report.timestamp;
    }

    if (report.type === "media-source" && report.kind === "video") {
      logRow.frameWidth = report.width ?? logRow.frameWidth;
      logRow.frameHeight = report.height ?? logRow.frameHeight;
      logRow.framesPerSecond = report.framesPerSecond ?? logRow.framesPerSecond;
    }
  });

  delayLogs.push(logRow);
}, 1000);

// CSV保存
function saveDelayLogsAsCSV() {
  if (delayLogs.length === 0) return;

  const headers = Object.keys(delayLogs[0]);
  const csvContent = [
    headers.join(","),
    ...delayLogs.map(log => headers.map(h => log[h] ?? "").join(","))
  ].join("\n");

  const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;

  const now = new Date();
  const jst = new Date(now.getTime() + 9 * 60 * 60 * 1000); // 日本時間 (JST) に変換
  const pad = n => n.toString().padStart(2, "0");
  const ts = `${jst.getUTCFullYear()}-${pad(jst.getUTCMonth() + 1)}-${pad(jst.getUTCDate())}_${pad(jst.getUTCHours())}-${pad(jst.getUTCMinutes())}-${pad(jst.getUTCSeconds())}`;

  a.download = `answer_webrtc_delay_log_${ts}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

document.getElementById("save-delay-log").addEventListener("click", saveDelayLogsAsCSV);
// ===== 遅延ログ機能ここまで =====
