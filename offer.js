//This repository is develoveped for virtual tractor project.

import {
  prioritizeSelectedVideoCodec,
  // setVideoQuality, // 外部ファイルからのインポートを削除
} from "./webrtcFunction.js";
import {
  onGamepadConnected,
  onGamepadDisconnected,
  getOrderedGamepads,
} from "./gamepad.js";
import {
  gamepadToAutorunInfo,
  drawGamepadInfo,
  farminGamepadToAutorunInfo,
} from "./gamepadfunction.js";
import startCanvasRendering from "./drawCanvas.js";
import { ONNXInferenceEngine } from './onnxInference.js';

const gamepadToAutorunInfoCon = gamepadToAutorunInfo();
const farminGamepadToAutorunInfoCon = farminGamepadToAutorunInfo();
const signalingWebSocket = new WebSocket("ws://localhost:8080");
// const virtualWebSocket = new WebSocket("ws://localhost:9090/"); // エラーの原因①：WebSocketサーバーへの接続
let peerConnection;
let movieChannel;
let dataChannel;
const videoElement = document.getElementById("remote-video");
const canvasElement = document.getElementById("video-canvas");
const canvasSonoki = document.getElementById('video-canvas-sonoki');
const video = document.getElementById('remote-video');

// ONNX Inference Engine
let onnxEngine = null;
let largeDetectionCanvas = null;
let largeDetectionContext = null;
let isInferenceEnabled = true; // 推論の有効/無効状態
let currentDetections = []; // 現在の検出結果を保存

// Initialize ONNX model with WebGPU acceleration
async function initONNXModel() {
  try {
    onnxEngine = new ONNXInferenceEngine();
    const success = await onnxEngine.initializeModel();
    
    if (success) {
      // Get large detection canvas
      largeDetectionCanvas = document.getElementById("large-detection-canvas");
      largeDetectionContext = largeDetectionCanvas.getContext("2d");
      console.log("ONNX inference engine initialized on offer side");
      return true;
    } else {
      console.error("Failed to initialize ONNX inference engine on offer side");
      return false;
    }
  } catch (error) {
    console.error("Error initializing ONNX inference engine on offer side:", error);
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
    console.error("Offer Large canvas not available");
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
  
  return await onnxEngine.runInference(videoElement, "offer");
}

// Performance monitoring variables for offer side
let inferenceCount = 0;
let totalInferenceTime = 0;
let dynamicInterval = 33; // Start with 33ms (30 FPS)

// Start continuous inference on received video stream with adaptive FPS
function startVideoInference(videoElement) {
  async function inferenceLoop() {
    if (videoElement && videoElement.videoWidth > 0) {
      const startTime = performance.now();

      // Stream video to large canvas
      streamToCanvas(videoElement, largeDetectionCanvas, largeDetectionContext);

      // Only run inference if enabled
      if (isInferenceEnabled) {
        const results = await runInference(videoElement);
        if (results && results.detections) {
          // Update current detections for video-canvas rendering
          currentDetections = results.detections;
          
          // Draw bounding boxes on large canvas
          drawBoundingBoxesOnLargeCanvas(results.detections);

          // Send results through WebRTC data channel
          if (dataChannel && dataChannel.readyState === "open") {
            dataChannel.send(
              JSON.stringify({
                type: "offerInferenceResults",
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
      } else {
        // Clear detection displays when inference is disabled
        currentDetections = []; // Clear current detections for video-canvas
      }

      const endTime = performance.now();
      const inferenceTime = endTime - startTime;

      // Update performance metrics
      inferenceCount++;
      totalInferenceTime += inferenceTime;

      if (inferenceCount % 30 === 0) {
        // Log every 30 inferences
        const avgInferenceTime = totalInferenceTime / 30;
        const actualFPS = 1000 / (avgInferenceTime + dynamicInterval);
        console.log(
          `Offer side - Average inference time: ${avgInferenceTime.toFixed(
            1
          )}ms, Actual FPS: ${actualFPS.toFixed(1)}`
        );

        // Adaptive interval adjustment for WebGPU
        if (avgInferenceTime < 20) {
          dynamicInterval = Math.max(16, dynamicInterval - 2); // Up to 60 FPS
        } else if (avgInferenceTime < 40) {
          dynamicInterval = Math.max(33, dynamicInterval - 1); // Up to 30 FPS
        } else if (avgInferenceTime > 80) {
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

const inputAutorunInfo = {
  type: "inputAutorunInfo",
  inputSteer: 0,
  inputEngineCycle: 0,
  inputGear: 1,
  inputShuttle: 0,
  inputSpeed: 3,
  inputPtoHeight: 100,
  inputPtoOn: 0,
  inputHorn: 0,
  isRemoteCont: true,
  isAutoRunStart: 0,
  isUseSafetySensorInTeleDrive: 0,
};

const outputAutorunInfo = {
  type: "outputAutorunInfo",
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

const virtualInputInfo = {
  outputLat: 0,
  outputLon: 0,
  outputHeading: 0,
  outputVelocity: 0,
  outputCalcAutoSteer: 0,
  outputSteer: 0,
  inputVelocity: 0,
  inputSteering: 0,
  inputShuttle: 0,
  inputRemoteCont: 0,
  start: false,
};

window.addEventListener("gamepadconnected", (event) => {
  onGamepadConnected(event);
  requestAnimationFrame(updateAutorunLoop);
});
window.addEventListener("gamepaddisconnected", onGamepadDisconnected);

function updateAutorunLoop() {
  const gamepads = navigator.getGamepads();
  const usingGamepad = gamepads[0];
  if (usingGamepad) {
    gamepadToAutorunInfoCon(inputAutorunInfo, usingGamepad);
    drawGamepadInfo(inputAutorunInfo);
  }
  requestAnimationFrame(updateAutorunLoop);
}

signalingWebSocket.onmessage = async (event) => {
  const { type, payload } = JSON.parse(event.data);

  if (type === "answer") {
    await peerConnection.setRemoteDescription(
      new RTCSessionDescription({ type: "answer", sdp: payload.sdp })
    );
  } else if (type === "ice") {
    await peerConnection.addIceCandidate(
      new RTCIceCandidate(payload.candidate)
    );
  }
};

signalingWebSocket.onopen = () => {
  signalingWebSocket.send(
    JSON.stringify({ type: "register-offer", payload: { id: "offer" } })
  );
};

/* エラーの原因②：接続が確立する前にsend()を呼び出している
virtualWebSocket.onopen = () => {
  virtualWebSocket.send(
    JSON.stringify({
      type: "from-offer-init",
      payload: inputAutorunInfo,
    })
  );
};
*/

async function startConnection() {
  peerConnection = new RTCPeerConnection({
    iceServers: [{ urls: "stun:10.100.0.35:3478" }],
  });
  console.log("PeerConnection is created!");

  movieChannel = peerConnection.addTransceiver("video", {
    direction: "recvonly",
  });

  dataChannel = peerConnection.createDataChannel("chat");
  dataChannel.onopen = () => {
    console.log("DataChannel is open");
    setInterval(() => {
      dataChannel.send(JSON.stringify(inputAutorunInfo));
    }, 33);
  };
  dataChannel.onmessage = (event) => {
    const fromAnswerWebRtcData = JSON.parse(event.data);
    if (fromAnswerWebRtcData.type === "outputAutorunInfo") {
      outputAutorunInfo.lat = fromAnswerWebRtcData.payload.lat;
      outputAutorunInfo.lon = fromAnswerWebRtcData.payload.lon;
      outputAutorunInfo.gnssQuality = fromAnswerWebRtcData.payload.gnssQuality;
      outputAutorunInfo.gnssSpeed = fromAnswerWebRtcData.payload.gnssSpeed;
      outputAutorunInfo.heading = fromAnswerWebRtcData.payload.heading;
      outputAutorunInfo.headingError =
        fromAnswerWebRtcData.payload.headingError;
      outputAutorunInfo.lateralError =
        fromAnswerWebRtcData.payload.lateralError;
      outputAutorunInfo.steerAngle = fromAnswerWebRtcData.payload.steerAngle;
      outputAutorunInfo.realSteerAngle =
        fromAnswerWebRtcData.payload.realSteerAngle;
      outputAutorunInfo.stopStatus = fromAnswerWebRtcData.payload.stopStatus;
    } else if (fromAnswerWebRtcData.type === "inferenceResults") {
      console.log(
        "Received inference results from answer side:",
        fromAnswerWebRtcData.payload
      );
    }
  };

  const remoteStream = new MediaStream();
  peerConnection.ontrack = (event) => {
    remoteStream.addTrack(event.track);
    videoElement.srcObject = remoteStream;

    videoElement.addEventListener("loadeddata", async () => {
      await initONNXModel();
      startVideoInference(videoElement);
      
      const inferenceOnRadio = document.getElementById("inference-on");
      const inferenceOffRadio = document.getElementById("inference-off");
      
      inferenceOnRadio.addEventListener("change", () => {
        if (inferenceOnRadio.checked) {
          isInferenceEnabled = true;
          console.log("Offer side inference enabled");
        }
      });
      
      inferenceOffRadio.addEventListener("change", () => {
        if (inferenceOffRadio.checked) {
          isInferenceEnabled = false;
          currentDetections = [];
          console.log("Offer side inference disabled");
        }
      });
    });
  };

  peerConnection.onicecandidate = (event) => {
    if (event.candidate) {
      signalingWebSocket.send(
        JSON.stringify({
          type: "ice-offer",
          payload: { candidate: event.candidate },
        })
      );
    }
  };

  const offer = await peerConnection.createOffer();
  offer.sdp = prioritizeSelectedVideoCodec(offer.sdp);

  await peerConnection.setLocalDescription(offer);
  signalingWebSocket.send(
    JSON.stringify({ type: "offer", payload: { sdp: offer.sdp } })
  );
}

document
  .getElementById("send-sdp-by-ws")
  .addEventListener("click", startConnection);

// ===== ここから遅延ログ機能 (詳細版) =====
const delayLogs = [];
// 前回の値記録用
let prevFramesDecoded = null, prevFramesDecodedTime = null;
let prevFramesReceived = null, prevFramesReceivedTime = null;

setInterval(async function debugRTCStats() {
  if (!peerConnection) return;
  const stats = await peerConnection.getStats();

  const now = new Date();
  const pad = n => n.toString().padStart(2, "0");
  const timestamp = `${pad(now.getHours())}:${pad(now.getMinutes())}:${pad(now.getSeconds())}`;
  
  let logRow = {
    timestamp,
    "[jitter]": "",
    "[packetsLost]": "",
    "[jitterBufferDelay]": "",
    "[jitterBufferEmittedCount]": "",
    "[jitterBufferDelay/jitterBufferEmittedCount_in_ms]": "",
    "[totalDecodeTime/framesDecoded_in_ms]": "",
    "[framesReceived/s]": "",
    "[interFrameDelayStDev_in_ms]": "",
    "[totalInterFrameDelay/framesDecoded_in_ms]": "",
    "[totalAssemblyTime/framesAssembledFromMultiplePackets_in_ms]": "",
    "[totalProcessingDelay/jitterBufferEmittedCount_in_ms]": "",
    "[framesDecoded/s]": "",
    "[frameWidth]": "",
    "[frameHeight]": "",
    "[framesPerSecond]": ""
  };

  let inboundRtpReport;
  stats.forEach((report) => {
    if (report.type === "inbound-rtp" && report.kind === "video" && !report.isRemote) {
      inboundRtpReport = report;
    }
  });

  if (inboundRtpReport) {
    const ts = inboundRtpReport.timestamp;
    logRow["[jitter]"] = inboundRtpReport.jitter ?? "";
    logRow["[packetsLost]"] = inboundRtpReport.packetsLost ?? "";
    logRow["[jitterBufferDelay]"] = inboundRtpReport.jitterBufferDelay ?? "";
    logRow["[jitterBufferEmittedCount]"] = inboundRtpReport.jitterBufferEmittedCount ?? "";

    if (inboundRtpReport.jitterBufferDelay && inboundRtpReport.jitterBufferEmittedCount > 0) {
        logRow["[jitterBufferDelay/jitterBufferEmittedCount_in_ms]"] = (
            (inboundRtpReport.jitterBufferDelay / inboundRtpReport.jitterBufferEmittedCount) * 1000
        ).toFixed(3);
    }
    if (inboundRtpReport.totalInterFrameDelay && inboundRtpReport.framesDecoded > 0) {
        logRow["[totalInterFrameDelay/framesDecoded_in_ms]"] = (
            (inboundRtpReport.totalInterFrameDelay / inboundRtpReport.framesDecoded) * 1000
        ).toFixed(3);
    }
    if (inboundRtpReport.totalSquaredInterFrameDelay && inboundRtpReport.framesDecoded > 1) {
        const mean = inboundRtpReport.totalInterFrameDelay / inboundRtpReport.framesDecoded;
        const variance = (inboundRtpReport.totalSquaredInterFrameDelay / inboundRtpReport.framesDecoded) - (mean ** 2);
        logRow["[interFrameDelayStDev_in_ms]"] = (Math.sqrt(variance) * 1000).toFixed(3);
    }
    if (inboundRtpReport.totalProcessingDelay && inboundRtpReport.jitterBufferEmittedCount > 0) {
        logRow["[totalProcessingDelay/jitterBufferEmittedCount_in_ms]"] = (
            (inboundRtpReport.totalProcessingDelay / inboundRtpReport.jitterBufferEmittedCount) * 1000
        ).toFixed(3);
    }
    logRow["[frameWidth]"] = inboundRtpReport.frameWidth ?? "";
    logRow["[frameHeight]"] = inboundRtpReport.frameHeight ?? "";
    logRow["[framesPerSecond]"] = inboundRtpReport.framesPerSecond ?? "";

    if (inboundRtpReport.totalDecodeTime !== undefined && inboundRtpReport.framesDecoded > 0) {
        logRow["[totalDecodeTime/framesDecoded_in_ms]"] = (
            (inboundRtpReport.totalDecodeTime / inboundRtpReport.framesDecoded) * 1000
        ).toFixed(3);
    }
    if (inboundRtpReport.totalAssemblyTime !== undefined && inboundRtpReport.framesAssembledFromMultiplePackets > 0) {
        logRow["[totalAssemblyTime/framesAssembledFromMultiplePackets_in_ms]"] = (
            (inboundRtpReport.totalAssemblyTime / inboundRtpReport.framesAssembledFromMultiplePackets) * 1000
        ).toFixed(3);
    }
    if (inboundRtpReport.framesReceived !== undefined) {
        if (prevFramesReceived !== null && prevFramesReceivedTime !== null) {
            const deltaTimeSec = (ts - prevFramesReceivedTime) / 1000;
            const deltaFrames = inboundRtpReport.framesReceived - prevFramesReceived;
            if (deltaTimeSec > 0) {
                logRow["[framesReceived/s]"] = (deltaFrames / deltaTimeSec).toFixed(3);
            }
        }
        prevFramesReceived = inboundRtpReport.framesReceived;
        prevFramesReceivedTime = ts;
    }
    if (inboundRtpReport.framesDecoded !== undefined) {
        if (prevFramesDecoded !== null && prevFramesDecodedTime !== null) {
            const deltaTimeSec = (ts - prevFramesDecodedTime) / 1000;
            const deltaFrames = inboundRtpReport.framesDecoded - prevFramesDecoded;
            if (deltaTimeSec > 0) {
                logRow["[framesDecoded/s]"] = (deltaFrames / deltaTimeSec).toFixed(3);
            }
        }
        prevFramesDecoded = inboundRtpReport.framesDecoded;
        prevFramesDecodedTime = ts;
    }
  }

  delayLogs.push(logRow);
}, 1000);

function saveDelayLogsAsCSV() {
  if (delayLogs.length === 0) return;
  const headers = Object.keys(delayLogs[0]);
  const csv = [
    headers.join(","),
    ...delayLogs.map(row => headers.map(h => row[h] ?? "").join(","))
  ].join("\n");

  const blob = new Blob([csv], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;

  const now = new Date();
  const jst = new Date(now.getTime() + 9 * 60 * 60 * 1000); // 日本時間 (JST) に変換
  const pad = n => n.toString().padStart(2, "0");
  const ts = `${jst.getUTCFullYear()}-${pad(jst.getUTCMonth() + 1)}-${pad(jst.getUTCDate())}_${pad(jst.getUTCHours())}-${pad(jst.getUTCMinutes())}-${pad(jst.getUTCSeconds())}`;

  a.download = `offer_webrtc_delay_log_${ts}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

document.getElementById("save-delay-log").addEventListener("click", saveDelayLogsAsCSV);
// ===== 遅延ログ機能ここまで =====

/**
 * @name setVideoQuality
 * @description HTMLの入力値を取得し、DataChannel経由でAnswerに映像品質の変更を要求します。
 * @param {RTCDataChannel} dataChannel - 設定を送信するためのデータチャネル
 */
function setVideoQuality(dataChannel) {
  const bitrate = document.getElementById("set-video-bitrate").value;
  const height = document.getElementById("set-video-height").value;
  const width = document.getElementById("set-video-width").value;
  const framerate = document.getElementById("set-video-framerate").value;

  if (dataChannel && dataChannel.readyState === "open") {
    const qualitySettings = {
      type: "videoQualityChange",
      payload: {
        bitrate: parseInt(bitrate, 10),
        height: parseInt(height, 10),
        width: parseInt(width, 10),
        framerate: parseInt(framerate, 10),
      }
    };
    dataChannel.send(JSON.stringify(qualitySettings));
    console.log("Sent video quality settings:", qualitySettings);
  } else {
    console.error("DataChannel is not open. Cannot send video quality settings.");
  }
}

// SetVideoQualityボタンのイベントリスナー
document.getElementById("set-video-quality").addEventListener("click", () => {
  setVideoQuality(dataChannel);
});

/* エラーの原因③：接続が確立する前にsend()を呼び出している
setInterval(() => {
  virtualInputInfo.inputSteering = inputAutorunInfo.inputSteer;
  virtualInputInfo.inputVelocity = inputAutorunInfo.inputSpeed;
  virtualInputInfo.inputShuttle = inputAutorunInfo.inputShuttle;
  virtualInputInfo.inputRemoteCont = inputAutorunInfo.isRemoteCont;
  virtualInputInfo.outputLat = outputAutorunInfo.lat;
  virtualInputInfo.outputLon = outputAutorunInfo.lon;
  virtualInputInfo.outputHeading = outputAutorunInfo.heading;
  virtualInputInfo.outputCalcAutoSteer = outputAutorunInfo.steerAngle;
  virtualInputInfo.outputSteer = outputAutorunInfo.realSteerAngle;
  virtualInputInfo.outputVelocity = outputAutorunInfo.gnssSpeed;
  virtualInputInfo.start = true;
  if (virtualWebSocket != null) {
    virtualWebSocket.send(
      JSON.stringify({
        type: "to-virtual-inputdata",
        payload: { virtualInputInfo },
      })
    );
  }
}, 33);
*/

// Detection info object for video-canvas rendering
const detectionInfo = {
  get detections() { return currentDetections; },
  get isEnabled() { return isInferenceEnabled; }
};

startCanvasRendering(
  canvasSonoki,
  videoElement,
  outputAutorunInfo,
  inputAutorunInfo,
  detectionInfo
);
