const WebSocket = require("ws");
const wsServer = new WebSocket.Server({ port: 9090 });
let clientBool = false;
let fromOfferSocket = null;
let fromVirtualSocket = null;

const net = require("net");
const {
  parseMR1000AMessage,
  makeMessageMR1000A,
} = require("./node-function.js");
const { latLonToUTM54 } = require("./positioningLib.js");
let vehicleSelecter;
const clients = new Set();

const inputAutorunInfo = {
  inputSteer: 0,
  inputEngineCycle: 0,
  inputGear: 1,
  inputShuttle: 0,
  inputSpeed: 0,
  inputPtoHeight: 0,
  inputPtoOn: 0,
  inputHorn: 0,
  isRemoteCont: 0,
  isAutoRunStart: 0,
  isUseSafetySensorInTeleDrive: 0,
};

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

const virtualInputInfo = {
  outputEasting: 0,
  outputNorthing: 0,
  outputHeading: 0,
  outputVelocity: 0,
  outputSteer: 0,
  outputCalcAutoSteer: 0,
  inputVelocity: 0,
  inputSteering: 0,
  inputShuttle: 0,
  inputRemoteCont: 0,
  start: false,
};

console.log("WebSocket Server is running on ws://localhost:8080");

wsServer.on("connection", (webSocket) => {
  webSocket.on("message", (wsMessage) => {
    // console.log("Received message:", message);
    try {
      // WebSocketからのメッセージを文字列として解析
      const wsData = JSON.parse(wsMessage);
      //console.log("Parsed data:", data);

      // typeプロパティに基づいてソケットを分類
      if (wsData.type === "from-offer-init") {
        fromOfferSocket = webSocket;
        console.log("Offer connected");
      } else if (wsData.type === "virtual-tractor") {
        fromVirtualSocket = webSocket;
        console.log("Virtual tractor connected");
      } else if (wsData.type === "to-virtual-inputdata") {
        console.log("Received data from offer:" + wsData);
        const wsMessageFromOfferInput = wsData;
        console.log("Received message from offer:", wsMessageFromOfferInput);
        virtualInputInfo.outputHeading = Number(
          wsMessageFromOfferInput.payload.outputHeading
        );
        virtualInputInfo.outputVelocity = Number(
          wsMessageFromOfferInput.payload.virtualInputInfo.outputVelocity
        );
        virtualInputInfo.inputVelocity = Number(
          wsMessageFromOfferInput.payload.virtualInputInfo.inputVelocity
        );
        virtualInputInfo.inputSteering = Number(
          wsMessageFromOfferInput.payload.virtualInputInfo.inputSteering
        );
        virtualInputInfo.outputSteer = Number(
          wsMessageFromOfferInput.payload.virtualInputInfo.outputSteer
        );
        virtualInputInfo.outputCalcAutoSteer = Number(
          wsMessageFromOfferInput.payload.virtualInputInfo.outputCalcAutoSteer
        );
        virtualInputInfo.inputShuttle = Number(
          wsMessageFromOfferInput.payload.virtualInputInfo.inputShuttle
        );
        virtualInputInfo.inputRemoteCont = Boolean(
          wsMessageFromOfferInput.payload.virtualInputInfo.inputRemoteCont
        );
        virtualInputInfo.start = Boolean(
          wsMessageFromOfferInput.payload.virtualInputInfo.start
        );
        virtualInputInfo.outputHeading = Number(
          wsMessageFromOfferInput.payload.virtualInputInfo.outputHeading
        );
        console.log(virtualInputInfo);
        const utmPosi = latLonToUTM54(
          Number(wsMessageFromOfferInput.payload.virtualInputInfo.outputLat),
          Number(wsMessageFromOfferInput.payload.virtualInputInfo.outputLon)
        );
        virtualInputInfo.outputEasting = utmPosi.x;
        virtualInputInfo.outputNorthing = utmPosi.y;

        if (typeof wsData === "object" && wsData !== null) {
          Object.keys(inputAutorunInfo).forEach((key) => {
            if (wsData.hasOwnProperty(key)) {
              inputAutorunInfo[key] = wsData[key];
            }
          });
        }

        //console.log("Updated inputAutorunInfo:", inputAutorunInfo);
        //console.log(`Received message`);
        //console.log(data);
      } else {
        // typeが指定されていない場合の処理
        console.log("Received message without type:", wsData);
      }
    } catch (err) {
      console.error("Error parsing message:", err);
    }
  });

  webSocket.on("close", () => {
    // ソケットがクローズされたときの処理
    if (webSocket === fromOfferSocket) {
      console.log("Offer disconnected");
      fromOfferSocket = null;
    } else if (webSocket === fromVirtualSocket) {
      console.log("Virtual tractor disconnected");
      fromVirtualSocket = null;
    }
    console.log("Connection has closed");
  });
});

setInterval(() => {
  const virtualMessage = virtualInputInfo;
  // クライアントにメッセージを送信
  if (fromVirtualSocket !== null) {
    fromVirtualSocket.send(JSON.stringify(virtualMessage));
    //console.log("Sent message to virtual tractor:", virtualMessage);
  }
}, 33);
