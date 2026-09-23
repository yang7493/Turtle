import {
  PoseLandmarker,
  FilesetResolver,
  DrawingUtils,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";

const LM = { LEFT_EAR: 7, RIGHT_EAR: 8, LEFT_SHOULDER: 11, RIGHT_SHOULDER: 12 };
const ENTER_MS = 1000;
const EXIT_MS = 500;
const CALIBRATION_FRAMES = 30;

const video = document.getElementById("video");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const startBtn = document.getElementById("startBtn");
const calibrateBtn = document.getElementById("calibrateBtn");
const sensitivityInput = document.getElementById("sensitivity");
const statusDot = document.getElementById("statusDot");
const statusLabel = document.getElementById("statusLabel");
const todayCountEl = document.getElementById("todayCount");
const todaySecondsEl = document.getElementById("todaySeconds");
const resetBtn = document.getElementById("resetBtn");

let poseLandmarker;
let drawingUtils;
let lastVideoTime = -1;
let lastFrameTs = 0;

let smoothedRatio = null;
let baseline = null;
let calibrationSamples = null;

let isTurtle = false;
let badSince = null;
let goodSince = null;

let stats = loadStats();
renderStats();

function todayKey() {
  const d = new Date();
  const pad = (n) => String(n).padStart(2, "0");
  return `turtle-${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
}

function loadStats() {
  try {
    const raw = localStorage.getItem(todayKey());
    if (raw) return { key: todayKey(), ...JSON.parse(raw) };
  } catch {}
  return { key: todayKey(), count: 0, seconds: 0 };
}

function saveStats() {
  try {
    localStorage.setItem(stats.key, JSON.stringify({ count: stats.count, seconds: stats.seconds }));
  } catch {}
}

function renderStats() {
  todayCountEl.textContent = stats.count;
  todaySecondsEl.textContent = Math.floor(stats.seconds);
}

function setStatus(label, kind) {
  statusLabel.textContent = label;
  statusDot.className = "dot" + (kind ? ` ${kind}` : "");
}

async function init() {
  startBtn.disabled = true;
  setStatus("모델 불러오는 중...");
  const vision = await FilesetResolver.forVisionTasks(WASM_URL);
  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
    runningMode: "VIDEO",
    numPoses: 1,
  });
  drawingUtils = new DrawingUtils(ctx);

  setStatus("카메라 켜는 중...");
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480 },
    audio: false,
  });
  video.srcObject = stream;
  await new Promise((r) => (video.onloadeddata = r));
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  startBtn.textContent = "실행 중";
  calibrateBtn.disabled = false;
  setStatus("바른 자세로 앉아서 '보정하기'를 눌러주세요");
  requestAnimationFrame(loop);
}

// 어깨선 위로 귀가 얼마나 올라와 있는지를 어깨 너비로 나눈 값. 목이 앞으로 빠지면 작아진다.
function computeRatio(lm) {
  const visible = (p) => p && (p.visibility ?? 1) > 0.5;
  const ls = lm[LM.LEFT_SHOULDER];
  const rs = lm[LM.RIGHT_SHOULDER];
  if (!visible(ls) || !visible(rs)) return null;

  const ears = [lm[LM.LEFT_EAR], lm[LM.RIGHT_EAR]].filter(visible);
  if (ears.length === 0) return null;

  const earY = ears.reduce((s, p) => s + p.y, 0) / ears.length;
  const earX = ears.reduce((s, p) => s + p.x, 0) / ears.length;
  const shoulderY = (ls.y + rs.y) / 2;
  const shoulderX = (ls.x + rs.x) / 2;
  const shoulderWidth = Math.hypot(ls.x - rs.x, ls.y - rs.y);
  if (shoulderWidth < 0.05) return null;

  return {
    ratio: (shoulderY - earY) / shoulderWidth,
    ear: { x: earX, y: earY },
    shoulder: { x: shoulderX, y: shoulderY },
  };
}

function loop() {
  const now = performance.now();
  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const result = poseLandmarker.detectForVideo(video, now);
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const lm = result.landmarks?.[0];
    if (lm) {
      drawingUtils.drawConnectors(lm, PoseLandmarker.POSE_CONNECTIONS, { color: "#5eead4", lineWidth: 2 });
      drawingUtils.drawLandmarks(lm, { color: "#e8ecf1", radius: 2 });
      const m = computeRatio(lm);
      if (m) {
        drawNeckLine(m);
        handleMeasurement(m.ratio, now);
      } else {
        setStatus("귀와 어깨가 화면에 보이게 앉아주세요");
      }
    } else {
      setStatus("사람이 보이지 않아요");
    }
  }

  if (isTurtle && lastFrameTs) {
    stats.seconds += (now - lastFrameTs) / 1000;
    renderStats();
  }
  lastFrameTs = now;
  requestAnimationFrame(loop);
}

function drawNeckLine({ ear, shoulder }) {
  ctx.strokeStyle = isTurtle ? "#ef4444" : "#facc15";
  ctx.lineWidth = 4;
  ctx.beginPath();
  ctx.moveTo(ear.x * canvas.width, ear.y * canvas.height);
  ctx.lineTo(shoulder.x * canvas.width, shoulder.y * canvas.height);
  ctx.stroke();
}

function handleMeasurement(ratio, now) {
  smoothedRatio = smoothedRatio == null ? ratio : smoothedRatio * 0.8 + ratio * 0.2;

  if (calibrationSamples) {
    calibrationSamples.push(ratio);
    setStatus(`보정 중... ${calibrationSamples.length}/${CALIBRATION_FRAMES}`);
    if (calibrationSamples.length >= CALIBRATION_FRAMES) {
      baseline = calibrationSamples.reduce((a, b) => a + b, 0) / calibrationSamples.length;
      calibrationSamples = null;
      calibrateBtn.disabled = false;
      calibrateBtn.textContent = "다시 보정하기";
    }
    return;
  }

  if (baseline == null) return;

  const threshold = baseline * (1 - Number(sensitivityInput.value) / 100);
  const bad = smoothedRatio < threshold;

  if (bad) {
    goodSince = null;
    badSince ??= now;
    if (!isTurtle && now - badSince >= ENTER_MS) enterTurtle();
  } else {
    badSince = null;
    goodSince ??= now;
    if (isTurtle && now - goodSince >= EXIT_MS) exitTurtle();
  }

  const score = Math.round((smoothedRatio / baseline) * 100);
  setStatus(isTurtle ? `거북목! (자세 점수 ${score})` : `바른 자세 (자세 점수 ${score})`, isTurtle ? "warn" : "ok");
}

function enterTurtle() {
  isTurtle = true;
  document.body.classList.add("turtle");
  rolloverIfNewDay();
  stats.count += 1;
  saveStats();
  renderStats();
}

function exitTurtle() {
  isTurtle = false;
  document.body.classList.remove("turtle");
  saveStats();
}

function rolloverIfNewDay() {
  if (stats.key !== todayKey()) stats = { key: todayKey(), count: 0, seconds: 0 };
}

startBtn.addEventListener("click", () => {
  init().catch((err) => {
    console.error(err);
    setStatus(`시작 실패: ${err.message}`, "warn");
    startBtn.disabled = false;
  });
});

calibrateBtn.addEventListener("click", () => {
  calibrationSamples = [];
  calibrateBtn.disabled = true;
  if (isTurtle) exitTurtle();
});

resetBtn.addEventListener("click", () => {
  stats = { key: todayKey(), count: 0, seconds: 0 };
  saveStats();
  renderStats();
});

setInterval(saveStats, 5000);
