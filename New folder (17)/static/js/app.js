const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const resultDiv = document.getElementById('result');
const startBtn = document.getElementById('start');
const stopBtn = document.getElementById('stop');

let recording = false;
let frames = [];
let recorderInterval;

navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => { video.srcObject = stream; })
  .catch(err => alert("Camera error: " + err));

function captureFrame() {
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL('image/jpeg');
}

startBtn.addEventListener("click", () => {
  frames = [];
  recording = true;
  resultDiv.innerText = "Recording...";
  recorderInterval = setInterval(() => {
    if (recording) {
      frames.push(captureFrame());
    }
  }, 100); // ~10 fps
});

stopBtn.addEventListener("click", async () => {
  recording = false;
  clearInterval(recorderInterval);
  resultDiv.innerText = "Predicting...";
  const res = await fetch("/predict_video", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({ frames })
  });
  const data = await res.json();
  if (data.success) {
    resultDiv.innerText = `Result: ${data.label} (${(data.confidence*100).toFixed(1)}%)`;
  } else {
    resultDiv.innerText = "No hand detected";
  }
});
