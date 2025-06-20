/**
 * @fileoverview This file contains the javascript code for the web streamer.
 * It handles the audio processing, the communication with the python backend,
 * and the UI.
 */

/**
 * Converts a buffer to a base64 string.
 * @param {!Buffer} buffer The buffer to convert.
 * @return {string} The base64 string.
 */
async function bufferToBase64(buffer) {
  const base64url = await new Promise((r) => {
    const reader = new FileReader();
    reader.onload = () => r(reader.result);
    reader.readAsDataURL(new Blob([buffer]));
  });
  return base64url.slice(37);
}

/**
 * Converts a base64 string to a buffer.
 * @param {string} binString The base64 string to convert.
 * @return {!Buffer} The buffer.
 */
function base64ToBuffer(binString) {
  binString = atob(binString);
  return Uint8Array.from(binString, (m) => m.codePointAt(0)).buffer;
}



/**
 * Handles the incoming buffers from the audio worklet.
 * It converts the incoming buffer to a base64 string, sends it to the python
 * backend, receives the processed buffer, converts it back to a Float32Array,
 * and sends it to the audio worklet.
 * It also updates the RTF indicator.
 * @param {!MessageEvent} event The message event.
 */
async function colabBufferCallback(event) {
  let startTime = window.audioContext.currentTime;

  if (window.rtf !== undefined) {
    window.rtfDiv.innerHTML = Math.floor(100 * window.rtf) + '%';
    window.rtfDiv.style.backgroundColor = window.rtf > 1 ? 'red' : 'green';
  }

  let data = event.data;
  if (data.type !== 'buffer') return;

  data.value = data.value.map((x) => x * (2 ** 15 - 1));
  data.value = Int16Array.from(data.value);

  let buffer = await bufferToBase64(data.value);

  let result = await google.colab.kernel.invokeFunction(
      'notebook.audioCallback',
      [buffer],
  );

  if (window.audioContext.state === 'suspended') {
    console.log('discarding output');
    return;
  }

  buffer =
      new Int16Array(base64ToBuffer(result.data['application/json'].audiob64));
  buffer = Float32Array.from(buffer).map((x) => x / (2 ** 15 - 1));

  window.ringBufferNode.port.postMessage({
    type: 'buffer',
    value: buffer,
  });

  let endTime = window.audioContext.currentTime;
  window.rtf = ((endTime - startTime) * window.sampleRate) / window.bufferSize;
}

/**
 * Resets the ring buffer.
 */
function resetRingBuffer() {
  if (window.ringBufferNode !== undefined) {
    window.ringBufferNode.port.postMessage({
      type: 'reset',
    });
  }
  window.audioContext.suspend();
  window.stopButton.disabled = true;
  window.startButton.disabled = false;
  window.rtfDiv.style.backgroundColor = 'rgb(240, 240, 240)';
  window.rtfDiv.innerHTML = '';
}

/**
 * Initializes the audio context.
 * @param {number=} sampleRate The sample rate of the audio context.
 * @param {number=} bufferSize The buffer size of the audio context.
 * @param {boolean=} enableInput Whether to enable the input stream.
 * @param {boolean=} rawInputAudio Whether to use raw input audio.
 * @param {boolean=} enableAutomaticGainControlOnInput Whether to enable
 *     automatic gain control on input.
 * @param {number=} channelCount The number of channels to use.
 * @param {number=} additionalBufferedSamples The number of additional samples
 *     to buffer.
 * @param {string=} ringBufferPath The path to the ring buffer worklet.
 */
async function audioContextInit(
    sampleRate = 48000,
    bufferSize = 512,
    enableInput = false,
    rawInputAudio = false,
    enableAutomaticGainControlOnInput = false,
    channelCount = 2,
    additionalBufferedSamples = 0,
    ringBufferPath = 'ring_buffer.js',
) {
  window.audioContext = new AudioContext({
    latencyHint: 'interactive',
    sampleRate: sampleRate,
  });
  window.bufferSize = bufferSize;
  window.sampleRate = sampleRate;
  window.parameterUpdated = -1;

  await window.audioContext.suspend();

  // create master gain node
  window.master = new GainNode(window.audioContext);
  window.master.connect(window.audioContext.destination);

  // create python node
  window.ringBufferNode = await createExternalRingBufferNode(
      window.audioContext,
      bufferSize,
      channelCount,
      additionalBufferedSamples,
      ringBufferPath,
  );
  window.ringBufferNode.port.onmessage = colabBufferCallback;
  window.ringBufferNode.connect(window.master);

  // create input node
  if (enableInput) {
    let stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: !rawInputAudio,
        noiseSuppression: !rawInputAudio,
        autoGainControl: enableAutomaticGainControlOnInput,
      },
    });
    window.microphone = window.audioContext.createMediaStreamSource(stream);
    window.microphone.connect(window.ringBufferNode);
  } else {
    window.oscillator = window.audioContext.createOscillator();
    window.oscillator.type = 'sine';
    window.oscillator.frequency.setValueAtTime(
        440, window.audioContext.currentTime);  // value in hertz
    window.oscillator.connect(window.ringBufferNode);
    window.oscillator.start();
  }

  // rtf indicator
  window.rtfDiv = document.createElement('div');
  window.rtfDiv.id = 'rtf-div';
  document.getElementById('buttons').append(window.rtfDiv);

  // start / stop buttons
  let startButton = document.getElementById('start_button');
  let stopButton = document.getElementById('stop_button');

  startButton.onclick = () => {
    google.colab.kernel.invokeFunction('notebook.startStreamingCallback', [])
        .then(() => {
          window.audioContext.resume();
          startButton.disabled = true;
          stopButton.disabled = false;
        });
  };
  startButton.disabled = false;

  stopButton.onclick = () => {
    window.audioContext.suspend();
    stopButton.disabled = true;
    startButton.disabled = false;
    window.rtfDiv.style.backgroundColor = 'rgb(240, 240, 240)';
    window.rtfDiv.innerHTML = '';
  };

  window.startButton = startButton;
  window.stopButton = stopButton;
}
