/**
 * A circular buffer implementation.
 */
class RingBuffer {
  /**
   * @param {number} length The size of the buffer in samples.
   */
  constructor(length) {
    this.head = 0;
    this.tail = 0;
    this.framesAvailable = 0;

    this.length = length;
    this.buffer = new Float32Array(length);
  }

  /**
   * Resets the buffer.
   */
  reset() {
    this.head = 0;
    this.tail = 0;
    this.framesAvailable = 0;
  }

  /**
   * Pushes an array of samples into the buffer.
   * @param {!Float32Array} array The array of samples to push.
   */
  push(array) {
    const sourceLength = array.length;
    for (let i = 0; i < sourceLength; ++i) {
      const writeIndex = (this.tail + i) % this.length;
      this.buffer[writeIndex] = array[i];
    }

    this.tail = (this.tail + sourceLength) % this.length;

    this.framesAvailable += sourceLength;
    if (this.framesAvailable > this.length) {
      this.framesAvailable = this.length;
      this.head = this.tail;
    }
  }

  /**
   * Pops an array of samples from the buffer.
   * @param {!Float32Array} array The array to fill with samples.
   */
  pop(array) {
    if (this.framesAvailable === 0) {
      return;
    }

    const destinationLength = array.length;

    for (let i = 0; i < destinationLength; ++i) {
      const readIndex = (this.head + i) % this.length;
      array[i] = this.buffer[readIndex];
    }

    this.head = (this.head + destinationLength) % this.length;

    this.framesAvailable -= destinationLength;
    if (this.framesAvailable < 0) {
      this.framesAvailable = 0;
    }
  }
}

/**
 * An AudioWorkletProcessor that uses a RingBuffer to handle audio data.
 * It communicates with an external process via messages.
 */
class ExternalRingBufferWorkletProcessor extends AudioWorkletProcessor {
  /**
   * @param {!AudioWorkletNodeOptions} options The options for the processor.
   */
  constructor(options) {
    super();
    this.kernelBufferSize = options.processorOptions.kernelBufferSize;
    this.channelCount = options.outputChannelCount;

    // Input buffer is twice as large as the kernel buffer size to make sure we
    // don't overwrite data that has not been read yet, which can happen if the
    // kernel buffer size is not a multiple of the DSP block size.
    this.inputRingBuffer = new RingBuffer(2 * this.kernelBufferSize);

    let outputRingLength = this.inputRingBuffer.length;
    let extraBuffer = options.processorOptions.additionalBufferedSamples;
    if (extraBuffer !== undefined) {
      outputRingLength += extraBuffer;
    }

    // Output buffer has the same size as the input buffer plus the additional
    // buffered samples. Since audio channels are interleaved, the output buffer
    // size is multiplied by the number of channels.
    this.outputRingBuffer =
        new RingBuffer(outputRingLength * this.channelCount);
    this.outputRingBuffer.framesAvailable = this.outputRingBuffer.length;

    this.externalOut = [];

    this.port.onmessage = (e) => {
      this.onmessage(e.data);
    };
  }

  /**
   * Handles messages from the main thread.
   * @param {!Object} data The message data.
   */
  onmessage(data) {
    switch (data.type) {
      case 'buffer':
        this.externalOut.push(data.value);
        break;
      case 'reset':
        console.log('ring buffer reset');
        this.inputRingBuffer.reset();
        this.outputRingBuffer.reset();
        break;
      default:
        console.error('Unknown message type: ' + data.type);
        break;
    }
  }

  /**
   * Processes audio data.
   * @param {!Array<!Array<!Float32Array>>} inputs The input audio data.
   * @param {!Array<!Array<!Float32Array>>} outputs The output audio data.
   * @param {!Object} parameters The parameters for the processor.
   * @return {boolean} True if the processor should keep running.
   */
  process(inputs, outputs, parameters) {
    if (inputs[0].length == 0) {
      return true;
    }

    const input = inputs[0][0];  // source 0, channel 0

    this.inputRingBuffer.push(input);

    // get whatever was made available by the external process
    for (let i = 0; i < this.externalOut.length; i++) {
      this.outputRingBuffer.push(this.externalOut.shift());
    }

    if (this.inputRingBuffer.framesAvailable >= this.kernelBufferSize) {
      // pop from input buffer
      let inputBuffer = new Float32Array(this.kernelBufferSize);
      this.inputRingBuffer.pop(inputBuffer);
      this.port.postMessage({
        type: 'buffer',
        value: inputBuffer,
        framesAvailable: this.inputRingBuffer.framesAvailable,
        processorBufferSize: input.length,
      });
    }

    let outputBuffer =
        new Float32Array(outputs[0][0].length * this.channelCount);

    if (this.outputRingBuffer.framesAvailable >= outputBuffer.length) {
      this.outputRingBuffer.pop(outputBuffer);
    } else {
      this.outputRingBuffer.reset();
    }

    for (let i = 0; i < outputs[0][0].length; i++) {
      for (let c = 0; c < this.channelCount; c++) {
        outputs[0][c][i] = outputBuffer[i * this.channelCount + c];
      }
    }

    return true;
  }
}

registerProcessor(
    'external-ring-buffer-processor',
    ExternalRingBufferWorkletProcessor,
);
