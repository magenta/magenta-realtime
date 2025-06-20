/**
 * Creates an external ring buffer node.
 *
 * This function creates an AudioWorkletNode that acts as a ring buffer. It
 * allows to buffer audio samples and to retrieve them in a continuous way.
 *
 * @param {!AudioContext} audioContext The audio context.
 * @param {number=} bufferSize The buffer size of the node.
 * @param {number=} channelCount The number of channels.
 * @param {number=} additionalBufferedSamples The number of additional samples
 *     to buffer.
 * @param {string=} ringBufferPath The path to the ring buffer worklet.
 * @return {!Promise<!AudioWorkletNode>} The ring buffer node.
 */
function createExternalRingBufferNode(
    audioContext, bufferSize = 512, channelCount = 1,
    additionalBufferedSamples = 0, ringBufferPath = 'ring_buffer.js') {
  return new Promise(
      (resolve) => {
        audioContext.audioWorklet.addModule(ringBufferPath).then(() => {
          resolve(
              new AudioWorkletNode(
                  audioContext, 'external-ring-buffer-processor', {
                    outputChannelCount: [channelCount],
                    processorOptions: {
                      kernelBufferSize: bufferSize,
                      additionalBufferedSamples: additionalBufferedSamples,
                    },
                  }),
          );
        });
      });
}
