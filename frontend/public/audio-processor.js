const CHUNK_SIZE = 4096;

class ChunkerProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = new Float32Array(CHUNK_SIZE);
    this.writeIndex = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (input && input.length > 0) {
      const channelData = input[0];
      for (let i = 0; i < channelData.length; i++) {
        this.buffer[this.writeIndex++] = channelData[i]; //accumulate audio samples until CHUNK_SIZE
        if (this.writeIndex === CHUNK_SIZE) { //send to backend once CHUNK_SIZE is reached
          const chunkToSend = this.buffer.slice(0);
          this.port.postMessage(chunkToSend, [chunkToSend.buffer]);
          this.writeIndex = 0;
        }
      }
    }
    return true;
  }
}

registerProcessor('chunker-processor', ChunkerProcessor);
