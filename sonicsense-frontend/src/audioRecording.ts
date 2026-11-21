const WAV_HEADER_BYTES = 44;
const PCM_BYTES_PER_SAMPLE = 2;

export function encodePcmToWav(
  chunks: Float32Array[],
  sampleRate: number,
  maxBytes: number,
): Blob {
  const frameCount = chunks.reduce((total, chunk) => total + chunk.length, 0);
  const dataBytes = frameCount * PCM_BYTES_PER_SAMPLE;
  if (!Number.isFinite(sampleRate) || sampleRate <= 0 || frameCount === 0) {
    throw new Error("No usable microphone audio was captured.");
  }
  if (WAV_HEADER_BYTES + dataBytes > maxBytes) {
    throw new Error("Recording exceeds the 25 MiB limit. Record a shorter clip.");
  }

  const wav = new ArrayBuffer(WAV_HEADER_BYTES + dataBytes);
  const view = new DataView(wav);
  const writeText = (value: string, offset: number) => {
    for (let index = 0; index < value.length; index += 1) {
      view.setUint8(offset + index, value.charCodeAt(index));
    }
  };

  writeText("RIFF", 0);
  view.setUint32(4, WAV_HEADER_BYTES - 8 + dataBytes, true);
  writeText("WAVE", 8);
  writeText("fmt ", 12);
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * PCM_BYTES_PER_SAMPLE, true);
  view.setUint16(32, PCM_BYTES_PER_SAMPLE, true);
  view.setUint16(34, 16, true);
  writeText("data", 36);
  view.setUint32(40, dataBytes, true);

  let offset = WAV_HEADER_BYTES;
  for (const chunk of chunks) {
    for (const inputSample of chunk) {
      const sample = Math.max(-1, Math.min(1, inputSample));
      view.setInt16(
        offset,
        sample < 0 ? Math.round(sample * 0x8000) : Math.round(sample * 0x7fff),
        true,
      );
      offset += PCM_BYTES_PER_SAMPLE;
    }
  }

  return new Blob([wav], { type: "audio/wav" });
}
