import Foundation
import MLX
import MLXLLM
import MLXNN

/// MoshiLLM handles loading the Moshi multi-stream audio model and decoding
/// its Mimi codec outputs into PCM frames suitable for playback.
final class MoshiLLM: ObservableObject {
    private var generator: Generator?
    private let modelQueue = DispatchQueue(label: "buddy.model.queue")
    private let mimiCodec = MimiNeuralAudioCodec(codebookCount: 8, samplesPerFrame: 1920, sampleRate: 24000)

    /// Hugging Face repository containing converted Moshi .mlx and .gguf weights.
    /// Download the files locally (e.g., Models/moshi.mlx) and point `prepare(modelPath:)` at them.
    let huggingFaceRepositoryURL = URL(string: "https://huggingface.co/fixie-ai/moshi-mlx")
    let huggingFaceGGUFRepositoryURL = URL(string: "https://huggingface.co/fixie-ai/moshi-gguf")

    func prepare(modelPath: String = "Models/moshi.mlx") {
        modelQueue.async { [weak self] in
            guard let self else { return }
            self.generator = self.loadModel(at: modelPath)
        }
    }

    private func loadModel(at path: String) -> Generator? {
        do {
            let llm = try LLM.load(from: path)
            return llm.streamingGenerator()
        } catch {
            print("Failed to load MLX model: \(error)")
            return nil
        }
    }

    /// Streams PCM frames decoded from Moshi's eight codebook audio tokens.
    /// - Parameter seedTokens: Optional initial tokens for priming the model, provided
    ///   as a flattened list of codebook indices (multiples of eight).
    func streamingDecodeToPCM(seedTokens: [Int] = []) -> AsyncStream<[Float]> {
        AsyncStream { continuation in
            modelQueue.async { [weak self] in
                guard
                    let self,
                    let generator,
                    let context = try? generator.makeContext()
                else {
                    continuation.finish()
                    return
                }

                generator.reset()
                var input = MLXArray(seedTokens.map(Float.init))

                while true {
                    do {
                        let output = try generator.generate(token: input, context: context)
                        guard let tokenValues: [Int32] = output.toArray(type: Int32.self) else {
                            continuation.finish()
                            break
                        }

                        let pcmFrames = self.decodeToPCMFrames(tokens: tokenValues.map(Int.init))
                        if pcmFrames.isEmpty {
                            continuation.finish()
                            break
                        }

                        for frame in pcmFrames {
                            continuation.yield(frame)
                        }

                        // Feed the last generated codebook set back for continued generation.
                        let nextInput = tokenValues.suffix(self.mimiCodec.codebookCount).map(Float.init)
                        input = MLXArray(nextInput)
                    } catch {
                        print("Streaming generation stopped: \(error)")
                        continuation.finish()
                        break
                    }
                }
            }
        }
    }

    private func decodeToPCMFrames(tokens: [Int]) -> [[Float]] {
        let stride = mimiCodec.codebookCount
        guard !tokens.isEmpty, tokens.count % stride == 0 else { return [] }

        var frames: [[Float]] = []
        var index = 0
        while index < tokens.count {
            let frameTokens = Array(tokens[index..<(index + stride)])
            frames.append(mimiCodec.decodeFrame(codebookTokens: frameTokens))
            index += stride
        }
        return frames
    }
}

/// Lightweight Mimi Neural Audio Codec placeholder that deterministically converts
/// Moshi's eight codebook tokens into PCM samples. Replace the lookup logic with the
/// official Mimi decoder tables when bundling real weights.
private final class MimiNeuralAudioCodec {
    let codebookCount: Int
    let samplesPerFrame: Int
    let sampleRate: Int

    init(codebookCount: Int, samplesPerFrame: Int, sampleRate: Int) {
        self.codebookCount = codebookCount
        self.samplesPerFrame = samplesPerFrame
        self.sampleRate = sampleRate
    }

    func decodeFrame(codebookTokens: [Int]) -> [Float] {
        guard codebookTokens.count == codebookCount else { return [] }

        // Simple deterministic synthesis: map codebook indices into smoothly varying
        // PCM samples. For production, swap this out for Mimi's learned codebook
        // embeddings and dequantization kernels.
        var pcm = [Float](repeating: 0, count: samplesPerFrame)
        for sampleIndex in 0..<samplesPerFrame {
            let token = codebookTokens[sampleIndex % codebookCount]
            let phase = Float((sampleIndex * (token + 1)) % 512) / 512.0
            pcm[sampleIndex] = sinf(2 * .pi * phase) * 0.2
        }
        return pcm
    }
}
