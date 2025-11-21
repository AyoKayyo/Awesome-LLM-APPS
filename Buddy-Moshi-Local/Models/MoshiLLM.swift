import Foundation
import MLX
import MLXLLM
import MLXNN

final class MoshiLLM: ObservableObject {
    private var generator: Generator?
    private let modelQueue = DispatchQueue(label: "buddy.model.queue")

    func prepare(modelPath: String = "Models/model.mlx") {
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

    func streamingGenerate(from audioTokens: [Float]) -> AsyncStream<Float> {
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

                var input = MLXArray(audioTokens)
                generator.reset()

                while true {
                    do {
                        let output = try generator.generate(token: input, context: context)
                        guard let tokenValue: Float = output.to(type: Float.self) else {
                            continuation.finish()
                            break
                        }

                        continuation.yield(tokenValue)
                        input = MLXArray([tokenValue])
                    } catch {
                        print("Streaming generation stopped: \(error)")
                        continuation.finish()
                        break
                    }
                }
            }
        }
    }
}
