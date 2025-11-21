import Foundation
import AVFoundation
import Accelerate

final class AudioManager: ObservableObject {
    private let engine = AVAudioEngine()
    private let player = AVAudioPlayerNode()
    private let mixer = AVAudioMixerNode()

    private let session = AVAudioSession.sharedInstance()
    private let processingQueue = DispatchQueue(label: "buddy.audio.queue")

    @Published var outputLevel: Float = 0.2
    var vadThreshold: Float = 0.08

    init() {
        configureSession()
        configureEngine()
    }

    func startDuplex() {
        processingQueue.async { [weak self] in
            do {
                try self?.engine.start()
            } catch {
                print("Audio engine failed to start: \(error)")
            }
        }
    }

    func stop() {
        processingQueue.async { [weak self] in
            self?.player.stop()
            self?.engine.stop()
        }
    }

    func enqueueOutput(buffer: AVAudioPCMBuffer) {
        processingQueue.async { [weak self] in
            self?.player.scheduleBuffer(buffer, completionHandler: nil)
            if self?.player.isPlaying == false {
                self?.player.play()
            }
        }
    }

    private func configureSession() {
        do {
            try session.setCategory(.playAndRecord, mode: .voiceChat, options: [.allowBluetooth, .defaultToSpeaker, .duckOthers])
            try session.setActive(true)
        } catch {
            print("Audio session configuration failed: \(error)")
        }
    }

    private func configureEngine() {
        let input = engine.inputNode
        let outputFormat = input.inputFormat(forBus: 0)

        engine.attach(player)
        engine.attach(mixer)

        engine.connect(player, to: mixer, format: outputFormat)
        engine.connect(mixer, to: engine.mainMixerNode, format: outputFormat)

        installInputTap(format: outputFormat)
        installOutputTap()
    }

    private func installInputTap(format: AVAudioFormat) {
        let frameCount = AVAudioFrameCount(format.sampleRate * 0.05)
        engine.inputNode.removeTap(onBus: 0)
        engine.inputNode.installTap(onBus: 0, bufferSize: frameCount, format: format) { [weak self] buffer, _ in
            guard let self else { return }
            self.evaluateVAD(buffer: buffer)
        }
    }

    private func installOutputTap() {
        let outputFormat = engine.mainMixerNode.outputFormat(forBus: 0)
        engine.mainMixerNode.removeTap(onBus: 0)
        engine.mainMixerNode.installTap(onBus: 0, bufferSize: 1024, format: outputFormat) { [weak self] buffer, _ in
            guard let self else { return }
            self.updateOutputLevel(buffer: buffer)
        }
    }

    private func evaluateVAD(buffer: AVAudioPCMBuffer) {
        guard let channelData = buffer.floatChannelData else { return }
        let frameLength = Int(buffer.frameLength)
        var maxAmplitude: Float = 0

        for frame in 0..<frameLength {
            maxAmplitude = max(maxAmplitude, abs(channelData[0][frame]))
        }

        if maxAmplitude > vadThreshold {
            stopOutputPlayback()
        }
    }

    private func stopOutputPlayback() {
        processingQueue.async { [weak self] in
            if self?.player.isPlaying == true {
                self?.player.pause()
            }
        }
    }

    private func updateOutputLevel(buffer: AVAudioPCMBuffer) {
        guard let channelData = buffer.floatChannelData else { return }
        let frameLength = Int(buffer.frameLength)
        var rms: Float = 0

        vDSP_rmsqv(channelData[0], 1, &rms, vDSP_Length(frameLength))
        DispatchQueue.main.async {
            self.outputLevel = max(0.05, min(1.0, rms * 4))
        }
    }
}
