import SwiftUI

struct ContentView: View {
    @StateObject private var audioManager = AudioManager()
    @StateObject private var model = MoshiLLM()

    var body: some View {
        ZStack {
            Color.black
                .ignoresSafeArea()

            GlowingOrb(intensity: audioManager.outputLevel)
                .frame(width: 240, height: 240)
        }
        .onAppear {
            audioManager.startDuplex()
            model.prepare()
        }
        .prefersHomeIndicatorAutoHidden(true)
    }
}

struct GlowingOrb: View {
    let intensity: Float

    private var scale: CGFloat {
        let clamped = max(0.2, min(1.0, CGFloat(intensity)))
        return 1.0 + (clamped * 0.8)
    }

    var body: some View {
        Circle()
            .fill(
                RadialGradient(
                    gradient: Gradient(colors: [Color.blue.opacity(0.8), Color.cyan.opacity(0.2)]),
                    center: .center,
                    startRadius: 10,
                    endRadius: 120
                )
            )
            .overlay(
                Circle()
                    .stroke(Color.cyan.opacity(0.6), lineWidth: 6)
            )
            .shadow(color: .cyan.opacity(0.7), radius: 20)
            .scaleEffect(scale)
            .animation(.easeInOut(duration: 0.3), value: intensity)
    }
}

#Preview {
    ContentView()
}
