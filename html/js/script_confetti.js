// ===== Special Effects =====
// canvas-confetti 로드 후 전역 함수 등록
import("https://cdn.jsdelivr.net/npm/canvas-confetti@1.6.0/dist/confetti.browser.min.js").then(() => {
    window.showConfetti = () => {
        const duration = 2000;
        const end = Date.now() + duration;

        (function frame() {
            confetti({ particleCount: 4, angle: 60, spread: 55, origin: { x: 0 } });
            confetti({ particleCount: 4, angle: 120, spread: 55, origin: { x: 1 } });
            if (Date.now() < end) requestAnimationFrame(frame);
        })();
    };
});