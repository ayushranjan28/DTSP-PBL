/**
 * Satin Ridges (from React Bits Pro)
 * Flowing satin folds catching soft highlights.
 * Smooth dimensional topography for cards and sections.
 */
(function (global) {
  'use strict';

  global.initSatinRidges = function (container, options = {}) {
    if (!container) return null;

    const ridgeCount = options.ridgeCount || 5;
    const speed = options.speed || 0.001;
    const opacity = options.opacity || 0.45;

    const canvas = document.createElement('canvas');
    canvas.style.position = 'absolute';
    canvas.style.inset = '0';
    canvas.style.width = '100%';
    canvas.style.height = '100%';
    canvas.style.pointerEvents = 'none';
    canvas.style.display = 'block';
    if (window.getComputedStyle(container).position === 'static') {
      container.style.position = 'relative';
    }
    container.appendChild(canvas);

    const ctx = canvas.getContext('2d');
    let rafId = null;
    let width = 0;
    let height = 0;
    let time = 0;

    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
      width = container.clientWidth || 300;
      height = container.clientHeight || 200;
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };

    const resizeObserver = window.ResizeObserver ? new ResizeObserver(resize) : null;
    if (resizeObserver) {
      resizeObserver.observe(container);
    }

    const draw = () => {
      time += speed;
      ctx.clearRect(0, 0, width, height);

      for (let r = 0; r < ridgeCount; r++) {
        const normR = r / ridgeCount;
        const baseY = height * (0.3 + normR * 0.6);

        ctx.beginPath();
        const steps = Math.min(50, Math.floor(width / 15));
        ctx.moveTo(0, height);

        for (let i = 0; i <= steps; i++) {
          const x = (i / steps) * width;
          const nx = x / width;

          const waveA = Math.sin(nx * 3.5 + time * 2 + r * 1.2) * 25;
          const waveB = Math.cos(nx * 6 - time * 1.5 + r * 0.8) * 15;
          const y = baseY + waveA + waveB;

          if (i === 0) ctx.lineTo(x, y);
          else ctx.lineTo(x, y);
        }

        ctx.lineTo(width, height);
        ctx.closePath();

        const grad = ctx.createLinearGradient(0, baseY - 30, 0, height);
        const alpha = (0.05 + (1 - normR) * 0.12) * opacity;
        grad.addColorStop(0, `rgba(167, 139, 250, ${alpha * 2})`);
        grad.addColorStop(0.3, `rgba(99, 102, 241, ${alpha})`);
        grad.addColorStop(1, 'rgba(15, 23, 42, 0)');

        ctx.fillStyle = grad;
        ctx.fill();

        // Satin crest highlight stroke
        ctx.beginPath();
        for (let i = 0; i <= steps; i++) {
          const x = (i / steps) * width;
          const nx = x / width;
          const waveA = Math.sin(nx * 3.5 + time * 2 + r * 1.2) * 25;
          const waveB = Math.cos(nx * 6 - time * 1.5 + r * 0.8) * 15;
          const y = baseY + waveA + waveB;
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.strokeStyle = `rgba(255, 255, 255, ${0.08 + (1 - normR) * 0.12})`;
        ctx.lineWidth = 1.2;
        ctx.stroke();
      }

      rafId = requestAnimationFrame(draw);
    };

    resize();
    window.addEventListener('resize', resize);
    rafId = requestAnimationFrame(draw);

    return {
      destroy: () => {
        if (rafId) cancelAnimationFrame(rafId);
        window.removeEventListener('resize', resize);
        if (resizeObserver) resizeObserver.disconnect();
        canvas.remove();
      }
    };
  };
})(window);

