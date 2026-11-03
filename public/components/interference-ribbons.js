/**
 * Interference Ribbons (from React Bits Pro)
 * A ribbon of fine glowing wires drifting and twisting.
 * High-performance, mobile-optimized Canvas component.
 */
(function (global) {
  'use strict';

  global.initInterferenceRibbons = function (container, options = {}) {
    if (!container) return null;

    const numWires = options.numWires || 16;
    const ribbonCount = options.ribbonCount || 2;
    const wireColor1 = options.wireColor1 || '#06b6d4'; // Cyan
    const wireColor2 = options.wireColor2 || '#8b5cf6'; // Violet
    const wireColor3 = options.wireColor3 || '#ec4899'; // Pink
    const speed = options.speed || 0.0015;
    const amplitude = options.amplitude || 45;
    const tension = options.tension || 0.003;
    const opacity = options.opacity || 0.65;
    const interactive = options.interactive !== false;

    const canvas = document.createElement('canvas');
    canvas.style.position = 'absolute';
    canvas.style.inset = '0';
    canvas.style.width = '100%';
    canvas.style.height = '100%';
    canvas.style.pointerEvents = 'none';
    canvas.style.display = 'block';
    container.style.position = 'relative';
    container.appendChild(canvas);

    const ctx = canvas.getContext('2d');
    let rafId = null;
    let width = 0;
    let height = 0;
    let time = 0;
    let pointerX = 0.5;
    let pointerY = 0.5;
    let targetPointerX = 0.5;
    let targetPointerY = 0.5;

    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
      width = container.clientWidth || 300;
      height = container.clientHeight || 200;
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };

    const onPointerMove = (e) => {
      const rect = container.getBoundingClientRect();
      targetPointerX = (e.clientX - rect.left) / Math.max(rect.width, 1);
      targetPointerY = (e.clientY - rect.top) / Math.max(rect.height, 1);
    };

    if (interactive) {
      window.addEventListener('pointermove', onPointerMove, { passive: true });
    }

    const draw = () => {
      time += speed;
      ctx.clearRect(0, 0, width, height);

      pointerX += (targetPointerX - pointerX) * 0.05;
      pointerY += (targetPointerY - pointerY) * 0.05;

      const colors = [wireColor1, wireColor2, wireColor3];

      for (let r = 0; r < ribbonCount; r++) {
        const rOffset = (r * Math.PI) / ribbonCount;
        const col = colors[r % colors.length];

        for (let w = 0; w < numWires; w++) {
          const wireRatio = (w - numWires / 2) / (numWires / 2);
          const wireOffset = wireRatio * 18;

          ctx.beginPath();
          ctx.strokeStyle = col;
          ctx.globalAlpha = Math.max(0.1, (1 - Math.abs(wireRatio) * 0.6) * opacity);
          ctx.lineWidth = 1.0;
          ctx.shadowBlur = 8;
          ctx.shadowColor = col;

          const steps = Math.min(60, Math.floor(width / 12));
          for (let i = 0; i <= steps; i++) {
            const x = (i / steps) * width;
            const normX = x / width;

            // Multi-frequency wave formula simulating acoustic harmonic twisting
            const wave1 = Math.sin(normX * 5 + time * 3 + rOffset) * amplitude;
            const wave2 = Math.cos(normX * 9 - time * 2 + w * 0.2) * (amplitude * 0.4);
            const twist = Math.sin(time * 2 + normX * 4) * wireOffset;

            // Interactive deflection
            const distToPointer = Math.abs(normX - pointerX);
            const defl = Math.exp(-distToPointer * 6) * (pointerY - 0.5) * 60;

            const y = height * 0.5 + wave1 + wave2 + twist + defl + (r - (ribbonCount - 1) / 2) * 20;

            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
          }
          ctx.stroke();
        }
      }

      ctx.shadowBlur = 0;
      ctx.globalAlpha = 1.0;
      rafId = requestAnimationFrame(draw);
    };

    resize();
    window.addEventListener('resize', resize);
    rafId = requestAnimationFrame(draw);

    return {
      destroy: () => {
        if (rafId) cancelAnimationFrame(rafId);
        window.removeEventListener('resize', resize);
        if (interactive) window.removeEventListener('pointermove', onPointerMove);
        canvas.remove();
      }
    };
  };
})(window);
