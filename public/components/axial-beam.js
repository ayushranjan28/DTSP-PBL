/**
 * Axial Beam (from React Bits Pro)
 * Soft rays of light rising from beyond the frame.
 * Volumetric illumination and atmospheric luminescence.
 */
(function (global) {
  'use strict';

  global.initAxialBeam = function (container, options = {}) {
    if (!container) return null;

    const beamCount = options.beamCount || 6;
    const baseColor = options.baseColor || '#4f46e5'; // Indigo
    const accentColor = options.accentColor || '#06b6d4'; // Cyan
    const glowIntensity = options.glowIntensity || 0.6;
    const speed = options.speed || 0.0012;

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

    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
      width = container.clientWidth || 300;
      height = container.clientHeight || 200;
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };

    const draw = () => {
      time += speed;
      ctx.clearRect(0, 0, width, height);

      // Ambient base radial glow at origin
      const originX = width * 0.5;
      const originY = height + 40;

      const baseGlow = ctx.createRadialGradient(originX, originY, 10, originX, originY, height * 1.2);
      baseGlow.addColorStop(0, 'rgba(99, 102, 241, 0.25)');
      baseGlow.addColorStop(0.5, 'rgba(6, 182, 212, 0.12)');
      baseGlow.addColorStop(1, 'rgba(13, 19, 34, 0)');

      ctx.fillStyle = baseGlow;
      ctx.fillRect(0, 0, width, height);

      // Draw volumetric rising shafts
      for (let i = 0; i < beamCount; i++) {
        const angleOffset = ((i - (beamCount - 1) / 2) * 0.18) + Math.sin(time * 2 + i) * 0.06;
        const beamWidth = width * (0.08 + Math.sin(time + i * 1.5) * 0.03);
        const topX = originX + Math.sin(angleOffset) * (height * 1.4);
        const topY = 0;

        const pulse = 0.5 + 0.5 * Math.sin(time * 3 + i * 1.2);
        const alpha = (0.08 + pulse * 0.12) * glowIntensity;

        const grad = ctx.createLinearGradient(originX, originY, topX, topY);
        grad.addColorStop(0, `rgba(99, 102, 241, ${alpha * 2})`);
        grad.addColorStop(0.4, `rgba(6, 182, 212, ${alpha * 1.2})`);
        grad.addColorStop(1, 'rgba(6, 182, 212, 0)');

        ctx.beginPath();
        ctx.fillStyle = grad;
        ctx.moveTo(originX - 15, originY);
        ctx.lineTo(originX + 15, originY);
        ctx.lineTo(topX + beamWidth * 0.5, topY);
        ctx.lineTo(topX - beamWidth * 0.5, topY);
        ctx.closePath();
        ctx.fill();
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
        canvas.remove();
      }
    };
  };
})(window);
