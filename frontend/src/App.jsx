import { useEffect, useMemo, useRef, useState } from "react";
import "./App.css";

const API = "http://localhost:8000";

/** --- tiny helpers --- **/
function clamp(n, a, b) {
  return Math.max(a, Math.min(b, n));
}
function fmt(n, digits = 2) {
  if (n === null || n === undefined || Number.isNaN(Number(n))) return "—";
  return Number(n).toFixed(digits);
}
function fmtScore(n, digits = 0) {
  if (n === null || n === undefined || Number.isNaN(Number(n))) return "—";
  return Number(n).toFixed(digits);
}
function niceName(filename) {
  if (!filename) return "";
  return filename.replace(/_/g, " ").replace(/\.wav$/i, "");
}
function isWav(name) {
  return (name || "").toLowerCase().endsWith(".wav");
}
function pickDefaultTrack(filesDict) {
  if (!filesDict) return null;
  const keys = Object.keys(filesDict);
  const wavs = keys.filter((k) => (k || "").toLowerCase().endsWith(".wav"));
  if (wavs.length === 0) return null;

  const lower = (s) => (s || "").toLowerCase();
  return (
    wavs.find((k) => lower(k).includes("complete_arrangement")) ||
    wavs.find((k) => lower(k).includes("complete")) ||
    wavs.find((k) => lower(k).includes("final")) ||
    wavs.find((k) => lower(k).includes("reference")) ||
    wavs.find((k) => lower(k).includes("stem")) ||
    wavs[0]
  );
}

function instrumentFromFilename(name) {
  const s = (name || "").toLowerCase();
  if (s.includes("drum")) return "drums";
  if (s.includes("bass")) return "bass";
  if (s.includes("guitar")) return "guitar";
  if (s.includes("complete")) return "mix";
  if (s.includes("arrangement")) return "mix";
  return "mix";
}

function shortStemLabel(name) {
  const inst = instrumentFromFilename(name);
  if (inst === "mix") return "Complete";
  if (inst === "bass") return "Bass";
  if (inst === "drums") return "Drums";
  if (inst === "guitar") return "Guitar";
  return "Stem";
}

function sortTracksCompleteFirst(a, b) {
  const la = (a || "").toLowerCase();
  const lb = (b || "").toLowerCase();
  const aIsComplete = la.includes("complete");
  const bIsComplete = lb.includes("complete");
  if (aIsComplete && !bIsComplete) return -1;
  if (!aIsComplete && bIsComplete) return 1;
  return la.localeCompare(lb);
}

/** --- simple responsive size observer --- **/
function useResizeObserver(ref) {
  const [rect, setRect] = useState({ width: 0, height: 0 });
  useEffect(() => {
    if (!ref.current) return;
    const el = ref.current;
    const ro = new ResizeObserver((entries) => {
      const r = entries[0]?.contentRect;
      if (r) setRect({ width: r.width, height: r.height });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [ref]);
  return rect;
}

/** --- Canvas chart helpers --- **/
function ensureRoundRect(ctx) {
  if (ctx.roundRect) return;
  ctx.roundRect = function (x, y, w, h, r) {
    const rr = Math.min(r, w / 2, h / 2);
    this.beginPath();
    this.moveTo(x + rr, y);
    this.arcTo(x + w, y, x + w, y + h, rr);
    this.arcTo(x + w, y + h, x, y + h, rr);
    this.arcTo(x, y + h, x, y, rr);
    this.arcTo(x, y, x + w, y, rr);
    this.closePath();
    return this;
  };
}

/** --- dark, diamond-spark canvas line chart (with hover tooltip) --- **/
function CanvasLineChart({
  title,
  x,
  y,
  yLabel,
  height = 118,
  palette,
  yMin = null,
  yMax = null,
  sparkle = true,
}) {
  const wrapRef = useRef(null);
  const canvasRef = useRef(null);
  const { width } = useResizeObserver(wrapRef);

  const hoverRef = useRef({ active: false, mx: 0, my: 0 });
  const rafRef = useRef(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !width) return;

    const dpr = window.devicePixelRatio || 1;
    const W = Math.max(10, Math.floor(width));
    const H = height;

    canvas.width = Math.floor(W * dpr);
    canvas.height = Math.floor(H * dpr);
    canvas.style.width = `${W}px`;
    canvas.style.height = `${H}px`;

    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ensureRoundRect(ctx);

    const render = () => {
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      // background (dark glass)
      ctx.clearRect(0, 0, W, H);
      const bg = ctx.createLinearGradient(0, 0, 0, H);
      bg.addColorStop(0, "rgba(6,10,20,0.88)");
      bg.addColorStop(1, "rgba(6,10,20,0.72)");
      ctx.fillStyle = bg;
      ctx.fillRect(0, 0, W, H);

      // subtle glow
      ctx.globalCompositeOperation = "screen";
      const glow = ctx.createRadialGradient(W * 0.2, H * 0.25, 10, W * 0.2, H * 0.25, Math.max(W, H));
      glow.addColorStop(0, palette?.glowSoft || "rgba(160,140,255,0.18)");
      glow.addColorStop(1, "rgba(0,0,0,0)");
      ctx.fillStyle = glow;
      ctx.fillRect(0, 0, W, H);
      ctx.globalCompositeOperation = "source-over";

      if (!x || !y || x.length < 2 || y.length < 2) {
        ctx.fillStyle = "rgba(233,236,255,0.55)";
        ctx.font = "12px system-ui";
        ctx.fillText("No data", 12, 22);
        return;
      }

      const N = Math.min(x.length, y.length);

      let ymin = yMin ?? Infinity;
      let ymax = yMax ?? -Infinity;
      if (yMin == null || yMax == null) {
        for (let i = 0; i < N; i++) {
          const v = y[i];
          if (v == null || Number.isNaN(v)) continue;
          if (yMin == null) ymin = Math.min(ymin, v);
          if (yMax == null) ymax = Math.max(ymax, v);
        }
      }
      if (!isFinite(ymin) || !isFinite(ymax) || ymin === ymax) {
        ymin = isFinite(ymin) ? ymin - 1 : 0;
        ymax = isFinite(ymax) ? ymax + 1 : 1;
      }

      // compact paddings
      const padL = 42;
      const padR = 10;
      const padT = 14;
      const padB = 16;
      const plotW = W - padL - padR;
      const plotH = H - padT - padB;

      const x0 = x[0];
      const x1 = x[N - 1];
      const xSpan = x1 - x0 || 1;

      const X = (i) => padL + ((x[i] - x0) / xSpan) * plotW;
      const Y = (v) => padT + (1 - (v - ymin) / (ymax - ymin)) * plotH;

      // grid
      ctx.strokeStyle = "rgba(233,236,255,0.06)";
      ctx.lineWidth = 1;
      for (let g = 0; g <= 3; g++) {
        const yy = padT + (g / 3) * plotH;
        ctx.beginPath();
        ctx.moveTo(padL, yy);
        ctx.lineTo(W - padR, yy);
        ctx.stroke();
      }

      // y labels
      ctx.fillStyle = "rgba(233,236,255,0.58)";
      ctx.font = "10px system-ui";
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      ctx.fillText(fmt(ymax, 2), padL - 8, padT);
      ctx.fillText(fmt((ymin + ymax) / 2, 2), padL - 8, padT + plotH / 2);
      ctx.fillText(fmt(ymin, 2), padL - 8, padT + plotH);

      // // y label (vertical on left, outside tick labels)
      // if (yLabel) {
      //   ctx.save();
      //   ctx.translate(12, padT + plotH / 2);
      //   ctx.rotate(-Math.PI / 2);
      //   ctx.fillStyle = "rgba(233,236,255,0.38)";
      //   ctx.font = "10px system-ui";
      //   ctx.textAlign = "center";
      //   ctx.textBaseline = "middle";
      //   ctx.fillText(yLabel, 0, 0);
      //   ctx.restore();
      // }

      const line = palette?.line || "rgba(233,236,255,0.9)";
      const fill = palette?.fill || "rgba(233,236,255,0.07)";
      const glowStrong = palette?.glow || "rgba(160,140,255,0.35)";
      const point = palette?.point || "rgba(255,255,255,0.92)";

      // polyline points
      const pts = [];
      for (let i = 0; i < N; i++) {
        const v = y[i];
        if (v == null || Number.isNaN(v)) continue;
        pts.push([X(i), Y(v), i]);
      }
      if (pts.length < 2) return;

      // fill under curve
      ctx.beginPath();
      ctx.moveTo(pts[0][0], pts[0][1]);
      for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
      ctx.lineTo(pts[pts.length - 1][0], padT + plotH);
      ctx.lineTo(pts[0][0], padT + plotH);
      ctx.closePath();
      ctx.fillStyle = fill;
      ctx.fill();

      // glow stroke
      ctx.globalCompositeOperation = "screen";
      ctx.strokeStyle = glowStrong;
      ctx.lineWidth = 4;
      ctx.beginPath();
      ctx.moveTo(pts[0][0], pts[0][1]);
      for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
      ctx.stroke();

      // main stroke
      ctx.strokeStyle = line;
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      ctx.moveTo(pts[0][0], pts[0][1]);
      for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
      ctx.stroke();
      ctx.globalCompositeOperation = "source-over";

      // small diamond points (smaller than before)
      if (sparkle) {
        ctx.globalCompositeOperation = "screen";
        const step = Math.max(7, Math.floor(pts.length / 42));
        for (let i = 0; i < pts.length; i += step) {
          const [px, py] = pts[i];
          const s = 1.4 + (i % (step * 4) === 0 ? 0.6 : 0);

          // glow
          ctx.fillStyle = glowStrong;
          ctx.beginPath();
          ctx.arc(px, py, s * 1.7, 0, Math.PI * 2);
          ctx.fill();

          // diamond
          ctx.save();
          ctx.translate(px, py);
          ctx.rotate(Math.PI / 4);
          ctx.fillStyle = point;
          ctx.fillRect(-s, -s, s * 2, s * 2);
          ctx.restore();
        }
        ctx.globalCompositeOperation = "source-over";
      }

      // hover tooltip
      const hv = hoverRef.current;
      if (hv.active) {
        const nx = clamp((hv.mx - padL) / Math.max(1, plotW), 0, 1);
        const idx = Math.round(nx * (N - 1));

        const vx = x[idx];
        const vy = y[idx];
        if (vx != null && vy != null && !Number.isNaN(vy)) {
          const px = X(idx);
          const py = Y(vy);

          ctx.globalCompositeOperation = "screen";

          // vertical line
          ctx.strokeStyle = "rgba(233,236,255,0.16)";
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(px, padT);
          ctx.lineTo(px, padT + plotH);
          ctx.stroke();

          // marker diamond (small)
          ctx.fillStyle = glowStrong;
          ctx.beginPath();
          ctx.arc(px, py, 5, 0, Math.PI * 2);
          ctx.fill();

          ctx.save();
          ctx.translate(px, py);
          ctx.rotate(Math.PI / 4);
          ctx.fillStyle = "rgba(245,247,255,0.92)";
          ctx.fillRect(-3, -3, 6, 6);
          ctx.restore();

          ctx.globalCompositeOperation = "source-over";

          // tooltip box
          const label = `${fmt(vx, 2)}s • ${fmt(vy, 3)}`;
          ctx.font = "10.5px system-ui";
          const tw = ctx.measureText(label).width;
          const bx = clamp(px + 10, 8, W - (tw + 18));
          const by = clamp(py - 22, 8, H - 26);

          ctx.fillStyle = "rgba(0,0,0,0.55)";
          ctx.strokeStyle = "rgba(255,255,255,0.10)";
          ctx.lineWidth = 1;
          ctx.roundRect(bx, by, tw + 14, 18, 8);
          ctx.fill();
          ctx.stroke();

          ctx.fillStyle = "rgba(233,236,255,0.86)";
          ctx.textAlign = "left";
          ctx.textBaseline = "middle";
          ctx.fillText(label, bx + 7, by + 9);
        }
      }

      // x labels
      ctx.fillStyle = "rgba(233,236,255,0.45)";
      ctx.font = "10px system-ui";
      ctx.textAlign = "right";
      ctx.textBaseline = "bottom";
      ctx.fillText(`${fmt(x1, 2)}s`, W - padR, H - 4);
      ctx.textAlign = "left";
      ctx.fillText(`${fmt(x0, 2)}s`, padL, H - 4);
    };

    const schedule = () => {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = requestAnimationFrame(render);
    };

    // initial
    render();

    const onMove = (e) => {
      const r = canvas.getBoundingClientRect();
      hoverRef.current.active = true;
      hoverRef.current.mx = e.clientX - r.left;
      hoverRef.current.my = e.clientY - r.top;
      schedule();
    };
    const onLeave = () => {
      hoverRef.current.active = false;
      schedule();
    };

    canvas.addEventListener("mousemove", onMove, { passive: true });
    canvas.addEventListener("mouseleave", onLeave);

    return () => {
      cancelAnimationFrame(rafRef.current);
      canvas.removeEventListener("mousemove", onMove);
      canvas.removeEventListener("mouseleave", onLeave);
    };
  }, [x, y, width, height, palette, yLabel, yMin, yMax, sparkle]);

  return (
    <div className="chartCard">
      <div className="chartTitleRow">
        <div className="chartTitle">{title}</div>
        <div className="chartHint">hover for values</div>
      </div>
      <div ref={wrapRef} className="chartWrap">
        <canvas ref={canvasRef} />
      </div>
    </div>
  );
}

/** --- Tesseract Field (canvas) --- **/
function lerp(a, b, t) {
  return a + (b - a) * t;
}
function clamp01(x) {
  return Math.max(0, Math.min(1, x));
}
function sampleCurve(tArr, yArr, t) {
  if (!tArr || !yArr || tArr.length < 2) return 0;
  const n = Math.min(tArr.length, yArr.length);
  if (t <= tArr[0]) return yArr[0] ?? 0;
  if (t >= tArr[n - 1]) return yArr[n - 1] ?? 0;

  let lo = 0,
    hi = n - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (tArr[mid] <= t) lo = mid;
    else hi = mid;
  }
  const t0 = tArr[lo],
    t1 = tArr[hi];
  const y0 = yArr[lo] ?? 0,
    y1 = yArr[hi] ?? 0;
  const u = (t - t0) / ((t1 - t0) || 1);
  return lerp(y0, y1, u);
}

function TesseractField({ audioRef, analysis, palette, label = "Tesseract" }) {
  const wrapRef = useRef(null);
  const canvasRef = useRef(null);
  const rafRef = useRef(null);
  const stateRef = useRef({
    lastNow: 0,
    smoothAmp: 0,
    smoothPitch: 0,
    phase: 0,
    particles: [],
  });

  const { width, height } = useResizeObserver(wrapRef);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !width || !height) return;

    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;

    canvas.width = Math.floor(width * dpr);
    canvas.height = Math.floor(height * dpr);
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const drawDiamond = (x, y, s, rot, fill, glow) => {
      ctx.globalCompositeOperation = "screen";
      ctx.fillStyle = glow;
      ctx.beginPath();
      ctx.arc(x, y, s * 1.8, 0, Math.PI * 2);
      ctx.fill();

      ctx.save();
      ctx.translate(x, y);
      ctx.rotate(Math.PI / 4 + rot);
      ctx.fillStyle = fill;
      ctx.fillRect(-s, -s, s * 2, s * 2);
      ctx.restore();

      ctx.globalCompositeOperation = "source-over";
    };

    const draw = (now) => {
      const st = stateRef.current;
      if (!st.lastNow) st.lastNow = now;
      const dt = Math.min(0.05, (now - st.lastNow) / 1000);
      st.lastNow = now;

      const audio = audioRef?.current;
      const isPlaying = !!audio && !audio.paused && !audio.ended;
      const curT = audio?.currentTime ?? 0;

      const rms = analysis?.rms;
      const f0 = analysis?.f0;
      const wav = analysis?.wave;

      const ampRaw = sampleCurve(rms?.t, rms?.y, curT);
      const pitchRaw = sampleCurve(f0?.t, f0?.y, curT);
      const wavRaw = sampleCurve(wav?.t, wav?.y, curT);

      const amp = clamp01(ampRaw * 1.25);
      const pitchN = clamp01((pitchRaw - 40) / 260);
      const jitter = clamp01(Math.abs(wavRaw) * 1.35);

      const smoothK = 1 - Math.pow(0.0001, dt);
      st.smoothAmp = lerp(st.smoothAmp, amp, smoothK);
      st.smoothPitch = lerp(st.smoothPitch, pitchN, smoothK);
      st.phase += dt * (isPlaying ? (0.55 + st.smoothAmp * 2.4) : 0.18);

      const W = width,
        H = height;
      ctx.clearRect(0, 0, W, H);

      const glowA = palette?.tessA || "rgba(160,140,255,0.22)";
      const glowB = palette?.tessB || "rgba(80,160,255,0.12)";

      // void
      const g = ctx.createRadialGradient(W * 0.35, H * 0.25, 24, W * 0.55, H * 0.65, Math.max(W, H));
      g.addColorStop(0, glowA);
      g.addColorStop(0.35, "rgba(14,16,32,0.80)");
      g.addColorStop(1, "rgba(0,0,0,0.94)");
      ctx.fillStyle = g;
      ctx.fillRect(0, 0, W, H);

      // faint bands
      ctx.globalCompositeOperation = "screen";
      for (let i = 0; i < 3; i++) {
        const a = 0.04 + st.smoothAmp * 0.10;
        ctx.fillStyle = `rgba(233,236,255,${a})`;
        const yy =
          H * (0.18 + i * 0.30) +
          Math.sin(st.phase * (0.7 + i * 0.25)) * (10 + 44 * st.smoothPitch);
        ctx.fillRect(0, yy, W, 2);
      }
      ctx.globalCompositeOperation = "source-over";

      // lattice
      const layers = [
        { z: 0.20, scale: 24, alpha: 0.085, lw: 1.0 },
        { z: 0.42, scale: 36, alpha: 0.11, lw: 1.0 },
        { z: 0.68, scale: 54, alpha: 0.15, lw: 1.1 },
        { z: 0.92, scale: 78, alpha: 0.18, lw: 1.2 },
      ];

      const centerX = W * 0.55;
      const centerY = H * 0.46;

      const warp = 0.10 + st.smoothAmp * 0.62;
      const twist = (st.smoothPitch - 0.5) * 1.2;
      const driftX = Math.sin(st.phase * 0.9) * (10 + 70 * st.smoothAmp);
      const driftY = Math.cos(st.phase * 0.7) * (10 + 55 * st.smoothAmp);

      for (const L of layers) {
        const s = L.scale;
        const alpha = L.alpha * (0.70 + st.smoothAmp * 1.10);
        const par = 1 - L.z;

        const ox = driftX * par + Math.sin(st.phase * (0.8 + L.z)) * (14 * par);
        const oy = driftY * par + Math.cos(st.phase * (0.6 + L.z)) * (10 * par);

        const ang = twist * 0.7 * par + Math.sin(st.phase * 0.35) * (0.18 * par);
        const skew = 0.35 + 0.25 * Math.sin(st.phase * (0.5 + L.z));

        ctx.save();
        ctx.translate(centerX + ox, centerY + oy);
        ctx.rotate(ang);
        ctx.transform(1, 0, skew, 1, 0, 0);
        ctx.translate(-centerX, -centerY);

        ctx.lineWidth = L.lw;
        ctx.strokeStyle = isPlaying
          ? `rgba(233,236,255,${alpha * (0.55 + st.smoothAmp)})`
          : `rgba(210,200,255,${alpha})`;

        const left = -W * 0.2;
        const right = W * 1.2;
        const top = -H * 0.2;
        const bottom = H * 1.2;

        for (let x = left; x <= right; x += s) {
          ctx.beginPath();
          for (let y = top; y <= bottom; y += 18) {
            const wx = x + Math.sin(y * 0.012 + st.phase * (0.8 + L.z)) * (warp * 30 * (0.6 + L.z));
            const wy = y + Math.sin(x * 0.010 - st.phase * (0.6 + L.z)) * (warp * 18 * (0.4 + L.z));
            if (y === top) ctx.moveTo(wx, wy);
            else ctx.lineTo(wx, wy);
          }
          ctx.stroke();
        }
        for (let y = top; y <= bottom; y += s) {
          ctx.beginPath();
          for (let x = left; x <= right; x += 18) {
            const wx = x + Math.sin(y * 0.010 - st.phase * (0.9 + L.z)) * (warp * 26 * (0.5 + L.z));
            const wy = y + Math.sin(x * 0.012 + st.phase * (0.7 + L.z)) * (warp * 16 * (0.4 + L.z));
            if (x === left) ctx.moveTo(wx, wy);
            else ctx.lineTo(wx, wy);
          }
          ctx.stroke();
        }

        // beams
        ctx.globalCompositeOperation = "screen";
        ctx.lineWidth = 1.8;
        const beamA = 0.045 + st.smoothAmp * 0.18;
        ctx.strokeStyle = `rgba(255,255,255,${beamA * L.z})`;
        for (let b = 0; b < 2; b++) {
          const by = H * (0.25 + 0.35 * b) + Math.sin(st.phase * (1.2 + b)) * (28 + 82 * jitter);
          ctx.beginPath();
          ctx.moveTo(0, by);
          ctx.lineTo(W, by + 28 * skew);
          ctx.stroke();
        }
        ctx.globalCompositeOperation = "source-over";

        ctx.restore();
      }

      // === SPARKLING DIAMOND PARTICLES OVERLAY (audio-reactive, smaller) ===
      const parts = st.particles;
      const spawn = isPlaying ? Math.floor(1 + st.smoothAmp * 6) : 0;

      for (let i = 0; i < spawn; i++) {
        const s = 0.7 + Math.random() * (1.1 + st.smoothAmp * 0.6); // smaller
        parts.push({
          x: Math.random() * W,
          y: H * (0.65 + Math.random() * 0.45),
          vx: (Math.random() - 0.5) * (18 + 40 * st.smoothAmp),
          vy: -(22 + Math.random() * 60) * (1 + st.smoothAmp * 1.8),
          life: 0.9 + Math.random() * 0.9,
          rot: Math.random() * 1.6,
          vr: (Math.random() - 0.5) * 1.8,
          s,
        });
      }

      // cap count for perf
      if (parts.length > 110) parts.splice(0, parts.length - 110);

      for (let i = parts.length - 1; i >= 0; i--) {
        const p = parts[i];
        p.life -= dt;
        p.x += p.vx * dt;
        p.y += p.vy * dt;
        p.rot += p.vr * dt;

        if (p.life <= 0 || p.y < -40 || p.x < -60 || p.x > W + 60) {
          parts.splice(i, 1);
          continue;
        }

        const a = clamp01(p.life);
        const fill = `rgba(245,247,255,${0.55 * a})`;
        const glow = palette?.glowSoft ? palette.glowSoft.replace("0.16", String(0.12 * a)) : `rgba(167,139,250,${0.10 * a})`;

        drawDiamond(p.x, p.y, p.s, p.rot, fill, glow);
      }

      rafRef.current = requestAnimationFrame(draw);
    };

    rafRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(rafRef.current);
  }, [width, height, audioRef, analysis, palette]);

  return (
    <div className="tessWrap" ref={wrapRef} aria-hidden="true">
      <canvas ref={canvasRef} className="tessCanvas" />
      <div className="tessChrome">
        <div className="tessPill">
          <span className="miniDiamond" /> {label}
        </div>
      </div>
    </div>
  );
}

/** --- Modern audio controls --- **/
function useAudioUI(audioRef) {
  const [playing, setPlaying] = useState(false);
  const [t, setT] = useState(0);
  const [dur, setDur] = useState(0);
  const [vol, setVol] = useState(0.85);

  useEffect(() => {
    const el = audioRef.current;
    if (!el) return;

    const onPlay = () => setPlaying(true);
    const onPause = () => setPlaying(false);
    const onTime = () => setT(el.currentTime || 0);
    const onMeta = () => setDur(el.duration || 0);

    el.volume = vol;

    el.addEventListener("play", onPlay);
    el.addEventListener("pause", onPause);
    el.addEventListener("timeupdate", onTime);
    el.addEventListener("loadedmetadata", onMeta);
    el.addEventListener("durationchange", onMeta);

    return () => {
      el.removeEventListener("play", onPlay);
      el.removeEventListener("pause", onPause);
      el.removeEventListener("timeupdate", onTime);
      el.removeEventListener("loadedmetadata", onMeta);
      el.removeEventListener("durationchange", onMeta);
    };
  }, [audioRef, vol]);

  const pct = dur ? clamp(t / dur, 0, 1) : 0;

  const toggle = async () => {
    const el = audioRef.current;
    if (!el) return;
    if (el.paused) await el.play();
    else el.pause();
  };

  const seek = (p) => {
    const el = audioRef.current;
    if (!el || !dur) return;
    el.currentTime = clamp(p, 0, 1) * dur;
  };

  const setVolume = (v) => {
    const el = audioRef.current;
    setVol(v);
    if (el) el.volume = v;
  };

  return { playing, t, dur, pct, vol, toggle, seek, setVolume };
}

function mmss(sec) {
  if (!isFinite(sec)) return "0:00";
  const s = Math.max(0, Math.floor(sec));
  const m = Math.floor(s / 60);
  const r = s % 60;
  return `${m}:${String(r).padStart(2, "0")}`;
}

export default function App() {
  // generation controls
  const [secondsStr, setSecondsStr] = useState("20");
  const [startIndexStr, setStartIndexStr] = useState("1855");

  // connectivity
  const [backendOk, setBackendOk] = useState(false);
  const [readyInfo, setReadyInfo] = useState(null);
  const ready = readyInfo && typeof readyInfo.ready === "boolean" ? readyInfo.ready : null;

  // runs browser
  const [runs, setRuns] = useState([]);
  const [selectedRunId, setSelectedRunId] = useState(null);
  const [runDetail, setRunDetail] = useState(null);

  // track analytics
  const [selectedTrack, setSelectedTrack] = useState(null);
  const [analysis, setAnalysis] = useState(null);

  // player
  const [audioUrl, setAudioUrl] = useState("");
  const [status, setStatus] = useState("");

  const audioRef = useRef(null);
  const ui = useAudioUI(audioRef);

  useEffect(() => {
    const el = audioRef.current;
    if (!el) return;

    const onPlay = () => document.body.classList.add("isPlaying");
    const onPause = () => document.body.classList.remove("isPlaying");
    const onEnded = () => document.body.classList.remove("isPlaying");

    el.addEventListener("play", onPlay);
    el.addEventListener("pause", onPause);
    el.addEventListener("ended", onEnded);

    return () => {
      el.removeEventListener("play", onPlay);
      el.removeEventListener("pause", onPause);
      el.removeEventListener("ended", onEnded);
    };
  }, [audioUrl]);

  // expose audio progress to CSS (0..1)
  useEffect(() => {
    const el = audioRef.current;
    if (!el) return;

    let raf = 0;
    const tick = () => {
      const p = el.duration ? el.currentTime / el.duration : 0;
      document.documentElement.style.setProperty("--tessT", String(p));
      raf = requestAnimationFrame(tick);
    };

    const start = () => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(tick);
    };
    const stop = () => cancelAnimationFrame(raf);

    el.addEventListener("play", start);
    el.addEventListener("pause", stop);
    el.addEventListener("ended", stop);

    return () => {
      cancelAnimationFrame(raf);
      el.removeEventListener("play", start);
      el.removeEventListener("pause", stop);
      el.removeEventListener("ended", stop);
    };
  }, [audioUrl]);

  useEffect(() => {
    if (!audioUrl) document.body.classList.remove("isPlaying");
  }, [audioUrl]);

  /** initial load */
  useEffect(() => {
    fetch(`${API}/health`)
      .then((r) => setBackendOk(r.ok))
      .catch(() => setBackendOk(false));

    fetch(`${API}/readiness`)
      .then(async (r) => {
        try {
          setReadyInfo(await r.json());
        } catch {
          setReadyInfo(null);
        }
      })
      .catch(() => setReadyInfo(null));

    refreshRuns();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function refreshRuns() {
    try {
      const r = await fetch(`${API}/runs`);
      if (!r.ok) throw new Error("runs list failed");
      const j = await r.json();
      setRuns(Array.isArray(j) ? j : []);
      if (!selectedRunId && Array.isArray(j) && j.length) {
        setSelectedRunId(j[0].run_id || j[0].id || null);
      }
    } catch {
      setRuns([]);
    }
  }

  /** when selected run changes */
  useEffect(() => {
    if (!selectedRunId) return;
    (async () => {
      setRunDetail(null);
      setAnalysis(null);
      setAudioUrl("");
      setStatus("Loading run…");
      try {
        const r = await fetch(`${API}/runs/${selectedRunId}`);
        if (!r.ok) throw new Error(`failed to load run ${selectedRunId}`);
        const j = await r.json();
        setRunDetail(j);

        const defaultTrack = pickDefaultTrack(j.files);
        setSelectedTrack(defaultTrack);
        if (defaultTrack) setAudioUrl(`${API}${j.files[defaultTrack]}`);

        setStatus("");
      } catch (e) {
        setStatus(String(e?.message || e));
      }
    })();
  }, [selectedRunId]);

  /** when selected track changes -> fetch analysis */
  useEffect(() => {
    if (!selectedRunId || !selectedTrack) return;
    if (!selectedTrack.toLowerCase().endsWith(".wav")) return;
    (async () => {
      setAnalysis(null);
      setStatus("Computing analytics…");
      try {
        const url = `${API}/runs/${selectedRunId}/analysis?file=${encodeURIComponent(selectedTrack)}`;
        const r = await fetch(url);
        if (!r.ok) throw new Error("analysis failed");
        const j = await r.json();
        setAnalysis(j);
        setStatus("");
      } catch (e) {
        setStatus(String(e?.message || e));
      }
    })();
  }, [selectedRunId, selectedTrack]);

  async function generate() {
    setStatus("Generating…");
    setAnalysis(null);
    setAudioUrl("");
    try {
      const si = clamp(Number(startIndexStr || 0), 0, 32000);
      const sec = clamp(Number(secondsStr || 0), 4, 240);

      const r = await fetch(`${API}/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ seconds: sec, start_index: si }),
      });
      if (!r.ok) {
        const t = await r.text();
        throw new Error(t || `Generate failed (${r.status})`);
      }
      const j = await r.json();

      await refreshRuns();
      setSelectedRunId(j.run_id);
      setStatus("Done ✦");
    } catch (e) {
      setStatus(String(e?.message || e));
    }
  }

  const files = runDetail?.files || {};
  const wavNames = useMemo(() => Object.keys(files).filter(isWav).sort(sortTracksCompleteFirst), [files]);
  const score = analysis?.scores || null;

  const selectedInst = useMemo(() => instrumentFromFilename(selectedTrack), [selectedTrack]);

  const palette = useMemo(() => {
    // per your request NOW:
    // drums = deep red
    // bass  = deeper blue (deeper than your zoom/purple vibe)
    const base = {
      mix: {
        line: "rgba(233,236,255,0.88)",
        fill: "rgba(233,236,255,0.06)",
        glow: "rgba(165,210,255,0.26)",
        glowSoft: "rgba(165,210,255,0.16)",
        point: "rgba(255,255,255,0.92)",
        tessA: "rgba(150,120,255,0.22)",
        tessB: "rgba(80,160,255,0.12)",
      },
      guitar: {
        line: "rgba(245,247,255,0.92)",
        fill: "rgba(245,247,255,0.06)",
        glow: "rgba(255,255,255,0.22)",
        glowSoft: "rgba(255,255,255,0.14)",
        point: "rgba(255,255,255,0.92)",
        tessA: "rgba(205,210,255,0.20)",
        tessB: "rgba(150,120,255,0.10)",
      },
      drums: {
        // deep red
        line: "rgba(190,18,60,0.95)",
        fill: "rgba(190,18,60,0.10)",
        glow: "rgba(244,63,94,0.26)",
        glowSoft: "rgba(244,63,94,0.14)",
        point: "rgba(255,235,240,0.95)",
        tessA: "rgba(190,18,60,0.22)",
        tessB: "rgba(244,63,94,0.10)",
      },
      bass: {
        // deeper blue
        line: "rgba(30,64,175,0.95)",
        fill: "rgba(30,64,175,0.10)",
        glow: "rgba(37,99,235,0.26)",
        glowSoft: "rgba(37,99,235,0.14)",
        point: "rgba(235,245,255,0.95)",
        tessA: "rgba(30,64,175,0.22)",
        tessB: "rgba(37,99,235,0.10)",
      },
    };
    return base[selectedInst] || base.mix;
  }, [selectedInst]);

  // const kpiMain = useMemo(() => {
  //   return {
  //     energy: score?.Energy ?? null,
  //     dynamics: score?.Dynamics ?? null,
  //     complexity: score?.Complexity ?? null,
  //     duration: analysis?.duration ?? null,
  //   };
  // }, [analysis, score]);

  const kpiMain = useMemo(() => {
    return {
      avgLoudness100: score?.AvgLoudness_100 ?? null,
      loudnessCV: score?.Loudness_CV ?? null,
      medianPitch: score?.MedianPitch_Hz ?? null,
      duration: analysis?.duration ?? null,
    };
  }, [analysis, score]);

  const kpiTiny = useMemo(() => {
    const m = runDetail?.meta || {};
    // ✅ Sec removed
    return [
      { k: "ID", v: m?.requested_start_index ?? m?.start_idx ?? null },
      { k: "SR", v: m?.audio?.sample_rate ?? null },
      { k: "Hop", v: m?.audio?.hop_samples ?? null },
      { k: "Lead", v: m?.instruments?.leader ?? m?.leader ?? null },
      { k: "Drum", v: m?.instruments?.add_drums ?? null },
    ].filter((x) => x.v !== null && x.v !== undefined);
  }, [runDetail]);

  const duckGain = useMemo(() => {
    const m = runDetail?.meta || {};
    const duck = m?.mix?.duck_strength ?? null;
    const gain = m?.mix?.leader_gain ?? null;
    return { duck, gain };
  }, [runDetail]);

  const marqueeRuns = useMemo(() => {
    if (!runs || runs.length === 0) return [];
    return runs.concat(runs);
  }, [runs]);

  return (
    <div className="shell">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="brandBar">
          <div className="logoMark">♫</div>
          <div>
            <div className="brandTitle">Perform Music</div>
            <div className="brandSub">Runs</div>
          </div>
        </div>

        <div className="statusPills">
          <span className={`pill ${backendOk ? "pillOk" : "pillBad"}`}>{backendOk ? "Backend OK" : "Backend Down"}</span>
          {ready !== null && <span className={`pill ${ready ? "pillOk" : "pillWarn"}`}>{ready ? "Ready" : "Not Ready"}</span>}
        </div>

        <div className="controlCard">
          <div className="controlTitle">New Run</div>

          <div className="controlRow">
            <label className="field">
              <span className="label">Seconds</span>
              <input
                className="input inputTight"
                type="number"
                min="4"
                max="240"
                step="1"
                value={secondsStr}
                onChange={(e) => setSecondsStr(e.target.value)} // allow ""
              />
            </label>

            <label className="field">
              <span className="label">Start Index</span>
              <input
                className="input inputTight"
                type="number"
                min="0"
                max="32000"
                step="1"
                value={startIndexStr}
                onChange={(e) => setStartIndexStr(e.target.value)}  // allow ""
              />
            </label>
          </div>

          <div className="buttonRow">
            <button className="button buttonTight" onClick={generate} disabled={!backendOk || ready === false}>
              Generate
            </button>
            <button className="buttonSecondary buttonTight" onClick={refreshRuns}>
              Refresh
            </button>
          </div>

          <div className="helperRow">
            <span className="helper"></span>
            <span className="helper">ID range 0–32000</span>
          </div>
        </div>

        <div className="runsHeader">
          <div className="runsTitle">Past Runs</div>
          <div className="runsHint">marquee • hover pauses</div>
        </div>

        <div className="runsMarquee" title="Hover to pause">
          {runs.length === 0 ? (
            <div className="runsEmpty">No runs found. Generate one.</div>
          ) : (
            <div className="marqueeInner">
              {marqueeRuns.map((r, i) => {
                const id = r.run_id || r.id || `run_${i}`;
                const active = (r.run_id || r.id) === selectedRunId;
                return (
                  <button
                    key={`${id}_${i}`}
                    className={`runItem ${active ? "runItemActive" : ""}`}
                    onClick={() => setSelectedRunId(r.run_id || r.id)}
                  >
                    <div className="runTop">
                      <div className="runId">{id}</div>
                      <div className="runSec">{r.seconds ? `${r.seconds}s` : ""}</div>
                    </div>
                    <div className="runMeta">
                      {r.leader ? `Leader: ${r.leader}` : ""}
                      {r.start_idx !== undefined ? ` • idx ${r.start_idx}` : ""}
                    </div>
                    <div className="runTime">{r.created_at || ""}</div>
                  </button>
                );
              })}
            </div>
          )}
        </div>

        <div className="sideFooter">
          <div className="sideFootSmall">API: {API}</div>
          {readyInfo?.features_path && <div className="sideFootSmall">Features: {String(readyInfo.features_path)}</div>}
        </div>
      </aside>

      {/* Main dashboard */}
      <main className="main">
        {/* KPI strip */}
        <section className="kpiStrip">
          <div className="kpiStripHead kpiStripHeadRow">
            {/* Left: Jump link */}
            <a
              className="topJumpLink"
              href="/rag"                 // change to your actual route
              target="_blank"             // remove if same-tab
              rel="noreferrer"
              title="Open the RAG page"
            >
              <span className="topJumpDot" />
              Audio RAG
            </a>

            {/* Right: Status pill (existing) */}
            <div className="kpiStatusPill">
              <span className="diamondGlow" />
              <span className="statusLine">{status || "Idle"}</span>
            </div>
          </div>

          <div className="kpiMain kpiMainPrimary">
            <div
              className="kpiMini kpiMiniPrimary hasTip"
              data-tip="Mean of the Loudness curve (RMS) scaled to 0–100"
            >
              <div className="kpiLabel">Mean Loudness</div>
              <div className="kpiValue">
                {kpiMain.avgLoudness100 != null ? fmt(kpiMain.avgLoudness100, 1) : "—"}
              </div>
            </div>

            <div
              className="kpiMini kpiMiniPrimary hasTip"
              data-tip="Std/mean (for loudness), Lower = steadier loudness"
            >
              <div className="kpiLabel">Loudness CV (%)</div>
              <div className="kpiValue">
                {kpiMain.loudnessCV != null ? `${fmt(kpiMain.loudnessCV * 100, 1)}` : "—"}
              </div>
            </div>

            <div
              className="kpiMini kpiMiniPrimary hasTip"
              data-tip="Median of pitch curve points (Hz) over voiced frames"
            >
              <div className="kpiLabel">Median Pitch</div>
              <div className="kpiValue">
                {kpiMain.medianPitch != null ? `${fmt(kpiMain.medianPitch, 1)}` : "—"}
              </div>
            </div>

            <div
              className="kpiMini kpiMiniPrimary hasTip"
              data-tip="Duration of the selected audio track (seconds)"
            >
              <div className="kpiLabel">Duration</div>
              <div className="kpiValue">
                {kpiMain.duration != null ? `${fmt(kpiMain.duration, 0)}s` : "—"}
              </div>
            </div>
          </div>

          <div className="kpiPills kpiPillsOneLine">
            {kpiTiny.map((it) => (
              <div key={it.k} className="kpiPill kpiPillTiny">
                <span className="kpiPillK">{it.k}</span>
                <span className="kpiPillV">{String(it.v)}</span>
              </div>
            ))}

            {/* ✅ IMPORTANT: make Ducking and Gain two separate pills so they never collide */}
            {duckGain.duck != null && (
              <div className="kpiPill kpiPillTiny">
                <span className="kpiPillK">Ducking</span>
                <span className="kpiPillV">{fmt(duckGain.duck, 2)}</span>
              </div>
            )}
            {duckGain.gain != null && (
              <div className="kpiPill kpiPillTiny">
                <span className="kpiPillK">Gain</span>
                <span className="kpiPillV">{fmt(duckGain.gain, 2)}</span>
              </div>
            )}
          </div>
        </section>

        {/* Tracks (compact) + charts */}
        <section className="contentGrid">
          <div className="leftStack">
            <div className="panel panelTight tracksPanel">
              <div className="tracksHeaderRow">
                <div className="panelTitle">Tracks</div>

                <div className="trackTabs">
                  {wavNames.length === 0 ? (
                    <span className="panelHintTight">No wav</span>
                  ) : (
                    wavNames.map((name) => (
                      <button
                        key={name}
                        className={`trackTab ${selectedTrack === name ? "trackTabActive" : ""}`}
                        onClick={() => {
                          setSelectedTrack(name);
                          setAudioUrl(`${API}${files[name]}`);
                        }}
                        title={name}
                      >
                        {shortStemLabel(name)}
                      </button>
                    ))
                  )}
                </div>

                <div className="panelHintTight">vibe: {selectedInst}</div>
              </div>

              {/* ✅ New compact layout: vertical list + player on right */}
              <div className="tracksCompact">
                <div className="playerMini">
                  <div className="playerMiniTop">
                    <div className="playerMiniSub">{selectedTrack ? niceName(selectedTrack) : ""}</div>
                  </div>

                  <audio ref={audioRef} src={audioUrl || ""} />

                  {audioUrl ? (
                    <div className="playerMiniUI">
                      <div className="transportTopRow">
                        <button
                          className={`playBtn playBtnInline ${ui.playing ? "isPlaying" : ""}`}
                          onClick={ui.toggle}
                          aria-label="Play/Pause"
                        >
                          <span className="playIcon" />
                        </button>

                        <input
                          className="range timeline timelineSm"
                          type="range"
                          min="0"
                          max="1"
                          step="0.001"
                          value={ui.pct}
                          onChange={(e) => ui.seek(Number(e.target.value))}
                          style={{ "--pct": ui.pct }}
                        />
                      </div>

                      <div className="timeRowSm">
                        <span className="timeText">{mmss(ui.t)}</span>
                        <span className="timeText">{mmss(ui.dur)}</span>
                      </div>

                      {/* keep volume hidden if you want */}
                      <div className="volWrap volWrapSm" title="Volume">
                        <span className="volIcon volIconSm" />
                        <input
                          className="range volume volumeSm"
                          type="range"
                          min="0"
                          max="1"
                          step="0.01"
                          value={ui.vol}
                          onChange={(e) => ui.setVolume(Number(e.target.value))}
                          style={{ "--pct": ui.vol }}
                        />
                      </div>
                    </div>
                  ) : (
                    <div className="empty">Select a track.</div>
                  )}

                  <div className="downloadRow downloadRowSm">
                    {runDetail?.download_zip_url && (
                      <a className="linkBtn linkBtnSm" href={`${API}${runDetail.download_zip_url}`}>
                        Download ZIP
                      </a>
                    )}
                    {selectedTrack && files[selectedTrack] && (
                      <a className="linkBtn linkBtnSm" href={`${API}${files[selectedTrack]}`} download>
                        Download Track
                      </a>
                    )}
                  </div>
                </div>
              </div>
            </div>

            {/* Charts stacked */}
            <div className="vizStack">
              <CanvasLineChart
                title="Loudness"
                x={analysis?.rms?.t || []}
                y={analysis?.rms?.y || []}
                yLabel="RMS"
                height={118}
                palette={palette}
                sparkle={true}
              />
              <CanvasLineChart
                title="Pitch"
                x={analysis?.f0?.t || []}
                y={analysis?.f0?.y || []}
                yLabel="Hz"
                height={118}
                palette={palette}
                yMin={0}
                sparkle={true}
              />
              <CanvasLineChart
                title="Waveform"
                x={analysis?.wave?.t || []}
                y={analysis?.wave?.y || []}
                yLabel="|amp|"
                height={118}
                palette={palette}
                yMin={0}
                sparkle={true}
              />
            </div>
          </div>

          <div className="rightStack">
            <div className="panel panelTight">
              <div className="panelHeaderTight">
                <div className="panelTitle">Config</div>
                <div className="panelHintTight">meta snapshot</div>
              </div>
              <pre className="jsonPre jsonPreTight">{JSON.stringify(runDetail?.meta || {}, null, 2)}</pre>
            </div>
          </div>
        </section>
      </main>

      {/* Right tesseract field */}
      <aside className="tesseract">
        <TesseractField audioRef={audioRef} analysis={analysis} palette={palette} label={`Tesseract • Vibe`} />
      </aside>
    </div>
  );
}
