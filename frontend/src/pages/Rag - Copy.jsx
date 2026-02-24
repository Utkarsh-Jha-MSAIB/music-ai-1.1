// Rag.jsx  (Stranger-Things Lights: REAL 3D FILAMENT • deep layers • NO color when silent • optimized)
// Drop-in replacement for your current Rag.jsx (keeps your UI + charts as-is, upgrades only the lights + fixes a bug).
//
// Key fixes/upgrades:
// ✅ Fix: LightsWall was calling a non-existent hook (useStableAudioAnalyser) -> now uses useStableAudioAnalyserRef
// ✅ Bulbs look like real 3D glass + socket + hot filament
// ✅ No colored glass when silent (bulbs look clear/grey until audio plays)
// ✅ Multi-depth layers + fog + parallax + zig-zag wiring (stable seeded scene)
// ✅ Performance: capped DPR, NO per-frame React re-renders, NO per-frame sort/slice allocations, no Math.random in bulb motion

import { useEffect, useMemo, useRef, useState } from "react";
import "./Rag.css";

const API = "http://localhost:8000";

/** --- tiny helpers --- **/
function clamp(n, a, b) {
  return Math.max(a, Math.min(b, n));
}
function lerp(a, b, t) {
  return a + (b - a) * t;
}
function clamp01(x) {
  return Math.max(0, Math.min(1, x));
}
function fmt(n, digits = 2) {
  if (n === null || n === undefined || Number.isNaN(Number(n))) return "—";
  return Number(n).toFixed(digits);
}
function niceName(filename) {
  if (!filename) return "";
  return filename.replace(/_/g, " ").replace(/\.wav$/i, "");
}

/** --- local history for “Past Runs” (RAG uploads) --- **/
const LS_KEY = "rag_runs_v1";
function loadRuns() {
  try {
    const j = JSON.parse(localStorage.getItem(LS_KEY) || "[]");
    return Array.isArray(j) ? j : [];
  } catch {
    return [];
  }
}
function saveRuns(runs) {
  try {
    localStorage.setItem(LS_KEY, JSON.stringify(runs.slice(0, 40)));
  } catch {}
}
function addRunEntry(entry) {
  const prev = loadRuns();
  const next = [entry, ...prev.filter((x) => x.upload_id !== entry.upload_id)];
  saveRuns(next);
  return next;
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

/** --- Canvas chart (same vibe as Perform Music) --- **/
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

function CanvasLineChart({
  title,
  x,
  y,
  yLabel,
  height = 130,
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

    const dpr = Math.min(1.6, window.devicePixelRatio || 1); // cap DPR = perf
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

      ctx.clearRect(0, 0, W, H);
      const bg = ctx.createLinearGradient(0, 0, 0, H);
      bg.addColorStop(0, "rgba(6,10,20,0.88)");
      bg.addColorStop(1, "rgba(6,10,20,0.72)");
      ctx.fillStyle = bg;
      ctx.fillRect(0, 0, W, H);

      ctx.globalCompositeOperation = "screen";
      const glow = ctx.createRadialGradient(
        W * 0.2,
        H * 0.25,
        10,
        W * 0.2,
        H * 0.25,
        Math.max(W, H)
      );
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

      const padL = 44;
      const padR = 10;
      const padT = 14;
      const padB = 18;
      const plotW = W - padL - padR;
      const plotH = H - padT - padB;

      const x0 = x[0];
      const x1 = x[N - 1];
      const xSpan = x1 - x0 || 1;

      const X = (i) => padL + ((x[i] - x0) / xSpan) * plotW;
      const Y = (v) => padT + (1 - (v - ymin) / (ymax - ymin)) * plotH;

      ctx.strokeStyle = "rgba(233,236,255,0.06)";
      ctx.lineWidth = 1;
      for (let g = 0; g <= 3; g++) {
        const yy = padT + (g / 3) * plotH;
        ctx.beginPath();
        ctx.moveTo(padL, yy);
        ctx.lineTo(W - padR, yy);
        ctx.stroke();
      }

      ctx.fillStyle = "rgba(233,236,255,0.58)";
      ctx.font = "10px system-ui";
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      ctx.fillText(fmt(ymax, 2), padL - 8, padT);
      ctx.fillText(fmt((ymin + ymax) / 2, 2), padL - 8, padT + plotH / 2);
      ctx.fillText(fmt(ymin, 2), padL - 8, padT + plotH);

      ctx.textAlign = "left";
      ctx.textBaseline = "top";
      ctx.fillStyle = "rgba(233,236,255,0.46)";
      ctx.fillText(yLabel || "", 12, 9);

      const line = palette?.line || "rgba(233,236,255,0.9)";
      const fill = palette?.fill || "rgba(233,236,255,0.07)";
      const glowStrong = palette?.glow || "rgba(160,140,255,0.35)";
      const point = palette?.point || "rgba(255,255,255,0.92)";

      // build points (minimal allocations)
      const pts = [];
      for (let i = 0; i < N; i++) {
        const v = y[i];
        if (v == null || Number.isNaN(v)) continue;
        pts.push([X(i), Y(v), i]);
      }
      if (pts.length < 2) return;

      ctx.beginPath();
      ctx.moveTo(pts[0][0], pts[0][1]);
      for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
      ctx.lineTo(pts[pts.length - 1][0], padT + plotH);
      ctx.lineTo(pts[0][0], padT + plotH);
      ctx.closePath();
      ctx.fillStyle = fill;
      ctx.fill();

      ctx.globalCompositeOperation = "screen";
      ctx.strokeStyle = glowStrong;
      ctx.lineWidth = 4;
      ctx.beginPath();
      ctx.moveTo(pts[0][0], pts[0][1]);
      for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
      ctx.stroke();

      ctx.strokeStyle = line;
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      ctx.moveTo(pts[0][0], pts[0][1]);
      for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
      ctx.stroke();
      ctx.globalCompositeOperation = "source-over";

      if (sparkle) {
        ctx.globalCompositeOperation = "screen";
        const step = Math.max(7, Math.floor(pts.length / 42));
        for (let i = 0; i < pts.length; i += step) {
          const [px, py] = pts[i];
          const s = 1.4 + (i % (step * 4) === 0 ? 0.6 : 0);

          ctx.fillStyle = glowStrong;
          ctx.beginPath();
          ctx.arc(px, py, s * 1.7, 0, Math.PI * 2);
          ctx.fill();

          ctx.save();
          ctx.translate(px, py);
          ctx.rotate(Math.PI / 4);
          ctx.fillStyle = point;
          ctx.fillRect(-s, -s, s * 2, s * 2);
          ctx.restore();
        }
        ctx.globalCompositeOperation = "source-over";
      }

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
          ctx.strokeStyle = "rgba(233,236,255,0.16)";
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(px, padT);
          ctx.lineTo(px, padT + plotH);
          ctx.stroke();

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

/** ---------- WebAudio: ultra-stable analyser (REF-based, no 60fps re-renders) ---------- **/
function useStableAudioAnalyserRef(activeAudioRef) {
  const rafRef = useRef(0);
  const ctxRef = useRef(null);
  const graphMapRef = useRef(new WeakMap());
  const bandCfgRef = useRef(null);

  const vizRef = useRef({
    playing: false,
    amp: 0,
    low: 0,
    mid: 0,
    high: 0,
    flux: 0,
    hit: false,
    hitStrength: 0,

    bands: new Float32Array(48),
    peakBand: 0,
    peak01: 0,

    stamp: 0,
  });

  const ensureCtx = async () => {
    if (!ctxRef.current) ctxRef.current = new (window.AudioContext || window.webkitAudioContext)();
    if (ctxRef.current.state === "suspended") await ctxRef.current.resume();
    return ctxRef.current;
  };

  const getGraph = async (el) => {
    const ctx = await ensureCtx();
    const existing = graphMapRef.current.get(el);
    if (existing) return existing;

    const analyser = ctx.createAnalyser();
    analyser.fftSize = 2048;
    analyser.smoothingTimeConstant = 0.68;

    const gain = ctx.createGain();
    gain.gain.value = 1.0;

    const source = ctx.createMediaElementSource(el);
    source.connect(analyser);
    analyser.connect(gain);
    gain.connect(ctx.destination);

    const timeData = new Uint8Array(analyser.fftSize);
    const freqData = new Uint8Array(analyser.frequencyBinCount);
    const prevMag = new Float32Array(analyser.frequencyBinCount);

    if (!bandCfgRef.current) {
      const n = analyser.frequencyBinCount;
      const B = vizRef.current.bands.length;

      const edges = new Array(B + 1);
      for (let i = 0; i <= B; i++) {
        const t = i / B;
        const u = Math.pow(t, 1.65);
        edges[i] = Math.max(0, Math.min(n, Math.floor(u * n)));
      }
      for (let i = 1; i <= B; i++) {
        if (edges[i] <= edges[i - 1]) edges[i] = Math.min(n, edges[i - 1] + 1);
      }
      bandCfgRef.current = { n, B, edges };
    }

    const graph = { ctx, source, analyser, gain, timeData, freqData, prevMag };
    graphMapRef.current.set(el, graph);
    return graph;
  };

  useEffect(() => {
    cancelAnimationFrame(rafRef.current);
    const el = activeAudioRef?.current;
    const v = vizRef.current;

    if (!el) {
      v.playing = false;
      v.amp = v.low = v.mid = v.high = v.flux = 0;
      v.hit = false;
      v.hitStrength = 0;
      v.peakBand = 0;
      v.peak01 = 0;
      v.bands.fill(0);
      return;
    }

    try {
      if (!el.crossOrigin) el.crossOrigin = "anonymous";
    } catch {}

    let mounted = true;

    let fluxEMA = 0;
    let fluxDevEMA = 0;
    let cooldown = 0;

    let ampSm = 0,
      lowSm = 0,
      midSm = 0,
      highSm = 0;
    let frame = 0;

    const tick = async () => {
      if (!mounted) return;

      const audio = activeAudioRef?.current;
      const playing = !!audio && !audio.paused && !audio.ended;

      let amp = 0,
        low = 0,
        mid = 0,
        high = 0,
        flux = 0;
      let hit = false,
        hitStrength = 0;

      if (playing) {
        let graph = null;
        try {
          graph = await getGraph(audio);
        } catch {
          graph = null;
        }

        if (graph?.analyser) {
          const an = graph.analyser;
          const td = graph.timeData;
          const fd = graph.freqData;

          an.getByteTimeDomainData(td);
          let sum = 0;
          for (let i = 0; i < td.length; i++) {
            const vv = (td[i] - 128) / 128;
            sum += vv * vv;
          }
          amp = Math.sqrt(sum / td.length);
          amp = Math.max(0, Math.min(1, amp * 2.2));

          an.getByteFrequencyData(fd);
          const n = fd.length;

          const iLow = Math.floor(n * 0.12);
          const iMid = Math.floor(n * 0.38);
          const iHigh = Math.floor(n * 0.78);

          let sl = 0,
            sm = 0,
            sh = 0;
          for (let i = 0; i < iLow; i++) sl += fd[i];
          for (let i = iLow; i < iMid; i++) sm += fd[i];
          for (let i = iMid; i < iHigh; i++) sh += fd[i];

          low = Math.max(0, Math.min(1, sl / Math.max(1, iLow) / 255));
          mid = Math.max(0, Math.min(1, sm / Math.max(1, iMid - iLow) / 255));
          high = Math.max(0, Math.min(1, sh / Math.max(1, iHigh - iMid) / 255));

          const cfg = bandCfgRef.current;
          const bands = v.bands;
          let peak = -1,
            peakIdx = 0;

          const floor = 0.02 + 0.08 * (1 - amp);

          for (let bi = 0; bi < cfg.B; bi++) {
            const a = cfg.edges[bi];
            const b = cfg.edges[bi + 1];
            let s = 0;
            for (let k = a; k < b; k++) s += fd[k];
            let val = s / Math.max(1, b - a) / 255;

            val = Math.max(0, val - floor);
            val = Math.pow(val, 0.65);
            bands[bi] = bands[bi] * 0.72 + val * 0.28;

            if (bands[bi] > peak) {
              peak = bands[bi];
              peakIdx = bi;
            }
          }
          v.peakBand = peakIdx;
          v.peak01 = cfg.B > 1 ? peakIdx / (cfg.B - 1) : 0;

          const prev = graph.prevMag;
          let num = 0,
            den = 0;
          const start = Math.floor(n * 0.02);
          const end = Math.floor(n * 0.92);

          for (let i = start; i < end; i++) {
            const m = fd[i] / 255;
            const d = m - prev[i];
            if (d > 0) num += d;
            den += m;
            prev[i] = m;
          }

          flux = den > 1e-6 ? Math.max(0, Math.min(1, num / (den * 0.55 + 1e-3) / 1.6)) : 0;

          fluxEMA = lerp(fluxEMA, flux, 0.10);
          const dev = Math.abs(flux - fluxEMA);
          fluxDevEMA = lerp(fluxDevEMA, dev, 0.12);

          const peakiness = Math.max(0, Math.min(1, peak * 1.25));
          const thresh = fluxEMA + 1.45 * fluxDevEMA + 0.02 - 0.015 * peakiness;

          cooldown = Math.max(0, cooldown - 1);
          if (cooldown === 0 && flux > Math.max(0.07, thresh) && amp > 0.025) {
            hit = true;
            hitStrength = Math.max(0, Math.min(1, (flux - thresh) / 0.22));
            cooldown = 7;
          }
        }
      }

      const attack = 0.30;
      const release = 0.12;
      ampSm = amp > ampSm ? lerp(ampSm, amp, attack) : lerp(ampSm, amp, release);
      lowSm = lerp(lowSm, low, 0.22);
      midSm = lerp(midSm, mid, 0.22);
      highSm = lerp(highSm, high, 0.22);

      v.playing = playing;
      v.amp = ampSm;
      v.low = lowSm;
      v.mid = midSm;
      v.high = highSm;
      v.flux = lerp(v.flux, flux, 0.26);
      v.hit = hit;
      v.hitStrength = hitStrength;

      frame++;
      if (frame % 6 === 0) v.stamp = (v.stamp + 1) % 1e9;

      rafRef.current = requestAnimationFrame(tick);
    };

    const onPlay = async () => {
      try {
        await ensureCtx();
        await getGraph(el);
      } catch {}
      cancelAnimationFrame(rafRef.current);
      rafRef.current = requestAnimationFrame(tick);
    };

    const onPauseOrEnd = () => {
      cancelAnimationFrame(rafRef.current);
      const decay = () => {
        v.playing = false;
        v.hit = false;
        v.hitStrength = 0;
        v.amp *= 0.92;
        v.low *= 0.9;
        v.mid *= 0.9;
        v.high *= 0.9;
        v.flux *= 0.9;
        for (let i = 0; i < v.bands.length; i++) v.bands[i] *= 0.92;
        rafRef.current = requestAnimationFrame(decay);
      };
      rafRef.current = requestAnimationFrame(decay);
    };

    el.addEventListener("play", onPlay);
    el.addEventListener("pause", onPauseOrEnd);
    el.addEventListener("ended", onPauseOrEnd);

    if (!el.paused && !el.ended) onPlay();
    else onPauseOrEnd();

    return () => {
      mounted = false;
      cancelAnimationFrame(rafRef.current);
      el.removeEventListener("play", onPlay);
      el.removeEventListener("pause", onPauseOrEnd);
      el.removeEventListener("ended", onPauseOrEnd);
    };
  }, [activeAudioRef]);

  return vizRef;
}

/** --- Seeded RNG so layout never changes on refresh --- */
const LW_SEED_KEY = "rag_lightswall_seed_v2";
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function getStableSeed() {
  try {
    const existing = localStorage.getItem(LW_SEED_KEY);
    if (existing) return Number(existing) >>> 0;
    const seed = (Math.random() * 2 ** 32) >>> 0;
    localStorage.setItem(LW_SEED_KEY, String(seed));
    return seed;
  } catch {
    return (Math.random() * 2 ** 32) >>> 0;
  }
}
function smoothstep(a, b, x) {
  const t = Math.max(0, Math.min(1, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
}

/** small deterministic “noise” (no Math.random per frame) */
function hash11(x) {
  // x: float -> 0..1
  const s = Math.sin(x * 127.1) * 43758.5453123;
  return s - Math.floor(s);
}
function noise2(t, seed) {
  // returns [-1..1]
  return (hash11(t + seed) * 2 - 1);
}

/** ---------- Stranger Lights Canvas (NEBULA BG • SINGLE BULB CHASE • STABLE SCENE) ---------- **/
function LightsWall({ audioRef, rms, label = "Alphabet Wall • 3D" }) {
  const wrapRef = useRef(null);
  const canvasRef = useRef(null);
  const rafRef = useRef(0);
  const [rect, setRect] = useState({ w: 0, h: 0 });

  // ✅ Use the correct analyser hook
  const vizRef = useStableAudioAnalyserRef(audioRef);

  const sceneRef = useRef(null);
  const bulbsRef = useRef([]);
  const strandsRef = useRef([]);
  const orderRef = useRef([]);

  // pre-sorted arrays (NO per-frame sorts)
  const strandsSortedRef = useRef([]);
  const bulbsSortedRef = useRef([]);

  const stateRef = useRef({
    seed: getStableSeed(),
    chaseIdx: 0,
    cooldown: 0,
    sparkles: [],
    drive: 0,
    lastAudioT: 0,
  });

  useEffect(() => {
    if (!wrapRef.current) return;
    const ro = new ResizeObserver((e) => {
      const r = e[0]?.contentRect;
      if (r) setRect({ w: r.width, h: r.height });
    });
    ro.observe(wrapRef.current);
    return () => ro.disconnect();
  }, []);

  /** ---------- Build stable scene (wires + bulbs + background particles) ---------- **/
  useEffect(() => {
    if (!rect.w || !rect.h) return;

    const W = rect.w;
    const H = rect.h;
    const VP = { x: W * 0.52, y: H * 0.28 };

    const seed = stateRef.current.seed;
    const rnd = mulberry32(seed);

    // More depth gaps
    const Z_LAYERS = [0.10, 0.18, 0.30, 0.46, 0.66, 0.96, 1.34];
    const strandCount = 20;

    const strands = [];
    const bulbs = [];
    const baseR = Math.max(1.45, Math.min(2.25, Math.min(W, H) / 520));

    const huePalette = [0.0, 0.08, 0.16, 0.34, 0.50, 0.62, 0.78, 0.86];

    // Prebaked nebula particles (bokeh + dust)
    const dust = [];
    const dustN = Math.floor((W * H) / 3200);
    for (let i = 0; i < dustN; i++) {
      const z = Z_LAYERS[Math.floor(rnd() * Z_LAYERS.length)];
      dust.push({
        x: rnd() * W,
        y: rnd() * H,
        r: (0.7 + rnd() * 2.4) * (0.45 + z * 0.75),
        a: 0.05 + rnd() * 0.10,
        hue: huePalette[Math.floor(rnd() * huePalette.length)],
        z,
        tw: 0.6 + rnd() * 1.6,
        ph: rnd() * 10,
      });
    }

    // Tiny star specks
    const stars = [];
    const starN = Math.floor((W * H) / 2100);
    for (let i = 0; i < starN; i++) {
      const z = 0.06 + rnd() * 1.2;
      stars.push({
        x: rnd() * W,
        y: rnd() * H,
        r: 0.35 + rnd() * 0.9,
        a: 0.04 + rnd() * 0.10,
        z,
        ph: rnd() * 10,
      });
    }

    for (let si = 0; si < strandCount; si++) {
      const z = Z_LAYERS[si % Z_LAYERS.length];

      const yBase = H * (0.14 + rnd() * 0.72);
      const p0 = { x: -60, y: yBase + (rnd() - 0.5) * 40 };
      const p3 = { x: W + 60, y: yBase + (rnd() - 0.5) * 70 };

      const sag = (44 + 165 * z) * (0.70 + rnd() * 0.55);
      const p1 = {
        x: W * (0.28 + rnd() * 0.08),
        y: yBase + sag * (0.55 + rnd() * 0.25),
      };
      const p2 = {
        x: W * (0.68 + rnd() * 0.08),
        y: yBase + sag * (0.55 + rnd() * 0.25),
      };

      strands.push({ id: si, p0, p1, p2, p3, z, seed: rnd() * 10 });

      const N = Math.max(6, Math.round(7 + z * 12 + rnd() * 2));
      for (let i = 0; i < N; i++) {
        const u = N === 1 ? 0.5 : i / (N - 1);

        const hue = huePalette[Math.floor(rnd() * huePalette.length)];
        const jitter = (rnd() - 0.5) * 0.035;

        bulbs.push({
          id: bulbs.length,
          si,
          u,
          z,
          r: baseR * (0.62 + z * 1.95) * (0.92 + rnd() * 0.22),
          hue,
          hueJ: jitter,
          on: 0,
          heat: 0,
          swing: (rnd() - 0.5) * 0.15,
          swingVel: 0,
          seed: rnd() * 10,
          jx: 0,
          jy: 0,
          jv: 0,
        });
      }
    }

    // Stable chase order: by strand then u
    const order = bulbs
      .slice()
      .sort((a, b) => a.si - b.si || a.u - b.u)
      .map((b) => b.id);

    // Pre-sorted draw order (back->front)
    const strandsSorted = strands.slice().sort((a, b) => a.z - b.z);
    const bulbsSorted = bulbs.slice().sort((a, b) => a.z - b.z);

    sceneRef.current = { W, H, VP, dust, stars };
    strandsRef.current = strands;
    bulbsRef.current = bulbs;
    orderRef.current = order;
    strandsSortedRef.current = strandsSorted;
    bulbsSortedRef.current = bulbsSorted;

    const L = order.length || 1;
    stateRef.current.chaseIdx = stateRef.current.chaseIdx % L;
    stateRef.current.drive = stateRef.current.drive % L;
  }, [rect.w, rect.h]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !rect.w || !rect.h) return;

    const W = rect.w;
    const H = rect.h;

    const dpr = Math.min(1.6, window.devicePixelRatio || 1);
    canvas.width = Math.floor(W * dpr);
    canvas.height = Math.floor(H * dpr);

    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const hsla = (h, s, l, a) =>
      `hsla(${Math.floor((((h % 1) + 1) % 1) * 360)}, ${s}%, ${l}%, ${a})`;

    if (!ctx.roundRect) {
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

    function getPoint(s, u, time, amp) {
      const it = 1 - u;
      const x =
        it ** 3 * s.p0.x +
        3 * it ** 2 * u * s.p1.x +
        3 * it * u ** 2 * s.p2.x +
        u ** 3 * s.p3.x;
      const y =
        it ** 3 * s.p0.y +
        3 * it ** 2 * u * s.p1.y +
        3 * it * u ** 2 * s.p2.y +
        u ** 3 * s.p3.y;

      const VP = sceneRef.current?.VP || { x: W * 0.5, y: H * 0.3 };
      const dx = (x - VP.x) * (s.z * 0.22);
      const dy = (y - VP.y) * (s.z * 0.22);

      const sway = Math.sin(time * 0.55 + s.seed) * (s.z * (2.0 + amp * 7.0));
      const micro = Math.sin(time * 1.15 + s.seed * 1.7) * (0.55 + 1.15 * s.z);

      return { x: x + dx + micro, y: y + dy + sway };
    }

    function spawnSparkle(st, x, y, hue, z, amp) {
      const s = 0.8 + Math.random() * (1.2 + amp * 1.3);
      st.sparkles.push({
        x,
        y,
        vx: (Math.random() - 0.5) * (18 + 52 * amp) * (0.55 + z),
        vy: -(22 + Math.random() * 62) * (0.7 + z * 0.6),
        life: 0.55 + Math.random() * 0.75,
        rot: Math.random() * 1.6,
        vr: (Math.random() - 0.5) * 2.0,
        s,
        hue,
      });
      if (st.sparkles.length > 120) st.sparkles.splice(0, st.sparkles.length - 120);
    }

    function drawDiamond(ctx2, x, y, s, rot, hue, a) {
      ctx2.globalCompositeOperation = "screen";
      ctx2.fillStyle = hsla(hue, 100, 60, 0.12 * a);
      ctx2.beginPath();
      ctx2.arc(x, y, s * 2.3, 0, Math.PI * 2);
      ctx2.fill();

      ctx2.save();
      ctx2.translate(x, y);
      ctx2.rotate(Math.PI / 4 + rot);
      ctx2.fillStyle = hsla(hue, 100, 92, 0.42 * a);
      ctx2.fillRect(-s, -s, s * 2, s * 2);
      ctx2.restore();

      ctx2.globalCompositeOperation = "source-over";
    }

    /** --- Bulb with MAGIC CORE --- */
    function drawBulbUltra(ctx2, x, y, r, intensity, hue, swing, z, playing) {
      ctx2.save();
      ctx2.translate(x, y);
      ctx2.rotate(swing);

      const I = clamp(intensity, 0, 1);
      const alphaZ = clamp(0.25 + z * 0.78, 0, 1);

      // socket
      const capW = r * 0.92;
      const capH = r * 0.72;
      ctx2.fillStyle = "rgba(6,6,8,0.95)";
      ctx2.roundRect(-capW / 2, -r * 1.28, capW, capH, 2.2);
      ctx2.fill();

      // ring sheen
      ctx2.globalCompositeOperation = "screen";
      const ring = ctx2.createLinearGradient(-capW / 2, 0, capW / 2, 0);
      ring.addColorStop(0, "rgba(255,255,255,0)");
      ring.addColorStop(0.35, "rgba(255,255,255,0.10)");
      ring.addColorStop(0.5, "rgba(255,255,255,0.18)");
      ring.addColorStop(0.65, "rgba(255,255,255,0.10)");
      ring.addColorStop(1, "rgba(255,255,255,0)");
      ctx2.fillStyle = ring;
      ctx2.fillRect(-capW / 2, -r * 1.02, capW, r * 0.14);
      ctx2.globalCompositeOperation = "source-over";

      // glass teardrop
      const rx = r * 0.92,
        ry = r * 1.52;

      ctx2.beginPath();
      ctx2.moveTo(0, -ry * 0.82);
      ctx2.bezierCurveTo(rx, -ry * 0.82, rx, ry * 0.46, 0, ry);
      ctx2.bezierCurveTo(-rx, ry * 0.46, -rx, -ry * 0.82, 0, -ry * 0.82);

      const glass = ctx2.createRadialGradient(0, -ry * 0.15, 0, 0, r * 0.2, ry);
      if (playing) {
        glass.addColorStop(0, hsla(hue, 100, 55, 0.12 * I * alphaZ));
        glass.addColorStop(0.35, hsla(hue, 100, 40, 0.07 * I * alphaZ));
        glass.addColorStop(1, "rgba(6,6,12,0.58)");
      } else {
        // ✅ colorless off state
        glass.addColorStop(0, "rgba(255,255,255,0.035)");
        glass.addColorStop(0.5, "rgba(120,140,190,0.03)");
        glass.addColorStop(1, "rgba(6,6,12,0.62)");
      }
      ctx2.fillStyle = glass;
      ctx2.fill();

      // highlight streak
      ctx2.globalCompositeOperation = "screen";
      ctx2.strokeStyle = "rgba(255,255,255,0.12)";
      ctx2.lineWidth = Math.max(0.7, 0.9 * z);
      ctx2.beginPath();
      ctx2.moveTo(-r * 0.25, -ry * 0.55);
      ctx2.quadraticCurveTo(r * 0.28, -ry * 0.10, -r * 0.05, ry * 0.55);
      ctx2.stroke();
      ctx2.globalCompositeOperation = "source-over";

      // subtle outline (still colorless)
      ctx2.globalCompositeOperation = "screen";
      ctx2.strokeStyle = "rgba(255,255,255,0.06)";
      ctx2.lineWidth = Math.max(0.6, 0.8 * z);
      ctx2.stroke();
      ctx2.globalCompositeOperation = "source-over";

      // MAGIC CORE + FILAMENT
      if (playing && I > 0.02) {
        ctx2.globalCompositeOperation = "screen";

        const coreR = r * (0.32 + I * 0.22);
        const core = ctx2.createRadialGradient(0, r * 0.35, 0, 0, r * 0.35, coreR * 4.2);
        core.addColorStop(0, hsla(hue, 100, 85, 0.95 * I));
        core.addColorStop(0.18, hsla(hue, 100, 65, 0.55 * I));
        core.addColorStop(0.55, hsla(hue, 100, 55, 0.18 * I));
        core.addColorStop(1, "rgba(0,0,0,0)");
        ctx2.fillStyle = core;
        ctx2.beginPath();
        ctx2.arc(0, r * 0.38, coreR * 2.1, 0, Math.PI * 2);
        ctx2.fill();

        // filament
        ctx2.shadowBlur = 28 * (0.5 + z) * (0.25 + I);
        ctx2.shadowColor = hsla(hue, 100, 60, 1);
        ctx2.strokeStyle = hsla(hue, 100, 88, 0.95 * I);
        ctx2.lineWidth = Math.max(1.4, 2.2 * z);

        ctx2.beginPath();
        const turns = 7;
        for (let k = 0; k <= turns; k++) {
          const tt = k / turns;
          const xx = (tt - 0.5) * r * 0.72;
          const yy = r * (0.18 + Math.sin(tt * Math.PI * 2 * turns) * 0.10);
          if (k === 0) ctx2.moveTo(xx, yy);
          else ctx2.lineTo(xx, yy);
        }
        ctx2.stroke();

        // spark point
        ctx2.shadowBlur = 34 * (0.4 + z) * I;
        ctx2.fillStyle = hsla(hue, 100, 90, 0.75 * I);
        ctx2.beginPath();
        ctx2.arc(0, r * 0.38, r * 0.08, 0, Math.PI * 2);
        ctx2.fill();

        ctx2.shadowBlur = 0;
        ctx2.globalCompositeOperation = "source-over";
      }

      ctx2.restore();
    }

    const draw = (t) => {
      const time = t * 0.001;
      const st = stateRef.current;
      const viz = vizRef.current || {};

      const playing = !!viz.playing;
      const amp = viz.amp || 0;
      const hit = !!viz.hit;
      const hitStrength = viz.hitStrength || 0;

      // ✅ follow audio time when possible
      const audioEl = audioRef?.current;
      const audioT = audioEl && isFinite(audioEl.currentTime) ? audioEl.currentTime : time;

      // punchier loudness proxy
      const loud = clamp(Math.pow(amp, 0.65), 0, 1);

      // --- Background: keep moody; don’t “wash” the wall with audio ---
      ctx.clearRect(0, 0, W, H);
      const g0 = ctx.createLinearGradient(0, 0, W, H);
      g0.addColorStop(0, "rgba(10,6,18,1)");
      g0.addColorStop(0.35, "rgba(46,10,36,1)");
      g0.addColorStop(0.65, "rgba(22,8,40,1)");
      g0.addColorStop(1, "rgba(6,6,14,1)");
      ctx.fillStyle = g0;
      ctx.fillRect(0, 0, W, H);

      // subtle nebula that barely responds (prevents “light coming off the wall”)
      const wallAmp = playing ? amp * 0.35 : 0; // hard cap influence
      ctx.globalCompositeOperation = "screen";
      const neb1 = ctx.createRadialGradient(W * 0.35, H * 0.35, 20, W * 0.35, H * 0.35, Math.max(W, H));
      neb1.addColorStop(0, `rgba(255,70,170,${0.05 + 0.05 * wallAmp})`);
      neb1.addColorStop(0.35, `rgba(120,90,255,${0.04 + 0.04 * wallAmp})`);
      neb1.addColorStop(1, "rgba(0,0,0,0)");
      ctx.fillStyle = neb1;
      ctx.fillRect(0, 0, W, H);

      const neb2 = ctx.createRadialGradient(W * 0.75, H * 0.65, 10, W * 0.75, H * 0.65, Math.max(W, H));
      neb2.addColorStop(0, `rgba(60,200,255,${0.02 + 0.04 * wallAmp})`);
      neb2.addColorStop(0.55, `rgba(160,140,255,${0.02 + 0.03 * wallAmp})`);
      neb2.addColorStop(1, "rgba(0,0,0,0)");
      ctx.fillStyle = neb2;
      ctx.fillRect(0, 0, W, H);
      ctx.globalCompositeOperation = "source-over";

      const scene = sceneRef.current;
      if (!scene) {
        rafRef.current = requestAnimationFrame(draw);
        return;
      }

      // Dust/bokeh + stars
      ctx.globalCompositeOperation = "screen";
      for (const p of scene.dust) {
        const wob = Math.sin(time * p.tw + p.ph) * (0.6 + 2.0 * wallAmp) * (0.25 + p.z);
        const px = p.x + wob;
        const py = p.y + Math.cos(time * 0.6 + p.ph) * (0.4 + wallAmp * 1.4) * (0.25 + p.z);
        ctx.fillStyle = hsla(p.hue, 100, 62, p.a * (0.85 + 0.45 * wallAmp));
        ctx.beginPath();
        ctx.arc(px, py, p.r, 0, Math.PI * 2);
        ctx.fill();
      }
      for (const s of scene.stars) {
        const tw = 0.6 + 0.4 * Math.sin(time * 1.2 + s.ph);
        ctx.fillStyle = `rgba(255,255,255,${s.a * tw})`;
        ctx.beginPath();
        ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.globalCompositeOperation = "source-over";

      // --- Loudness-synced chase (locks to audio time) ---
      if (st.cooldown > 0) st.cooldown -= 1;

      const order = orderRef.current;
      const bulbsAll = bulbsRef.current;
      const L = order?.length ? order.length : Math.max(1, bulbsAll.length);

      if (playing) {
        const baseBps = 2;
        const loudBps = 4 * loud;
        const hitBps = hit ? 7.0 * (0.35 + hitStrength) : 0;

        const last = st.lastAudioT ?? audioT;
        let dt = audioT - last;
        dt = clamp(dt, 0, 0.075);
        st.lastAudioT = audioT;

        const bps = baseBps + loudBps + hitBps;
        st.drive += bps * dt;

        const nextIdx = Math.floor(st.drive) % L;
        if (nextIdx !== st.chaseIdx) {
          st.chaseIdx = nextIdx;
          st.cooldown = hit ? 1 : 0;
        }
      } else {
        st.lastAudioT = audioT;
      }

      const activeId = playing
        ? order?.length
          ? order[st.chaseIdx % order.length]
          : bulbsAll[st.chaseIdx % Math.max(1, bulbsAll.length)]?.id ?? -1
        : -1;

      // --- Wires (back->front) ---
      const strandsSorted = strandsSortedRef.current;
      for (const s of strandsSorted) {
        const aZ = clamp(0.08 + s.z * 0.9, 0, 1);
        ctx.strokeStyle = `rgba(8,8,12,${aZ})`;
        ctx.lineWidth = 3.6 * s.z;

        ctx.beginPath();
        for (let u = 0; u <= 1.00001; u += 0.05) {
          const p = getPoint(s, u, time, wallAmp);
          if (u === 0) ctx.moveTo(p.x, p.y);
          else ctx.lineTo(p.x, p.y);
        }
        ctx.stroke();

        if (playing) {
          ctx.globalCompositeOperation = "screen";
          ctx.strokeStyle = `rgba(255,255,255,${0.006 + 0.018 * wallAmp * s.z})`;
          ctx.lineWidth = 1.0 * s.z;
          ctx.stroke();
          ctx.globalCompositeOperation = "source-over";
        }
      }

      // --- Bulbs (back->front) ---
      const bulbsSorted = bulbsSortedRef.current;

      for (const b of bulbsSorted) {
        const s = strandsRef.current[b.si];
        if (!s) continue;

        const p0 = getPoint(s, b.u, time, wallAmp);
        const isActive = b.id === activeId;

        let target = 0;
        if (playing && isActive) {
          const breath = smoothstep(0.02, 0.20, loud) * 0.95;
          const flash = hit ? 0.55 * smoothstep(0.05, 0.35, hitStrength) : 0;
          const shimmer = 0.10 * (0.5 + 0.5 * Math.sin(audioT * 9.0 + b.seed));
          const floor = 0.18;
          target = clamp(floor + breath + flash + shimmer, 0, 1);
        }

        b.on = lerp(b.on, target, 0.20);
        b.heat = lerp(b.heat, Math.max(b.heat, b.on), 0.08);
        b.heat *= playing ? 0.990 : 0.960;

        // deterministic micro motion (NO Math.random)
        const hop = isActive ? b.on : 0;
        b.jv = lerp(b.jv, hop, 0.12);
        const n1 = noise2(time * 6.0, b.seed * 11.7);
        const n2 = noise2(time * 6.7 + 10.0, b.seed * 19.3);
        b.jx = lerp(b.jx, (Math.sin(time * 10 + b.seed) * 1.6 + n1 * 1.2) * b.jv, 0.10);
        b.jy = lerp(b.jy, (Math.cos(time * 9 + b.seed) * 1.2 + n2 * 1.0) * b.jv, 0.10);

        const px = p0.x + b.jx * (1 + b.z * 0.7);
        const py = p0.y + b.jy * (1 + b.z * 0.7);

        const hue = b.hue + b.hueJ + (playing ? 0.02 * Math.sin(time * 0.6 + b.seed) : 0);

        const shown = playing && isActive ? clamp(0.35 + b.on * 1.9, 0, 1) : 0;

        // BIG halo ONLY for active bulb
        if (playing && isActive && shown > 0.06) {
          ctx.save();
          ctx.globalCompositeOperation = "screen";
          const glowR = b.r * (18.0 + 14.0 * b.z) * (0.55 + shown * 1.6);
          const g = ctx.createRadialGradient(px, py, 0, px, py, glowR);
          g.addColorStop(0, hsla(hue, 100, 65, 0.34 * shown * (0.85 + b.z)));
          g.addColorStop(0.35, hsla(hue, 100, 55, 0.16 * shown * (0.85 + b.z)));
          g.addColorStop(1, "rgba(0,0,0,0)");
          ctx.fillStyle = g;
          ctx.fillRect(px - glowR, py - glowR, glowR * 2, glowR * 2);
          ctx.restore();
        }

        if (playing && isActive && shown > 0.18 && Math.random() < 0.09) {
          spawnSparkle(st, px + (Math.random() - 0.5) * 14, py + (Math.random() - 0.5) * 10, hue, b.z, wallAmp);
        }

        const swingTarget =
          Math.sin(time * (0.85 + b.z * 0.22) + b.seed) * (0.06 + 0.11 * b.z) * (0.25 + wallAmp);
        b.swingVel = lerp(b.swingVel, (swingTarget - b.swing) * 0.35, 0.08);
        b.swing += b.swingVel;

        drawBulbUltra(ctx, px, py, b.r * 1.15, shown, hue, b.swing, b.z, playing);
      }

      // sparkle overlay
      if (st.sparkles.length) {
        const dt = 1 / 60;
        for (let i = st.sparkles.length - 1; i >= 0; i--) {
          const p = st.sparkles[i];
          p.life -= dt;
          p.x += p.vx * dt;
          p.y += p.vy * dt;
          p.rot += p.vr * dt;

          if (p.life <= 0 || p.y < -60 || p.x < -90 || p.x > W + 90) {
            st.sparkles.splice(i, 1);
            continue;
          }
          const a = clamp(p.life, 0, 1);
          drawDiamond(ctx, p.x, p.y, p.s, p.rot, p.hue, a);
        }
      }

      rafRef.current = requestAnimationFrame(draw);
    };

    rafRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(rafRef.current);
  }, [rect.w, rect.h, vizRef, audioRef]);

  return (
    <div
      className="ragLights"
      ref={wrapRef}
      style={{
        background: "#000",
        height: "100%",
        overflow: "hidden",
        position: "relative",
      }}
      aria-hidden="true"
    >
      <canvas ref={canvasRef} style={{ display: "block", width: "100%", height: "100%" }} />

      <div
        style={{
          position: "absolute",
          top: 10,
          left: 10,
          padding: "6px 10px",
          borderRadius: 999,
          fontSize: 12,
          letterSpacing: 0.3,
          color: "rgba(233,236,255,0.78)",
          background: "rgba(0,0,0,0.35)",
          border: "1px solid rgba(255,255,255,0.10)",
          backdropFilter: "blur(6px)",
          WebkitBackdropFilter: "blur(6px)",
          userSelect: "none",
          pointerEvents: "none",
        }}
      >
        <span
          style={{
            display: "inline-block",
            width: 9,
            height: 9,
            marginRight: 8,
            transform: "rotate(45deg)",
            background: "rgba(160,140,255,0.55)",
            boxShadow: "0 0 14px rgba(160,140,255,0.35)",
          }}
        />
        {label}
      </div>
    </div>
  );
}

/** ---------- Audio UI ---------- **/
function useAudioUI(audioRef) {
  const [playing, setPlaying] = useState(false);
  const [t, setT] = useState(0);
  const [dur, setDur] = useState(0);

  useEffect(() => {
    const el = audioRef.current;
    if (!el) return;

    const onPlay = () => setPlaying(true);
    const onPause = () => setPlaying(false);
    const onTime = () => setT(el.currentTime || 0);
    const onMeta = () => setDur(el.duration || 0);

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
  }, [audioRef]);

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

  return { playing, t, dur, pct, toggle, seek };
}
function mmss(sec) {
  if (!isFinite(sec)) return "0:00";
  const s = Math.max(0, Math.floor(sec));
  const m = Math.floor(s / 60);
  const r = s % 60;
  return `${m}:${String(r).padStart(2, "0")}`;
}

/** ---------- Main RAG screen ---------- **/
export default function Rag() {
  const fileRef = useRef(null);

  const [backendOk, setBackendOk] = useState(false);
  const [readyInfo, setReadyInfo] = useState(null);
  const ready = readyInfo && typeof readyInfo.ready === "boolean" ? readyInfo.ready : null;

  const [targetDur, setTargetDur] = useState(20);
  const [topK, setTopK] = useState(5);

  const [uploadId, setUploadId] = useState(null);
  const [uploadMeta, setUploadMeta] = useState(null);
  const [inputAnalysis, setInputAnalysis] = useState(null);

  const [runs, setRuns] = useState(() => loadRuns());
  const [selectedRunId, setSelectedRunId] = useState(null);

  const [results, setResults] = useState([]);
  const [activeIdx, setActiveIdx] = useState(0);

  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("Idle");

  const inputAudioRef = useRef(null);
  const recoAudioRef = useRef(null);
  const inputUI = useAudioUI(inputAudioRef);
  const recoUI = useAudioUI(recoAudioRef);

  const [activeViz, setActiveViz] = useState("none"); // input | reco | none
  const activeAudioRef = useMemo(() => {
    if (activeViz === "input") return inputAudioRef;
    if (activeViz === "reco") return recoAudioRef;
    return { current: null };
  }, [activeViz]);

  const inputUrl = useMemo(() => (uploadId ? `${API}/rag/${uploadId}/files/input.wav` : ""), [uploadId]);
  const activeReco = useMemo(
    () => (results?.length ? results[clamp(activeIdx, 0, results.length - 1)] : null),
    [results, activeIdx]
  );
  const recoUrl = useMemo(() => (activeReco?.extension_wav_url ? `${API}${activeReco.extension_wav_url}` : ""), [
    activeReco,
  ]);

  // ✅ Preload reco audio as soon as selection changes (so Play is instant)
  useEffect(() => {
    const el = recoAudioRef.current;
    if (!el) return;

    if (!recoUrl) {
      el.removeAttribute("src");
      return;
    }

    // IMPORTANT: set crossOrigin before src if needed
    try { el.crossOrigin = "anonymous"; } catch {}

    // Only reload if the URL actually changed
    if (el.src !== recoUrl) {
      el.src = recoUrl;
      el.preload = "auto";
      el.load(); // kicks off fetch+decode now, not on click
    }
  }, [recoUrl]);

  const [recoAnalysisByUrl, setRecoAnalysisByUrl] = useState({});
  const recoAnalysis = useMemo(() => (recoUrl ? recoAnalysisByUrl[recoUrl] || null : null), [recoUrl, recoAnalysisByUrl]);

  const palette = useMemo(
    () => ({
      line: "rgba(233,236,255,0.88)",
      fill: "rgba(233,236,255,0.06)",
      glow: "rgba(165,210,255,0.26)",
      glowSoft: "rgba(165,210,255,0.16)",
      point: "rgba(255,255,255,0.92)",
    }),
    []
  );

  const kpiMain = useMemo(() => {
    const s = inputAnalysis?.scores || {};
    return {
      energy: s.Energy ?? null,
      dynamics: s.Dynamics ?? null,
      complexity: s.Complexity ?? null,
      duration: inputAnalysis?.duration ?? uploadMeta?.duration_sec ?? null,
    };
  }, [inputAnalysis, uploadMeta]);

  const kpiTiny = useMemo(() => {
    const sr = uploadMeta?.sample_rate ?? inputAnalysis?.sample_rate ?? null;
    const frames = inputAnalysis?.frames ?? null;
    return [
      { k: "Upload ID", v: uploadId ?? null },
      { k: "SR", v: sr ?? null },
      { k: "Target", v: `${Number(targetDur)}s` },
      { k: "Top-K", v: String(topK) },
      { k: "Frames", v: frames ?? null },
    ].filter((x) => x.v !== null && x.v !== undefined && x.v !== "");
  }, [uploadId, uploadMeta, inputAnalysis, targetDur, topK]);

  const marqueeRuns = useMemo(() => (runs?.length ? runs.concat(runs) : []), [runs]);

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
  }, []);

  useEffect(() => {
    const a = inputAudioRef.current;
    const b = recoAudioRef.current;
    if (!a && !b) return;

    const onInputPlay = () => setActiveViz("input");
    const onRecoPlay = () => setActiveViz("reco");

    const onStop = () => {
      const ap = a && !a.paused && !a.ended;
      const bp = b && !b.paused && !b.ended;
      if (!ap && !bp) setActiveViz("none");
    };

    a?.addEventListener("play", onInputPlay);
    b?.addEventListener("play", onRecoPlay);

    a?.addEventListener("pause", onStop);
    b?.addEventListener("pause", onStop);
    a?.addEventListener("ended", onStop);
    b?.addEventListener("ended", onStop);

    return () => {
      a?.removeEventListener("play", onInputPlay);
      b?.removeEventListener("play", onRecoPlay);
      a?.removeEventListener("pause", onStop);
      b?.removeEventListener("pause", onStop);
      a?.removeEventListener("ended", onStop);
      b?.removeEventListener("ended", onStop);
    };
  }, [inputUrl, recoUrl]);

  async function fetchInputAnalysis(id) {
    if (!id) return;
    try {
      const r = await fetch(`${API}/rag/${id}/analysis`);
      if (!r.ok) throw new Error(await r.text());
      const j = await r.json();
      setInputAnalysis(j);
    } catch {
      setInputAnalysis(null);
    }
  }

  async function uploadWav(file) {
    setBusy(true);
    setStatus("Uploading…");
    setResults([]);
    setActiveIdx(0);
    setRecoAnalysisByUrl({});
    setInputAnalysis(null);

    try {
      const fd = new FormData();
      fd.append("file", file);
      const r = await fetch(`${API}/rag/upload`, { method: "POST", body: fd });
      if (!r.ok) throw new Error(await r.text());
      const j = await r.json();

      setUploadId(j.upload_id);
      setUploadMeta(j);
      setSelectedRunId(j.upload_id);
      setStatus(`Uploaded: ${j.filename}`);

      const entry = {
        upload_id: j.upload_id,
        filename: j.filename,
        created_at: new Date().toISOString(),
        target_s: Number(targetDur),
        top_k: Number(topK),
      };
      setRuns(addRunEntry(entry));

      await fetchInputAnalysis(j.upload_id);

      requestAnimationFrame(() => {
        if (inputAudioRef.current) {
          inputAudioRef.current.src = `${API}/rag/${j.upload_id}/files/input.wav`;
        }
        requestAnimationFrame(() => {
          const el = inputAudioRef.current;
          if (!el) return;
          el.preload = "auto";
          el.load();
        });
      });
    } catch (e) {
      setStatus(String(e?.message || e));
      setUploadId(null);
      setUploadMeta(null);
      setSelectedRunId(null);
    } finally {
      setBusy(false);
    }
  }

  async function stitch() {
    if (!uploadId) return;
    setBusy(true);
    setStatus("Stitching…");
    setResults([]);
    setActiveIdx(0);
    setRecoAnalysisByUrl({});

    try {
      const r = await fetch(`${API}/rag/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          upload_id: uploadId,
          target_duration_sec: Number(targetDur),
          top_k: Number(topK),
        }),
      });
      if (!r.ok) throw new Error(await r.text());
      const j = await r.json();
      const arr = Array.isArray(j.results) ? j.results : [];
      setResults(arr);
      setStatus(arr.length ? "Recommendations ready ✦" : "No matches found");
    } catch (e) {
      setStatus(String(e?.message || e));
    } finally {
      setBusy(false);
    }
  }

  function selectRun(run) {
    if (!run?.upload_id) return;
    setSelectedRunId(run.upload_id);

    if (run.upload_id === uploadId) return;

    setUploadId(run.upload_id);
    setUploadMeta({ filename: run.filename });
    setResults([]);
    setActiveIdx(0);
    setRecoAnalysisByUrl({});
    setStatus("Loaded prior run (input only)…");
    fetchInputAnalysis(run.upload_id);

    requestAnimationFrame(() => {
      if (inputAudioRef.current) {
        inputAudioRef.current.src = `${API}/rag/${run.upload_id}/files/input.wav`;
      }
    });
  }

  async function playReco() {
    if (!recoUrl) return;
    try {
      const el = recoAudioRef.current;
      if (!el) return;

      // src already preloaded by the effect above
      await el.play();

      // analysis fetch (FIX: use existing backend route /rag/{upload_id}/analysis?file=...)
      if (!recoAnalysisByUrl[recoUrl]) {
        try {
          // Prefer the URL your backend already includes in results (best)
          let analysisUrl = activeReco?.analysis_extension_url || null;

          // Fallback: derive filename from recoUrl and hit /rag/{uploadId}/analysis?file=...
          if (!analysisUrl && uploadId && recoUrl) {
            const fname = decodeURIComponent(recoUrl.split("/").pop() || "");
            if (fname) analysisUrl = `/rag/${uploadId}/analysis?file=${encodeURIComponent(fname)}`;
          }

          if (analysisUrl) {
            const r = await fetch(`${API}${analysisUrl}`);
            if (r.ok) {
              const j = await r.json();
              setRecoAnalysisByUrl((m) => ({ ...m, [recoUrl]: j }));
            }
          }
        } catch {}
      }
    } catch (e) {
      setStatus(String(e?.message || e));
    }
  }

  const canStitch = !!uploadId && !busy && backendOk && ready !== false;

  return (
    <div className="shellRag">
      <aside className="sidebar">
        <div className="brandBar">
          <div className="logoMark">✦</div>
          <div>
            <div className="brandTitle">RAG</div>
            <div className="brandSub">Stitch Mode</div>
          </div>
        </div>

        <div className="statusPills">
          <span className={`pill ${backendOk ? "pillOk" : "pillBad"}`}>
            {backendOk ? "Backend OK" : "Backend Down"}
          </span>
          {ready !== null && (
            <span className={`pill ${ready ? "pillOk" : "pillWarn"}`}>{ready ? "Ready" : "Not Ready"}</span>
          )}
        </div>

        <div className="controlCard">
          <div className="controlTitle">New Stitch</div>

          <div className="controlRow">
            <label className="field">
              <span className="label">Target (seconds)</span>
              <input
                className="input inputTight"
                type="number"
                min="4"
                max="240"
                step="1"
                value={targetDur}
                onChange={(e) => setTargetDur(Number(e.target.value))}
                disabled={busy}
              />
            </label>

            <label className="field">
              <span className="label">Top-K</span>
              <input
                className="input inputTight"
                type="number"
                min="1"
                max="12"
                step="1"
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
                disabled={busy}
              />
            </label>
          </div>

          <div className="buttonRow">
            <button className="buttonSecondary buttonTight" onClick={() => fileRef.current?.click()} disabled={busy}>
              Upload WAV
            </button>
            <button className="button buttonTight" onClick={stitch} disabled={!canStitch}>
              {busy ? "Working…" : "Stitch ✦"}
            </button>

            <input
              ref={fileRef}
              className="fileHidden"
              type="file"
              accept=".wav,audio/wav"
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) uploadWav(f);
              }}
              disabled={busy}
            />
          </div>

          <div className="helperRow">
            <span className="helper">Upload first</span>
            <span className="helper">then Stitch</span>
          </div>
        </div>

        <div className="runsHeader">
          <div className="runsTitle">Past Runs</div>
          <div className="runsHint">marquee • hover pauses</div>
        </div>

        <div className="runsMarquee" title="Hover to pause">
          {runs.length === 0 ? (
            <div className="runsEmpty">No runs yet. Upload a WAV.</div>
          ) : (
            <div className="marqueeInner">
              {marqueeRuns.map((r, i) => {
                const id = r.upload_id || `run_${i}`;
                const active = id === selectedRunId;
                return (
                  <button
                    key={`${id}_${i}`}
                    className={`runItem ${active ? "runItemActive" : ""}`}
                    onClick={() => selectRun(r)}
                  >
                    <div className="runTop">
                      <div className="runId">{id}</div>
                      <div className="runSec">{r.target_s ? `${r.target_s}s` : ""}</div>
                    </div>
                    <div className="runMeta">{r.filename ? `File: ${niceName(r.filename)}` : ""}</div>
                    <div className="runTime">{r.created_at ? new Date(r.created_at).toLocaleString() : ""}</div>
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

      <main className="main">
        <section className="kpiStrip">
          <div className="kpiStripHead">
            <div className="kpiStripTitle">
              <div className="kpiStripTitleLabel">Uploaded</div>
              <div className="kpiStripTitleValue">{uploadMeta?.filename ? niceName(uploadMeta.filename) : "—"}</div>
            </div>

            <div className="kpiStatusPill">
              <span className="diamondGlow" />
              <span className="statusLine">{status || "Idle"}</span>
            </div>
          </div>

          <div className="kpiMain kpiMainPrimary">
            <div className="kpiMini kpiMiniPrimary">
              <div className="kpiLabel">Energy</div>
              <div className="kpiValue">{kpiMain.energy ?? "—"}</div>
            </div>
            <div className="kpiMini kpiMiniPrimary">
              <div className="kpiLabel">Dynamics</div>
              <div className="kpiValue">{kpiMain.dynamics ?? "—"}</div>
            </div>
            <div className="kpiMini kpiMiniPrimary">
              <div className="kpiLabel">Complexity</div>
              <div className="kpiValue">{kpiMain.complexity ?? "—"}</div>
            </div>
            <div className="kpiMini kpiMiniPrimary">
              <div className="kpiLabel">Duration</div>
              <div className="kpiValue">{kpiMain.duration != null ? `${fmt(kpiMain.duration, 2)}s` : "—"}</div>
            </div>
          </div>

          <div className="kpiPills kpiPillsOneLine">
            {kpiTiny.map((it) => (
              <div key={it.k} className="kpiPill kpiPillTiny" title={`${it.k}: ${it.v}`}>
                <span className="kpiPillK">{it.k}</span>
                <span className="kpiPillV">{String(it.v)}</span>
              </div>
            ))}
          </div>
        </section>

        <section className="contentGridSingle">
          <div className="panel panelTight">
            <div className="panelHeaderTight">
              <div className="panelTitle">Input</div>
              <div className="panelHintTight">single player • no stems</div>
            </div>

            <div className="playerMini">
              <div className="playerMiniTop">
                <div className="playerMiniSub">{uploadMeta?.filename ? niceName(uploadMeta.filename) : "No file"}</div>
              </div>

              <audio ref={inputAudioRef} src={inputUrl || ""} controls className="nativeAudioHidden" />

              {uploadId ? (
                <div className="playerMiniUI">
                  <button
                    className={`playBtn ${inputUI.playing ? "isPlaying" : ""}`}
                    onClick={inputUI.toggle}
                    aria-label="Play/Pause"
                    type="button"
                  >
                    <span className="playIcon" />
                  </button>

                  <input
                    className="range timeline"
                    type="range"
                    min="0"
                    max="1"
                    step="0.001"
                    value={inputUI.pct}
                    onChange={(e) => inputUI.seek(Number(e.target.value))}
                    style={{ "--pct": inputUI.pct }}
                  />

                  <div className="timeRowSm">
                    <span className="timeText">{mmss(inputUI.t)}</span>
                    <span className="timeText">{mmss(inputUI.dur)}</span>
                  </div>

                  <div className="downloadRowSm">
                    <a className="linkBtnSm" href={inputUrl} download>
                      Download input.wav
                    </a>
                  </div>
                </div>
              ) : (
                <div className="empty">Upload a WAV to enable playback.</div>
              )}
            </div>

            <div className="vizStack">
              <CanvasLineChart
                title="Loudness"
                x={inputAnalysis?.rms?.t || []}
                y={inputAnalysis?.rms?.y || []}
                yLabel="RMS"
                height={150}
                palette={palette}
                sparkle={true}
              />
            </div>

            {!results.length ? (
              <div className="premiumBuffer">
                <div className="premiumIcon" />
                <div className="premiumText">
                  <div className="premiumTitle">Premium Buffer</div>
                  <div className="premiumSub">Stitched audio will load here once you hit “Stitch ✦”.</div>
                </div>
              </div>
            ) : null}
          </div>

          {results.length ? (
            <div className="panel panelTight">
              <div className="panelHeaderTight">
                <div className="panelTitle">Top-K Recommendations</div>
                <div className="panelHintTight">scroll or use arrows • play + score</div>
              </div>

              <div className="recoToolbar">
                <button
                  className="chipBtn"
                  onClick={() => setActiveIdx((p) => (p - 1 + results.length) % results.length)}
                  type="button"
                >
                  Prev
                </button>
                <div className="chip">{`${activeIdx + 1} / ${results.length}`}</div>
                <button className="chipBtn" onClick={() => setActiveIdx((p) => (p + 1) % results.length)} type="button">
                  Next
                </button>

                <div className="spacer" />

                <button className="chipBtn strong" onClick={playReco} disabled={!recoUrl} type="button">
                  ▶ Play
                </button>
                <div className="chip scoreChip">Score: {activeReco?.score != null ? fmt(activeReco.score, 4) : "—"}</div>

                {recoUrl && (
                  <a className="chipBtn" href={recoUrl} download>
                    Download
                  </a>
                )}
              </div>

              <audio ref={recoAudioRef} src={recoUrl || ""} className="nativeAudioHidden" />

              <div
                className="recoRail"
                onScrollCapture={(e) => {
                  const el = e.currentTarget;
                  const cardW = el.firstElementChild?.getBoundingClientRect?.().width || 1;
                  const idx = Math.round(el.scrollLeft / Math.max(1, cardW + 12));
                  setActiveIdx(clamp(idx, 0, results.length - 1));
                }}
              >
                {results.map((r, i) => {
                  const url = r?.extension_wav_url ? `${API}${r.extension_wav_url}` : "";
                  const active = i === activeIdx;
                  return (
                    <button
                      key={`${r.rank || i}-${i}`}
                      className={`recoCard ${active ? "recoCardActive" : ""}`}
                      onClick={() => setActiveIdx(i)}
                      type="button"
                    >
                      <div className="recoTop">
                        <div className="recoTitle">Rank {r.rank ?? i + 1}</div>
                        <div className="recoTiny">{r.db_id_4s != null ? `DB ${r.db_id_4s}` : ""}</div>
                      </div>

                      <div className="recoMid">
                        <div className="recoScore">Score</div>
                        <div className="recoScoreVal">{r.score != null ? fmt(r.score, 4) : "—"}</div>
                      </div>

                      <div className="recoMid">
                        <div className="recoScore">Coherence</div>
                        <div className="recoScoreVal">{r.coherence != null ? fmt(r.coherence, 4) : "—"}</div>
                      </div>

                      <div className="recoMid">
                        <div className="recoScore">Relevance</div>
                        <div className="recoScoreVal">{r.relevance != null ? fmt(r.relevance, 4) : "—"}</div>
                      </div>

                      <div className="recoActions">
                        <span className="recoUrl">{url ? "extension.wav" : "—"}</span>
                        <span className="recoHint">tap to select</span>
                      </div>
                    </button>
                  );
                })}
              </div>

              <div className="recoPlayerBox">
                <div className="recoPlayerHead">
                  <div className="recoPlayerTitle">
                    Selected: <b>{activeReco?.rank != null ? `Rank ${activeReco.rank}` : "—"}</b>
                  </div>
                  <div className="recoPlayerHint">{recoUI.playing ? "Playing" : "Idle"}</div>
                </div>

                <div className="playerMini playerMiniReco">
                  <div className="playerMiniTop">
                    <div className="playerMiniSub">{recoUrl ? "extension.wav" : "No selection"}</div>
                  </div>

                  <div className="playerMiniUI">
                    <button
                      className={`playBtn ${recoUI.playing ? "isPlaying" : ""}`}
                      onClick={async () => {
                        if (!recoUrl) return;
                        if (recoAudioRef.current?.paused) await playReco();
                        else recoAudioRef.current?.pause();
                      }}
                      aria-label="Play/Pause"
                      type="button"
                      disabled={!recoUrl}
                    >
                      <span className="playIcon" />
                    </button>

                    <input
                      className="range timeline"
                      type="range"
                      min="0"
                      max="1"
                      step="0.001"
                      value={recoUI.pct}
                      onChange={(e) => recoUI.seek(Number(e.target.value))}
                      style={{ "--pct": recoUI.pct }}
                      disabled={!recoUrl}
                    />

                    <div className="timeRowSm">
                      <span className="timeText">{mmss(recoUI.t)}</span>
                      <span className="timeText">{mmss(recoUI.dur)}</span>
                    </div>

                    <div className="downloadRowSm">
                      {recoUrl ? (
                        <a className="linkBtnSm" href={recoUrl} download>
                          Download extension.wav
                        </a>
                      ) : (
                        <span className="helper">Select a recommendation</span>
                      )}
                    </div>
                  </div>
                </div>

                <div className="vizStack">
                  <CanvasLineChart
                    title="Loudness (Reco)"
                    x={recoAnalysis?.rms?.t || []}
                    y={recoAnalysis?.rms?.y || []}
                    yLabel="RMS"
                    height={140}
                    palette={palette}
                    sparkle={true}
                  />
                </div>
              </div>
            </div>
          ) : null}
        </section>
      </main>

      <aside className="tesseractRag">
        <div className="tessInset">
          <LightsWall
            audioRef={activeAudioRef}
            label="Tesseract • mix"
            rms={activeViz === "input" ? inputAnalysis?.rms : activeViz === "reco" ? recoAnalysis?.rms : null}
          />
        </div>
      </aside>
    </div>
  );
}