// Rag.jsx  (LAYOUT FIX: LightWall truly top • no black band • right column smaller • single reco block • no reco curve)
// Drop-in replacement for your current Rag.jsx.
// What changed:
// ✅ Middle column is a clean vertical stack: LightWall -> KPIs -> Input Player -> Loudness -> Premium Buffer
// ✅ LightWall panel height is controlled by CSS (no extra black area)
// ✅ Right column: smaller width + ONLY the reco list (no reco loudness chart)
// ✅ Keeps your LightsWall + analyser logic intact (only layout + small cleanup)

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
function fmt(n, digits = 2) {
  if (n === null || n === undefined || Number.isNaN(Number(n))) return "—";
  return Number(n).toFixed(digits);
}
function niceName(filename) {
  if (!filename) return "";
  return filename.replace(/_/g, " ").replace(/\.wav$/i, "");
}

function clampInt(n, a, b) {
  const x = Math.trunc(Number(n));
  if (!Number.isFinite(x)) return a;
  return Math.max(a, Math.min(b, x));
}

function asNumOr(fallback, v) {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

function once(el, ev) {
  return new Promise((res) => {
    const h = () => {
      el.removeEventListener(ev, h);
      res();
    };
    el.addEventListener(ev, h, { once: true });
  });
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
  const next = [
    { ...entry }, // newest first
    ...prev
      .filter((x) => x.upload_id !== entry.upload_id)
      .map((x) => ({ ...x })),
  ];
  saveRuns(next);
  return next;
}

// NEW: attach results to an existing run (or create if missing)
function upsertRunResults(upload_id, patch) {
  const prev = loadRuns();
  let found = false;

  const next = prev.map((r) => {
    if (r.upload_id !== upload_id) return r;
    found = true;
    return { ...r, ...patch };
  });

  if (!found) next.unshift({ upload_id, ...patch });

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
  height = 150,
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

    const dpr = Math.min(1.6, window.devicePixelRatio || 1);
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
function hash11(x) {
  const s = Math.sin(x * 127.1) * 43758.5453123;
  return s - Math.floor(s);
}
function noise2(t, seed) {
  return hash11(t + seed) * 2 - 1;
}

/** ---------- Stranger Lights Canvas (FIXED: no black patches • nebula kept • correct draw order/composites) ---------- **/
function LightsWall({ audioRef, label = "Tesseract • mix" }) {
  const wrapRef = useRef(null);
  const canvasRef = useRef(null);
  const rafRef = useRef(0);
  const [rect, setRect] = useState({ w: 0, h: 0 });

  const vizRef = useStableAudioAnalyserRef(audioRef);

  const sceneRef = useRef(null);
  const bulbsRef = useRef([]);
  const strandsRef = useRef([]);
  const orderRef = useRef([]);

  const strandBulbIdsRef = useRef([]); // [ [bulbId, bulbId, ...], ... ] per strand

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

  useEffect(() => {
    if (!rect.w || !rect.h) return;

    const W = rect.w;
    const H = rect.h;
    const VP = { x: W * 0.52, y: H * 0.28 };

    const seed = stateRef.current.seed;
    const rnd = mulberry32(seed);

    const Z_LAYERS = [0.10, 0.18, 0.30, 0.46, 0.66, 0.96, 1.34];
    const strandCount = 15;

    const strands = [];
    const bulbs = [];
    const baseR = Math.max(1.45, Math.min(2.25, Math.min(W, H) / 520));

    const huePalette = [0.0, 0.08, 0.16, 0.34, 0.5, 0.62, 0.78, 0.86];

    // --- background “nebula” particles (PRECOMPUTED, no per-frame Math.random => no flicker/patched areas)
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

    // “film grain” speckle, subtle, stable
    const speck = [];
    const speckN = Math.floor((W * H) / 5200);
    for (let i = 0; i < speckN; i++) {
      speck.push({
        x: rnd() * W,
        y: rnd() * H,
        r: 0.35 + rnd() * 0.9,
        a: 0.018 + rnd() * 0.04,
        ph: rnd() * 10,
      });
    }

    // Nebula “cloud blobs” (a handful of big, stable radial gradients)
    const clouds = [];
    const cloudN = 9;
    for (let i = 0; i < cloudN; i++) {
      const z = 0.2 + rnd() * 1.1;
      const h = huePalette[Math.floor(rnd() * huePalette.length)];
      clouds.push({
        x: (rnd() * 1.2 - 0.1) * W,
        y: (rnd() * 1.2 - 0.1) * H,
        r: (0.42 + rnd() * 0.85) * Math.max(W, H),
        hue: h,
        a: 0.08 + rnd() * 0.12,
        z,
        ph: rnd() * 10,
        tw: 0.35 + rnd() * 0.9,
      });
    }

    for (let si = 0; si < strandCount; si++) {
      const z = Z_LAYERS[si % Z_LAYERS.length];

      const yBase = H * (0.14 + rnd() * 0.72);
      const p0 = { x: -60, y: yBase + (rnd() - 0.5) * 40 };
      const p3 = { x: W + 60, y: yBase + (rnd() - 0.5) * 70 };

      const sag = (44 + 165 * z) * (0.7 + rnd() * 0.55);
      const p1 = { x: W * (0.28 + rnd() * 0.08), y: yBase + sag * (0.55 + rnd() * 0.25) };
      const p2 = { x: W * (0.68 + rnd() * 0.08), y: yBase + sag * (0.55 + rnd() * 0.25) };

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

    // Build per-strand bulb id lists (sorted left->right by u)
    const perStrand = Array.from({ length: strandCount }, () => []);
    for (const b of bulbs) perStrand[b.si].push(b);
    for (let si = 0; si < perStrand.length; si++) {
      perStrand[si].sort((a, b) => a.u - b.u);
      perStrand[si] = perStrand[si].map((bb) => bb.id);
    }
    strandBulbIdsRef.current = perStrand;

    const order = bulbs
      .slice()
      .sort((a, b) => a.si - b.si || a.u - b.u)
      .map((b) => b.id);

    const strandsSorted = strands.slice().sort((a, b) => a.z - b.z);
    const bulbsSorted = bulbs.slice().sort((a, b) => a.z - b.z);

    sceneRef.current = { W, H, VP, dust, stars, speck, clouds };
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
    canvas.style.width = `${W}px`;
    canvas.style.height = `${H}px`;

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

    function getPoint(s, u, time, wallAmp) {
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

      const sway = Math.sin(time * 0.55 + s.seed) * (s.z * (2.0 + wallAmp * 7.0));
      const micro = Math.sin(time * 1.15 + s.seed * 1.7) * (0.55 + 1.15 * s.z);

      return { x: x + dx + micro, y: y + dy + sway };
    }

    function spawnSparkle(st, x, y, hue, z, wallAmp) {
      const s = 0.8 + Math.random() * (1.2 + wallAmp * 1.3);
      st.sparkles.push({
        x,
        y,
        vx: (Math.random() - 0.5) * (18 + 52 * wallAmp) * (0.55 + z),
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

    function drawBulbUltra(ctx2, x, y, r, intensity, hue, swing, z, playing) {
      ctx2.save();
      ctx2.translate(x, y);
      ctx2.rotate(swing);

      const I = clamp(intensity, 0, 1);
      const alphaZ = clamp(0.25 + z * 0.78, 0, 1);

      const capW = r * 0.92;
      const capH = r * 0.72;
      ctx2.fillStyle = "rgba(6,6,8,0.95)";
      ctx2.roundRect(-capW / 2, -r * 1.28, capW, capH, 2.2);
      ctx2.fill();

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
        glass.addColorStop(0, "rgba(255,255,255,0.035)");
        glass.addColorStop(0.5, "rgba(120,140,190,0.03)");
        glass.addColorStop(1, "rgba(6,6,12,0.62)");
      }
      ctx2.fillStyle = glass;
      ctx2.fill();

      ctx2.globalCompositeOperation = "screen";
      ctx2.strokeStyle = "rgba(255,255,255,0.12)";
      ctx2.lineWidth = Math.max(0.7, 0.9 * z);
      ctx2.beginPath();
      ctx2.moveTo(-r * 0.25, -ry * 0.55);
      ctx2.quadraticCurveTo(r * 0.28, -ry * 0.1, -r * 0.05, ry * 0.55);
      ctx2.stroke();
      ctx2.globalCompositeOperation = "source-over";

      ctx2.globalCompositeOperation = "screen";
      ctx2.strokeStyle = "rgba(255,255,255,0.06)";
      ctx2.lineWidth = Math.max(0.6, 0.8 * z);
      ctx2.stroke();
      ctx2.globalCompositeOperation = "source-over";

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

    function drawFullNebulaBackground(ctx2, time, wallAmp) {
      ctx2.globalCompositeOperation = "source-over";
      ctx2.globalAlpha = 1;

      const scene = sceneRef.current;
      if (!scene) return;

      // ---------- 0) Cache a stable “poster texture” (no flicker) ----------
      if (!scene.bgTex2 || scene.bgTex2W !== W || scene.bgTex2H !== H) {
        const tex = document.createElement("canvas");
        const TW = Math.max(320, Math.floor(W / 2.6));
        const TH = Math.max(200, Math.floor(H / 2.6));
        tex.width = TW;
        tex.height = TH;
        const tctx = tex.getContext("2d", { alpha: true });

        const seed = (stateRef.current.seed >>> 0) || 1;
        const hash = (x) => {
          const s = Math.sin(x * 127.1 + seed * 0.001) * 43758.5453123;
          return s - Math.floor(s);
        };

        // Grain + cloud mask (brighter-biased than last time)
        const img = tctx.createImageData(TW, TH);
        const d = img.data;

        for (let y = 0; y < TH; y++) {
          for (let x = 0; x < TW; x++) {
            const i = (y * TW + x) * 4;

            const n1 = hash(x * 1.1 + y * 1.7);
            const n2 = hash(x * 0.55 + y * 0.85 + 33.3);
            const n3 = hash(x * 0.19 + y * 0.14 + 91.7);

            let n = n1 * 0.52 + n2 * 0.33 + n3 * 0.15;
            // bias toward midtones (poster is not “crushed blacks”)
            n = Math.pow(n, 1.15);

            const g = Math.floor(255 * n);
            d[i + 0] = g;
            d[i + 1] = g;
            d[i + 2] = g;
            d[i + 3] = 255;
          }
        }
        tctx.putImageData(img, 0, 0);

        // soften into cloudy stuff
        tctx.filter = "blur(5px)";
        tctx.globalCompositeOperation = "source-over";
        tctx.drawImage(tex, 0, 0);
        tctx.filter = "blur(0px)";

        // a few very soft dark blobs (but gentle — poster still bright)
        tctx.globalCompositeOperation = "multiply";
        for (let k = 0; k < 5; k++) {
          const u = hash(1000 + k * 17.7);
          const v = hash(2000 + k * 29.3);
          const r = (0.22 + 0.38 * hash(3000 + k * 41.9)) * Math.max(TW, TH);
          const cx = u * TW;
          const cy = v * TH;

          const g = tctx.createRadialGradient(cx, cy, r * 0.05, cx, cy, r);
          g.addColorStop(0.0, "rgba(0,0,0,0)");
          g.addColorStop(0.65, "rgba(0,0,0,0.10)");
          g.addColorStop(1.0, "rgba(0,0,0,0.22)");
          tctx.fillStyle = g;
          tctx.fillRect(0, 0, TW, TH);
        }

        // tiny bright flecks (poster glitter)
        tctx.globalCompositeOperation = "screen";
        for (let i = 0; i < 240; i++) {
          const x = hash(5000 + i * 3.1) * TW;
          const y = hash(7000 + i * 4.7) * TH;
          const r = 0.25 + hash(9000 + i * 5.9) * 0.9;
          tctx.fillStyle = `rgba(255,255,255,${0.03 + hash(11000 + i * 2.3) * 0.08})`;
          tctx.beginPath();
          tctx.arc(x, y, r, 0, Math.PI * 2);
          tctx.fill();
        }

        tctx.globalAlpha = 1;
        tctx.globalCompositeOperation = "source-over";

        scene.bgTex2 = tex;
        scene.bgTex2W = W;
        scene.bgTex2H = H;
      }

      // ---------- 1) Base: brighter deep magenta-to-blue (NOT flat red, NOT too dark) ----------
      const base = ctx2.createLinearGradient(0, 0, W, H);
      base.addColorStop(0.00, "rgba(14, 12, 32, 1)");  // deep blue-violet
      base.addColorStop(0.28, "rgba(44, 10, 46, 1)");  // purple
      base.addColorStop(0.55, "rgba(118, 24, 70, 1)"); // magenta-red midtone (brighter!)
      base.addColorStop(0.78, "rgba(84, 18, 68, 1)");  // magenta
      base.addColorStop(1.00, "rgba(16, 14, 34, 1)");  // blue-violet edge
      ctx2.fillStyle = base;
      ctx2.fillRect(0, 0, W, H);

      // ---------- 2) Big “magenta mist” lift (SCREEN) ----------
      // This is what your current render is missing: the poster has bright nebula haze.
      ctx2.globalCompositeOperation = "screen";
      const mistBoost = 0.02 + 0.01 * wallAmp;

      const mist = ctx2.createRadialGradient(
        W * 0.55, H * 0.55, 20,
        W * 0.55, H * 0.55, Math.max(W, H) * 0.95
      );
      mist.addColorStop(0.00, `rgba(255, 70, 150, ${0.05 + mistBoost})`);
      mist.addColorStop(0.25, `rgba(255, 50, 110, ${0.14 + mistBoost * 0.4})`);
      mist.addColorStop(0.55, `rgba(160, 35, 120, ${0.08 + mistBoost * 0.2})`);
      mist.addColorStop(1.00, "rgba(0,0,0,0)");
      ctx2.fillStyle = mist;
      ctx2.fillRect(0, 0, W, H);

      // ---------- 3) Blue pockets (SCREEN, present but not dominant) ----------
      const blueBoost = 0.04 + 0.08 * wallAmp;

      function bluePool(cx, cy, R, a0) {
        const g = ctx2.createRadialGradient(cx, cy, 0, cx, cy, R);
        g.addColorStop(0.0, `rgba(60, 95, 215, ${a0 + blueBoost})`);
        g.addColorStop(0.55, `rgba(35, 60, 170, ${0.45 * (a0 + blueBoost)})`);
        g.addColorStop(1.0, "rgba(0,0,0,0)");
        ctx2.fillStyle = g;
        ctx2.fillRect(0, 0, W, H);
      }

      bluePool(W * 0.16, H * 0.22, Math.max(W, H) * 0.80, 0.10);
      bluePool(W * 0.90, H * 0.22, Math.max(W, H) * 0.72, 0.09);
      bluePool(W * 0.86, H * 0.86, Math.max(W, H) * 0.78, 0.07);

      // ---------- 4) Texture pass: “distant galaxy” mottling ----------
      const tex = scene.bgTex2;
      if (tex) {
        // soft-light gives the printed nebula feel
        ctx2.globalCompositeOperation = "soft-light";
        ctx2.globalAlpha = 0.75;
        ctx2.drawImage(tex, 0, 0, W, H);

        // tiny multiply to deepen *some* areas, but do NOT crush
        ctx2.globalCompositeOperation = "multiply";
        ctx2.globalAlpha = 0.10;
        ctx2.drawImage(tex, 0, 0, W, H);

        // screen to keep it sparkly and alive
        ctx2.globalCompositeOperation = "screen";
        ctx2.globalAlpha = 0.18;
        ctx2.drawImage(tex, 0, 0, W, H);

        ctx2.globalAlpha = 1;
        ctx2.globalCompositeOperation = "source-over";
      }

      // ---------- 5) Dust + stars (SCREEN) — brighter like the poster ----------
      ctx2.globalCompositeOperation = "screen";

      for (const p of scene.dust) {
        const wob = Math.sin(time * p.tw + p.ph) * (0.6 + 1.8 * wallAmp) * (0.25 + p.z);
        const px = p.x + wob;
        const py = p.y + Math.cos(time * 0.6 + p.ph) * (0.4 + wallAmp * 1.2) * (0.25 + p.z);

        ctx2.fillStyle = `rgba(255, 150, 195, ${p.a * 0.95})`;
        ctx2.beginPath();
        ctx2.arc(px, py, p.r, 0, Math.PI * 2);
        ctx2.fill();
      }

      for (const s of scene.stars) {
        const tw = 0.65 + 0.35 * Math.sin(time * 1.2 + s.ph);
        ctx2.fillStyle = `rgba(255,235,245,${(s.a * tw) * 1.15})`;
        ctx2.beginPath();
        ctx2.arc(s.x, s.y, s.r, 0, Math.PI * 2);
        ctx2.fill();
      }

      // ---------- 6) Very gentle vignette (multiply) ----------
      // Poster edges are darker, but yours got crushed. Keep this subtle.
      ctx2.globalCompositeOperation = "multiply";
      const vig = ctx2.createRadialGradient(
        W * 0.52, H * 0.52, Math.min(W, H) * 0.18,
        W * 0.52, H * 0.52, Math.max(W, H) * 0.98
      );
      vig.addColorStop(0.00, "rgba(0,0,0,0)");
      vig.addColorStop(0.70, "rgba(0,0,0,0.08)");
      vig.addColorStop(1.00, "rgba(0,0,0,0.18)");
      ctx2.fillStyle = vig;
      ctx2.fillRect(0, 0, W, H);

      // Reset
      ctx2.globalCompositeOperation = "source-over";
      ctx2.globalAlpha = 1;
    }

    const draw = (t) => {
      const time = t * 0.001;
      const st = stateRef.current;
      const viz = vizRef.current || {};

      const playing = !!viz.playing;
      const amp = viz.amp || 0;
      const hit = !!viz.hit;
      const hitStrength = viz.hitStrength || 0;

      const audioEl = audioRef?.current;
      const audioT = audioEl && isFinite(audioEl.currentTime) ? audioEl.currentTime : time;

      // Wall amplitude is subtle: bulbs are the stars; wall stays moody.
      const wallAmp = playing ? amp * 0.16 : 0;
      const loud = clamp(Math.pow(amp, 0.65), 0, 1);

      // Always clear and repaint full background in correct order.
      ctx.clearRect(0, 0, W, H);

      ctx.save();
      ctx.beginPath();
      ctx.rect(0, 0, W, H);
      ctx.clip();

      // 1) BACKGROUND (FULL-CANVAS) — no black slabs, no flat red wall.
      drawFullNebulaBackground(ctx, time, wallAmp);

      const scene = sceneRef.current;
      if (!scene) {
        rafRef.current = requestAnimationFrame(draw);
        return;
      }

      // --- chase driver
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

        // Stronger loudness lock: near-still when quiet, fast when loud
        const bps =
          (0.35 + 7.5 * loud * loud) +     // loudness drives most of the motion
          (hit ? 6.0 * (0.25 + hitStrength) : 0);

        st.drive += bps * dt;

        const nextIdx = Math.floor(st.drive) % L;
        if (nextIdx !== st.chaseIdx) {
          st.chaseIdx = nextIdx;
          st.cooldown = hit ? 1 : 0;
        }
      } else {
        st.lastAudioT = audioT;
      }

      // Even strands: left->right, odd strands: right->left
      const activeSet = new Set();

      if (playing) {
        const perStrand = strandBulbIdsRef.current || [];
        const S = perStrand.length;

        // 👇 HOW MANY ROWS (STRANDS) ARE ACTIVE AT ONCE
        const activeRows = 5;              // try 3–8 (smaller = fewer active rows)
        const rowStride = 1;               // 1 = contiguous block, 2 = every-other row, etc.
        const rowStagger = 0.5;           // how “out of phase” rows are (similar to your old phase)

        // Move a “window” of active rows over time (locked to st.drive so it follows music speed)
        const baseRow = Math.floor(st.drive * 0.75) % Math.max(1, S);

        for (let r = 0; r < activeRows; r++) {
          const si = (baseRow + r * rowStride) % S;
          const ids = perStrand[si];
          const n = ids?.length || 0;
          if (!n) continue;

          // same left->right / right->left zig-zag as before
          const dir = si % 2 === 0 ? 1 : -1;
          const phase = si * rowStagger;

          let k = Math.floor(st.drive + phase) % n;
          if (k < 0) k += n;
          if (dir < 0) k = (n - 1) - k;

          activeSet.add(ids[k]); // ✅ one bulb in THIS active row
        }
      }

      // 2) WIRES (source-over first) + subtle highlight (screen) — correct order under bulbs.
      const strandsSorted = strandsSortedRef.current;
      for (const s of strandsSorted) {
        const aZ = clamp(0.10 + s.z * 0.88, 0, 1);

        ctx.globalCompositeOperation = "source-over";
        ctx.strokeStyle = `rgba(8,8,12,${aZ})`;
        ctx.lineWidth = 3.6 * s.z;

        ctx.beginPath();
        for (let u = 0; u <= 1.00001; u += 0.05) {
          const p = getPoint(s, u, time, wallAmp);
          if (u === 0) ctx.moveTo(p.x, p.y);
          else ctx.lineTo(p.x, p.y);
        }
        ctx.stroke();

        // tiny wire specular (not a wall glow)
        if (playing) {
          ctx.globalCompositeOperation = "screen";
          ctx.strokeStyle = `rgba(255,255,255,${0.004 + 0.014 * wallAmp * s.z})`;
          ctx.lineWidth = 1.0 * s.z;
          ctx.stroke();
        }

        ctx.globalCompositeOperation = "source-over";
      }

      // 3) BULB GLOWS (screen) then BULB BODIES (source-over) — correct order.
      const bulbsSorted = bulbsSortedRef.current;
      for (const b of bulbsSorted) {
        const s = strandsRef.current[b.si];
        if (!s) continue;

        const p0 = getPoint(s, b.u, time, wallAmp);
        const isActive = activeSet.has(b.id);

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
        b.heat *= playing ? 0.99 : 0.96;

        const hop = isActive ? b.on : 0;
        b.jv = lerp(b.jv, hop, 0.12);
        const n1 = noise2(time * 6.0, b.seed * 11.7);
        const n2 = noise2(time * 6.7 + 10.0, b.seed * 19.3);
        b.jx = lerp(b.jx, (Math.sin(time * 10 + b.seed) * 1.6 + n1 * 1.2) * b.jv, 0.10);
        b.jy = lerp(b.jy, (Math.cos(time * 9 + b.seed) * 1.2 + n2 * 1.0) * b.jv, 0.10);

        const px = p0.x + b.jx * (1 + b.z * 0.7);
        const py = p0.y + b.jy * (1 + b.z * 0.7);

        const hue = b.hue + b.hueJ + (playing ? 0.02 * Math.sin(time * 0.6 + b.seed) : 0);

        // Show color ONLY on active bulb; others remain clear/grey when silent or inactive.
        const shown = playing && isActive ? clamp(0.2 + b.on * 0.5, 0, 1) : 0;

        // Glow pass first (screen)
        if (playing && isActive && shown > 0.06) {
          ctx.save();
          ctx.globalCompositeOperation = "screen";
          // const glowR = b.r * (18.0 + 14.0 * b.z) * (0.55 + shown * 1.6);
          const glowR = b.r * (12 + 10 * b.z) * (0.5 + shown * 1.4);
          const g = ctx.createRadialGradient(px, py, 0, px, py, glowR);
          g.addColorStop(0, hsla(hue, 100, 65, 0.34 * shown * (0.85 + b.z)));
          g.addColorStop(0.35, hsla(hue, 100, 55, 0.16 * shown * (0.85 + b.z)));
          g.addColorStop(1, "rgba(0,0,0,0)");
          ctx.fillStyle = g;
          ctx.fillRect(px - glowR, py - glowR, glowR * 2, glowR * 2);
          ctx.restore();
        }

        // Sparkles (only near active bulb)
        if (playing && isActive && shown > 0.18 && Math.random() < 0.09) {
          spawnSparkle(
            st,
            px + (Math.random() - 0.5) * 14,
            py + (Math.random() - 0.5) * 10,
            hue,
            b.z,
            wallAmp
          );
        }

        const swingTarget =
          Math.sin(time * (0.85 + b.z * 0.22) + b.seed) * (0.06 + 0.11 * b.z) * (0.25 + wallAmp);
        b.swingVel = lerp(b.swingVel, (swingTarget - b.swing) * 0.35, 0.08);
        b.swing += b.swingVel;

        // Bulb body pass (source-over)
        ctx.globalCompositeOperation = "source-over";
        drawBulbUltra(ctx, px, py, b.r * 1.15, shown, hue, b.swing, b.z, playing);
      }

      // 4) Sparkles over everything (screen) — final pass
      if (st.sparkles.length) {
        ctx.globalCompositeOperation = "screen";
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
        ctx.globalCompositeOperation = "source-over";
      }

      // Reset state (prevents “mystery compositing” artifacts)
      ctx.globalAlpha = 1;
      ctx.globalCompositeOperation = "source-over";

      ctx.restore();

      rafRef.current = requestAnimationFrame(draw);
    };

    rafRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(rafRef.current);
  }, [rect.w, rect.h, vizRef, audioRef]);

  return (
    <div className="ragLights" ref={wrapRef} aria-hidden="true">
      <canvas ref={canvasRef} className="ragLightsCanvas" />
      <div className="ragLightsBadge">
        <span className="ragLightsBadgeDiamond" />
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

function computeRmsSeriesFromPcm(pcm, sr, hop = 1024, win = 2048, maxPoints = 1200) {
  const N = pcm.length;
  const t = [];
  const y = [];

  const winInv = 1 / Math.max(1, win);

  for (let i = 0; i + win < N; i += hop) {
    let s = 0;
    for (let k = 0; k < win; k++) {
      const v = pcm[i + k];
      s += v * v;
    }
    const rms = Math.sqrt(s * winInv);
    t.push(i / sr);
    y.push(rms);
  }

  // downsample for UI
  if (t.length > maxPoints) {
    const stride = Math.ceil(t.length / maxPoints);
    const tt = [];
    const yy = [];
    for (let i = 0; i < t.length; i += stride) {
      tt.push(t[i]);
      yy.push(y[i]);
    }
    return { t: tt, y: yy };
  }

  return { t, y };
}

function useLoudnessForUrl(url) {
  const cacheRef = useRef(new Map());
  const [series, setSeries] = useState({ t: [], y: [] });
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let alive = true;

    async function run() {
      if (!url) {
        setSeries({ t: [], y: [] });
        setLoading(false);
        return;
      }

      const cached = cacheRef.current.get(url);
      if (cached) {
        setSeries(cached);
        setLoading(false);
        return;
      }

      setLoading(true);
      try {
        const res = await fetch(url);
        if (!res.ok) throw new Error(`Failed to fetch audio (${res.status})`);
        const buf = await res.arrayBuffer();

        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        const audioBuf = await ctx.decodeAudioData(buf.slice(0));
        const sr = audioBuf.sampleRate;

        // mono mixdown
        const ch0 = audioBuf.getChannelData(0);
        let pcm = ch0;
        if (audioBuf.numberOfChannels > 1) {
          const ch1 = audioBuf.getChannelData(1);
          const mix = new Float32Array(ch0.length);
          for (let i = 0; i < mix.length; i++) mix[i] = 0.5 * (ch0[i] + ch1[i]);
          pcm = mix;
        }

        const out = computeRmsSeriesFromPcm(pcm, sr);
        cacheRef.current.set(url, out);

        // close context (polite)
        try { await ctx.close(); } catch {}

        if (alive) setSeries(out);
      } catch {
        if (alive) setSeries({ t: [], y: [] });
      } finally {
        if (alive) setLoading(false);
      }
    }

    run();
    return () => {
      alive = false;
    };
  }, [url]);

  return { series, loading };
}

/** ---------- Main RAG screen ---------- **/
export default function Rag() {
  const fileRef = useRef(null);

  const [backendOk, setBackendOk] = useState(false);
  const [readyInfo, setReadyInfo] = useState(null);
  const ready = readyInfo && typeof readyInfo.ready === "boolean" ? readyInfo.ready : null;

  const [targetDur, setTargetDur] = useState("20");
  const [topK, setTopK] = useState("5");

  const [uploadId, setUploadId] = useState(null);
  const [uploadMeta, setUploadMeta] = useState(null);
  const [inputAnalysis, setInputAnalysis] = useState(null);

  const [runs, setRuns] = useState(() => loadRuns());
  const [selectedRunId, setSelectedRunId] = useState(null);

  const [results, setResults] = useState([]);
  const [activeIdx, setActiveIdx] = useState(0);

  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("Idle");

  // ✅ declare refs early
  const inputAudioRef = useRef(null);
  const recoAudioRef = useRef(null);

  // ✅ declare activeViz BEFORE using it anywhere
  const [activeViz, setActiveViz] = useState("none"); // input | reco | none

  // ✅ derived URLs BEFORE anything that reads them
  const inputUrl = useMemo(
    () => (uploadId ? `${API}/rag/${uploadId}/files/input.wav` : ""),
    [uploadId]
  );

  const activeReco = useMemo(
    () => (results?.length ? results[clamp(activeIdx, 0, results.length - 1)] : null),
    [results, activeIdx]
  );

  const recoUrl = useMemo(
    () => (activeReco?.extension_wav_url ? `${API}${activeReco.extension_wav_url}` : ""),
    [activeReco]
  );

  // ✅ UI hooks after refs exist
  const inputUI = useAudioUI(inputAudioRef);
  const recoUI = useAudioUI(recoAudioRef);

  // ✅ only now it's safe to compute these
  const activePlayer = activeViz === "reco" ? "reco" : "input";
  const activeUI = activePlayer === "reco" ? recoUI : inputUI;

  const activeName =
    activePlayer === "reco"
      ? (activeReco ? `Reco • Rank ${activeReco.rank ?? activeIdx + 1}` : "Reco")
      : (uploadMeta?.filename ? niceName(uploadMeta.filename) : "Input");

  const loudnessUrl = activePlayer === "reco" ? recoUrl : inputUrl;

  // client-side loudness (works for both)
  const loud = useLoudnessForUrl(loudnessUrl);

  const recoListRef = useRef(null);

  const activeAudioRef = useMemo(() => {
    if (activeViz === "input") return inputAudioRef;
    if (activeViz === "reco") return recoAudioRef;
    return { current: null };
  }, [activeViz]);

  function scrollNextReco() {
    const el = recoListRef.current;
    if (!el) return;

    const firstCard = el.querySelector(".recoV");
    const step = firstCard ? firstCard.getBoundingClientRect().height + 14 : 180;
    el.scrollBy({ top: step, behavior: "smooth" });
  }

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
      { k: "Target", v: `${clampInt(targetDur || 20, 4, 240)}s` },
      { k: "Top-K", v: String(clampInt(topK || 5, 1, 12)) },
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
        target_s: clampInt(targetDur || 20, 4, 240),
        top_k: clampInt(topK || 5, 1, 12),
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
    setStatus("Finding extensions…");
    setResults([]);
    setActiveIdx(0);

    try {
      const r = await fetch(`${API}/rag/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          upload_id: uploadId,
          target_duration_sec: clampInt(targetDur || 20, 4, 240),
          top_k: clampInt(topK || 5, 1, 12),
        }),
      });
      if (!r.ok) throw new Error(await r.text());
      const j = await r.json();
      const arr = Array.isArray(j.results) ? j.results : [];

      setResults(arr);
      setStatus(arr.length ? "Recommendations ready ✦" : "No matches found");

      // ✅ Persist recos in the Past Run entry
      setRuns(
        upsertRunResults(uploadId, {
          filename: uploadMeta?.filename,
          created_at: new Date().toISOString(),
          target_s: clampInt(targetDur || 20, 4, 240),
          top_k: clampInt(topK || 5, 1, 12),
          results: arr,              // 👈 the magic
          last_active_idx: 0,
        })
      );
    } catch (e) {
      setStatus(String(e?.message || e));
    } finally {
      setBusy(false);
    }
  }

  function selectRun(run) {
    if (!run?.upload_id) return;

    const id = run.upload_id;
    setSelectedRunId(id);

    // Load input for that run
    setUploadId(id);
    setUploadMeta({ filename: run.filename });

    // ✅ Load saved results (if any)
    const savedResults = Array.isArray(run.results) ? run.results : [];
    setResults(savedResults);
    const idx = Number.isFinite(run.last_active_idx) ? run.last_active_idx : 0;
    setActiveIdx(clamp(idx, 0, Math.max(0, savedResults.length - 1)));

    setStatus(savedResults.length ? "Loaded prior run ✦ (input + recos)" : "Loaded prior run (input only)…");

    fetchInputAnalysis(id);

    requestAnimationFrame(() => {
      if (inputAudioRef.current) {
        inputAudioRef.current.src = `${API}/rag/${id}/files/input.wav`;
        inputAudioRef.current.preload = "auto";
        inputAudioRef.current.load();
      }
    });
  }

  async function playRecoDirect(index) {
    const rObj = results?.[index];
    if (!rObj?.extension_wav_url) return;

    const url = `${API}${rObj.extension_wav_url}`;
    setActiveIdx(index);

    if (uploadId) setRuns(upsertRunResults(uploadId, { last_active_idx: index }));

    setActiveViz("reco");
    try { inputAudioRef.current?.pause(); } catch {}

    const el = recoAudioRef.current;
    if (!el) return;

    try {
      try { el.crossOrigin = "anonymous"; } catch {}

      if (el.src !== url) {
        el.pause();
        el.src = url;
        el.preload = "auto";
        el.load();
        await Promise.race([once(el, "canplay"), once(el, "loadeddata")]);
      }

      await el.play();
    } catch (e) {
      setStatus(String(e?.message || e));
    }
  }

  const canStitch = !!uploadId && !busy && backendOk && ready !== false;

  return (
    <div className="shellRag">
      {/* LEFT SIDEBAR */}
      <aside className="sidebar">
        <div className="brandBar">
          <div className="logoMark">✦</div>
          <div>
            <div className="brandTitle">Audio RAG</div>
            <div className="brandSub"></div>
          </div>
        </div>

        <div className="statusPills">
          <span className={`pill ${backendOk ? "pillOk" : "pillBad"}`}>{backendOk ? "Backend OK" : "Backend Down"}</span>
          {ready !== null && <span className={`pill ${ready ? "pillOk" : "pillWarn"}`}>{ready ? "Ready" : "Not Ready"}</span>}
        </div>

        <div className="controlCard">
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
                onChange={(e) => {
                  // allow empty + partial typing without forcing min/max
                  const v = e.target.value; // string
                  if (v === "") return setTargetDur("");

                  // keep only digits (optional, but helps)
                  if (!/^\d+$/.test(v)) return;

                  // don't clamp here — just store what user typed
                  setTargetDur(v);
                }}
                onBlur={() => {
                  // clamp when user leaves the field
                  const n = clampInt(targetDur || 20, 4, 240);
                  setTargetDur(String(n));
                }}
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
                onChange={(e) => {
                  const v = e.target.value;
                  if (v === "") return setTopK("");
                  if (!/^\d+$/.test(v)) return;
                  setTopK(v);
                }}
                onBlur={() => {
                  const n = clampInt(topK || 5, 1, 12);
                  setTopK(String(n));
                }}
                disabled={busy}
              />
            </label>
          </div>

          <div className="buttonRow">
            <button className="buttonSecondary buttonTight" onClick={() => fileRef.current?.click()} disabled={busy}>
              Upload WAV
            </button>
            <button className="button buttonTight" onClick={stitch} disabled={!canStitch}>
              {busy ? "Working…" : "Find ✦"}
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

          {/* <div className="helperRow">
            <span className="helper">Upload first</span>
            <span className="helper">then Extend</span>
          </div> */}
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

      {/* MIDDLE MAIN — vertical stack */}
      <main className="main">
        {/* 1) LightWall at TOP */}
        <section className="panel panelTight lightPanel">
          <div className="lightPanelInner">
            <LightsWall audioRef={activeAudioRef} label="LightWall • Vibe" />
          </div>
        </section>

        {/* 2) KPIs next */}
        {/* <section className="kpiStrip">
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
        </section> */}

        {/* 3) Input Player + 4) Loudness below */}
        <section className="panel panelTight">
          <div className="panelHeaderTight">
            <div className="panelTitle">{activeName}</div>
            <div className="headerRight">
              <div className="idlePill">
                <span className="idleDot" />
                {status || "Idle"}
              </div>
            </div>
          </div>
          <div className="playerMini playerMiniFixed">
            <div className="playerMiniTop">
              <div className="playerMiniSub">{uploadMeta?.filename ? niceName(uploadMeta.filename) : "No file"}</div>
            </div>

            <audio ref={inputAudioRef} src={inputUrl || ""} controls className="nativeAudioHidden" />

            {uploadId ? (
              <div className="playerMiniUI playerMiniUIFixed">
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

                {/* <div className="downloadRowSm">
                  <a className="linkBtnSm" href={inputUrl} download>
                    Download input.wav
                  </a>
                </div> */}
              </div>
            ) : (
              <div className="empty">Upload a WAV to enable playback.</div>
            )}
          </div>

          <div className="vizStack">
            <CanvasLineChart
              title={`Loudness • ${activePlayer === "reco" ? "Reco" : "Input"}`}
              x={loud.series.t}
              y={loud.series.y}
              yLabel="RMS"
              height={150}
              palette={palette}
              sparkle={true}
            />
          </div>

 
        </section>
      </main>

      {/* RIGHT RECO COLUMN — smaller + single block (NO reco curve) */}
      <aside className="recoCol">
        <div className="recoColHead">
          <div className="recoColTitle">Top Suggestions</div>
          <div className="recoColSub">{results.length ? `${activeIdx + 1} / ${results.length}` : "—"}</div>
        </div>

        {/* <div className="recoColToolbar">
          <button
            className="chipBtn"
            onClick={() => results.length && setActiveIdx((p) => (p - 1 + results.length) % results.length)}
            type="button"
            disabled={!results.length}
          >
            Prev
          </button>
          <button
            className="chipBtn"
            onClick={() => results.length && setActiveIdx((p) => (p + 1) % results.length)}
            type="button"
            disabled={!results.length}
          >
            Next
          </button>
          <div className="spacer" />
          <button
            className="chipBtn strong"
            onClick={() => results.length && playRecoDirect(activeIdx)}
            type="button"
            disabled={!results.length || !activeReco?.extension_wav_url}
          >
            ▶ Play
          </button>
          <div className="chip scoreChip">Score: {activeReco?.score != null ? fmt(activeReco.score, 4) : "—"}</div>
        </div> */}

        <audio ref={recoAudioRef} src={recoUrl || ""} className="nativeAudioHidden" />

        <div className="recoList" ref={recoListRef}>
          {results.length ? (
            results.map((r, i) => {
              const active = i === activeIdx;
              return (
                <div key={`${r.rank || i}-${i}`} className={`recoV ${active ? "recoVActive" : ""}`}>
                  <button
                    className={`recoVPlay ${recoUI.playing && active ? "isPlaying" : ""}`}
                    onClick={() => playRecoDirect(i)}
                    type="button"
                    aria-label="Play recommendation"
                  >
                    <span className="playIcon" />
                  </button>

                  <button className="recoVBody" onClick={() => setActiveIdx(i)} type="button" title="Select">
                    <div className="recoVTop">
                      <div className="recoVTitle">Rank {r.rank ?? i + 1}</div>
                      <div className="recoVTiny">{r.db_id_4s != null ? `DB ${r.db_id_4s}` : ""}</div>
                    </div>

                    <div className="recoVRow">
                      <span className="recoVK">Score</span>
                      <span className="recoVV">{r.score != null ? fmt(r.score, 2) : "—"}</span>
                    </div>
                    <div className="recoVRow">
                      <span className="recoVK">Coherence</span>
                      <span className="recoVV">{r.coherence != null ? fmt(r.coherence, 2) : "—"}</span>
                    </div>
                    <div className="recoVRow">
                      <span className="recoVK">Relevance</span>
                      <span className="recoVV">{r.relevance != null ? fmt(r.relevance, 2) : "—"}</span>
                    </div>
                  </button>
                </div>
              );
            })
          ) : (
            <div className="runsEmpty">No extensions yet. Upload + Find.</div>
          )}
          <div className="recoFooterNav">
          <button className="recoNextBtn" onClick={scrollNextReco} type="button" disabled={!results.length}>
            Next reco ↓
          </button>
        </div>
        </div>
      </aside>
    </div>
  );
}