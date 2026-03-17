// ==UserScript==
// @name         Crash Predict AI — Gemini-Powered Prediction
// @namespace    crash-predict-ai
// @version      1.0
// @description  AI-powered crash prediction using local vector DB (IndexedDB) + Gemini 2.5 Flash. Stores round features, KNN similarity search, real-time predictions.
// @match        https://melbet-srilanka.com/games-frame/games/371*
// @match        https://*.melbet*.com/games-frame/games/371*
// @grant        none
// @run-at       document-start
// ==/UserScript==

// ═══════════════════════════════════════════════════════════════
//  CRASH PREDICT AI
//
//  Architecture:
//    1. WebSocket hook intercepts SignalR crash game events
//    2. OnCrash → store round as feature vector in IndexedDB
//    3. OnBetting → KNN query + Gemini 2.5 Flash API call
//    4. Display prediction with confidence + advice
//
//  Features stored per round (9D vector):
//    crashValue(log), avgDelta, minDelta, stdDelta,
//    cashoutRatio, roundDuration, lowStreak, highStreak, tickLeak
//
//  Debug: window.__predict_debug()
//  Remove: window.__predict_destroy()
//  Config: click ⚙ gear icon in UI
// ═══════════════════════════════════════════════════════════════

(function () {
  "use strict";

  if (window.__cpActive) return;
  window.__cpActive = true;

  // ===================== CONFIG =====================
  const CFG = {
    API_KEY: localStorage.getItem("cp_api_key") || "YOUR_GEMINI_API_KEY_HERE",
    MODEL: localStorage.getItem("cp_model") || "gemini-2.5-flash",
    KNN_K: parseInt(localStorage.getItem("cp_knn_k")) || 5,
    MIN_ROUNDS: parseInt(localStorage.getItem("cp_min_rounds")) || 10,
    AUTO_PREDICT: localStorage.getItem("cp_auto") !== "false",
    STREAK_LOW: 1.5,
    STREAK_HIGH: 5.0,
    P1_DELTA: 5,
    DB_NAME: "crash-predict-db",
    DB_STORE: "rounds",
    DB_VER: 1,
  };

  function saveCfg() {
    localStorage.setItem("cp_api_key", CFG.API_KEY);
    localStorage.setItem("cp_model", CFG.MODEL);
    localStorage.setItem("cp_knn_k", CFG.KNN_K);
    localStorage.setItem("cp_min_rounds", CFG.MIN_ROUNDS);
    localStorage.setItem("cp_auto", CFG.AUTO_PREDICT);
  }

  // ===================== VECTOR DB (IndexedDB) =====================
  const VDB = {
    db: null,
    queue: [],

    // Normalization ranges: [min, max] per feature dimension
    RANGES: [
      [0, 3.6], // f0: ln(crashValue)  — ln(1)=0 to ln(~35)=3.6
      [0, 2000], // f1: avgDelta (ms)
      [0, 1000], // f2: minDelta (ms)
      [0, 800], // f3: stdDelta (ms)
      [0, 1], // f4: cashoutRatio
      [0, 60000], // f5: roundDuration (ms)
      [0, 10], // f6: lowStreak
      [0, 10], // f7: highStreak
      [0, 1], // f8: tickLeak (binary)
    ],

    init() {
      return new Promise((resolve, reject) => {
        const req = indexedDB.open(CFG.DB_NAME, CFG.DB_VER);
        req.onupgradeneeded = (e) => {
          const db = e.target.result;
          if (!db.objectStoreNames.contains(CFG.DB_STORE)) {
            const store = db.createObjectStore(CFG.DB_STORE, {
              keyPath: "id",
              autoIncrement: true,
            });
            store.createIndex("ts", "ts", { unique: false });
          }
        };
        req.onsuccess = (e) => {
          VDB.db = e.target.result;
          // Flush queued rounds
          if (VDB.queue.length) {
            VDB.queue.forEach((r) => VDB.addRound(r));
            VDB.queue = [];
          }
          resolve();
        };
        req.onerror = (e) => reject(e.target.error);
      });
    },

    normalize(f) {
      return f.map((v, i) => {
        const [lo, hi] = VDB.RANGES[i];
        return hi === lo ? 0 : Math.max(0, Math.min(1, (v - lo) / (hi - lo)));
      });
    },

    cosine(a, b) {
      let dot = 0,
        ma = 0,
        mb = 0;
      for (let i = 0; i < a.length; i++) {
        dot += a[i] * b[i];
        ma += a[i] * a[i];
        mb += b[i] * b[i];
      }
      ma = Math.sqrt(ma);
      mb = Math.sqrt(mb);
      return ma && mb ? dot / (ma * mb) : 0;
    },

    buildFeatures(rec) {
      const deltas = rec.profitsDeltas || [];
      const avg = deltas.length
        ? deltas.reduce((a, b) => a + b, 0) / deltas.length
        : 0;
      const min = deltas.length ? Math.min(...deltas) : 0;
      const std =
        deltas.length > 1
          ? Math.sqrt(
              deltas.reduce((s, d) => s + (d - avg) ** 2, 0) /
                (deltas.length - 1),
            )
          : 0;
      const totalEvents = (rec.profitCount || 0) + (rec.cashoutCount || 0);
      const coRatio =
        totalEvents > 0 ? (rec.cashoutCount || 0) / totalEvents : 0;

      return [
        Math.log(Math.max(rec.crashValue || 1, 1)),
        avg,
        min,
        std,
        coRatio,
        rec.roundDuration || 0,
        rec.lowStreak || 0,
        rec.highStreak || 0,
        rec.tickLeak ? 1 : 0,
      ];
    },

    addRound(rec) {
      if (!VDB.db) {
        VDB.queue.push(rec);
        return Promise.resolve(-1);
      }
      return new Promise((resolve, reject) => {
        const features = VDB.buildFeatures(rec);
        const entry = {
          cv: rec.crashValue,
          features,
          norm: VDB.normalize(features),
          lowStreak: rec.lowStreak || 0,
          highStreak: rec.highStreak || 0,
          avgDelta: features[1],
          minDelta: features[2],
          tickLeak: rec.tickLeak || false,
          duration: rec.roundDuration || 0,
          profitCount: rec.profitCount || 0,
          cashoutCount: rec.cashoutCount || 0,
          rewardScore: 1.0, // V2: Baseline reward score
          ts: Date.now(),
        };
        const tx = VDB.db.transaction(CFG.DB_STORE, "readwrite");
        tx.objectStore(CFG.DB_STORE).add(entry).onsuccess = (e) =>
          resolve(e.target.result);
        tx.onerror = (e) => reject(e.target.error);
      });
    },

    _getAll() {
      return new Promise((resolve, reject) => {
        if (!VDB.db) {
          resolve([]);
          return;
        }
        const tx = VDB.db.transaction(CFG.DB_STORE, "readonly");
        const req = tx.objectStore(CFG.DB_STORE).getAll();
        req.onsuccess = () => resolve(req.result || []);
        req.onerror = (e) => reject(e.target.error);
      });
    },

    async queryKNN(queryFeatures, k) {
      const all = await VDB._getAll();
      if (!all.length) return [];
      const qn = VDB.normalize(queryFeatures);
      const now = Date.now();

      const scored = all.map((r) => {
        const sim = VDB.cosine(qn, r.norm);
        // V2: Weight similarity by learned reward score. Base is 10, so log10(10+1) ~ 1.04
        const rewardMultiplier = Math.max(
          0.5,
          Math.log10(10 + (r.rewardScore || 1.0)),
        );
        // V3: Exponential Recency Bias (Half-life of ~2 hours)
        const hoursOld = (now - r.ts) / (1000 * 60 * 60);
        const timeDecay = Math.exp(-hoursOld / 2); // Drops slowly, focuses on recent algorithm seeds

        return { ...r, rawSim: sim, sim: sim * rewardMultiplier * timeDecay };
      });
      scored.sort((a, b) => b.sim - a.sim);

      // V3: Adaptive K - filter out noise, only keep high similarity (threshold 0.85)
      // If none are > 0.85, just return the single best match to avoid confusing the AI
      let topK = scored.slice(0, k || CFG.KNN_K);
      const highConfidence = topK.filter((r) => r.rawSim > 0.85);
      if (highConfidence.length > 0) {
        topK = highConfidence;
      } else if (topK.length > 0) {
        topK = [topK[0]];
      }

      return topK;
    },

    updateRewards(ids, multiplier) {
      if (!VDB.db || !ids || !ids.length) return Promise.resolve();
      return new Promise((resolve, reject) => {
        const tx = VDB.db.transaction(CFG.DB_STORE, "readwrite");
        const store = tx.objectStore(CFG.DB_STORE);
        let completed = 0;

        ids.forEach((id) => {
          const req = store.get(id);
          req.onsuccess = () => {
            const data = req.result;
            if (data) {
              // Clamp reward score between 0.1 and 100 to prevent infinite explosion/decay
              data.rewardScore = Math.max(
                0.1,
                Math.min(100, (data.rewardScore || 1.0) * multiplier),
              );
              store.put(data);
            }
            completed++;
            if (completed === ids.length) resolve();
          };
          req.onerror = (e) => reject(e.target.error);
        });
      });
    },

    async getRecent(n) {
      const all = await VDB._getAll();
      return all.slice(-(n || 15));
    },

    async getStats() {
      const all = await VDB._getAll();
      if (!all.length) return { count: 0, avg: 0, min: 0, max: 0, median: 0 };
      const vals = all.map((r) => r.cv).sort((a, b) => a - b);
      return {
        count: all.length,
        avg: vals.reduce((a, b) => a + b, 0) / vals.length,
        min: vals[0],
        max: vals[vals.length - 1],
        median: vals[Math.floor(vals.length / 2)],
      };
    },

    async clearAll() {
      if (!VDB.db) return;
      const tx = VDB.db.transaction(CFG.DB_STORE, "readwrite");
      tx.objectStore(CFG.DB_STORE).clear();
    },

    async exportJSON() {
      const all = await VDB._getAll();
      return JSON.stringify(all, null, 2);
    },

    async importJSON(jsonData) {
      if (!VDB.db) throw new Error("DB not initialized");
      const data = JSON.parse(jsonData);
      if (!Array.isArray(data))
        throw new Error("Invalid format: expected array");

      await VDB.clearAll(); // Clear existing

      return new Promise((resolve, reject) => {
        const tx = VDB.db.transaction(CFG.DB_STORE, "readwrite");
        const store = tx.objectStore(CFG.DB_STORE);
        let count = 0;

        data.forEach((item) => {
          // Strip old ID to let autoIncrement generate new ones safely
          delete item.id;
          // Ensure rewardScore exists for v1 -> v2 migration
          if (typeof item.rewardScore === "undefined") item.rewardScore = 1.0;
          store.add(item).onsuccess = () => count++;
        });

        tx.oncomplete = () => resolve(count);
        tx.onerror = (e) => reject(e.target.error);
      });
    },
  };

  // ===================== GEMINI API =====================
  const AI = {
    total: 0,
    correct: 0,
    lastPred: null,
    callCount: 0,
    loading: false,

    async call(prompt, temperature = 0.4) {
      const url = `https://generativelanguage.googleapis.com/v1beta/models/${CFG.MODEL}:generateContent?key=${CFG.API_KEY}`;
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          contents: [{ parts: [{ text: prompt }] }],
          generationConfig: {
            responseMimeType: "application/json",
            temperature: temperature, // V3: Dynamic temperature scaling
            maxOutputTokens: 512,
            // Disable thinking mode — it conflicts with JSON output mode.
            // With thinking on, the model wraps JSON in prose/markdown.
            // thinkingBudget: 0 is officially supported for 2.5 Flash.
            thinkingConfig: { thinkingBudget: 0 },
          },
        }),
      });
      if (!res.ok) {
        const errText = await res.text().catch(() => "");
        throw new Error(`API ${res.status}: ${errText.substring(0, 120)}`);
      }
      const data = await res.json();
      AI.callCount++;

      // Gemini 2.5 Flash is a "thinking" model — response.parts[] contains:
      //   parts[0..N-1] with thought:true  (internal reasoning, NOT our JSON)
      //   parts[N] = the actual content     (our JSON output)
      // We must skip thought parts and find the real content.
      const parts = data.candidates?.[0]?.content?.parts || [];
      let txt = "{}";

      // Strategy 1: Find the last non-thought part (standard thinking model behavior)
      for (let i = parts.length - 1; i >= 0; i--) {
        if (!parts[i].thought && parts[i].text) {
          txt = parts[i].text;
          break;
        }
      }

      // Strategy 2: If that didn't work, try any part with text
      if (txt === "{}" && parts.length > 0) {
        for (const p of parts) {
          if (p.text) {
            txt = p.text;
            break;
          }
        }
      }

      // Clean up common wrapping issues
      txt = txt.trim();
      // Strip markdown code fences: ```json ... ``` or ``` ... ```
      txt = txt
        .replace(/^```(?:json)?\s*\n?/i, "")
        .replace(/\n?```\s*$/i, "")
        .trim();

      try {
        return JSON.parse(txt);
      } catch {
        // Last resort: try to extract JSON object from the text
        const jsonMatch = txt.match(/\{[\s\S]*\}/);
        if (jsonMatch) {
          try {
            return JSON.parse(jsonMatch[0]);
          } catch {
            /* fall through */
          }
        }
        log("⚠ Raw response: " + txt.substring(0, 100), "w");
        return {
          prediction: 0,
          confidence: 0,
          advice: "ERROR",
          reasoning: "Parse failed — check console",
        };
      }
    },

    buildPrompt(recent, knn, stats) {
      const vals = recent.map((r) => r.cv.toFixed(2)).join(", ");
      let knnBlock = "Insufficient data for similarity search.";
      if (knn.length) {
        knnBlock = knn
          .map(
            (r, i) =>
              `  #${i + 1} [sim=${r.sim.toFixed(3)}] crashed=${r.cv.toFixed(2)}x | avgΔ=${r.avgDelta.toFixed(0)}ms | minΔ=${r.minDelta.toFixed(0)}ms | tickLeak=${r.tickLeak ? "YES" : "no"} | lowStrk=${r.lowStreak} | dur=${(r.duration / 1000).toFixed(1)}s`,
          )
          .join("\n");
      }

      return `You are a statistical analyst for a crash/multiplier game. A multiplier rises from 1.00x and randomly crashes. Predict the NEXT crash multiplier.

RECENT CRASHES (newest first, last ${recent.length} rounds):
[${vals}]

CURRENT STATE:
- Consecutive low crashes (<${CFG.STREAK_LOW}x): ${S.lowStreak}
- Consecutive high crashes (>${CFG.STREAK_HIGH}x): ${S.highStreak}

ALL-TIME STATISTICS (${stats.count} rounds):
- Mean: ${stats.avg.toFixed(2)}x | Median: ${stats.median.toFixed(2)}x
- Range: ${stats.min.toFixed(2)}x – ${stats.max.toFixed(2)}x

${knn.length} MOST SIMILAR ROUNDS (by latency/feature cosine similarity):
${knnBlock}

ANALYSIS GUIDELINES:
- The game uses a provably fair RNG — no deterministic pattern exists
- However, short-term statistical clustering and mean-reversion tendencies are observable
- Latency features (avgΔ, minΔ, tickLeak) often correlate with crash timing
- After streaks of low crashes, slightly higher results tend to follow (regression to mean)
- After very high crashes (>5x), next round tends to be lower
- Be honest about uncertainty — high confidence should be rare

Return ONLY valid JSON:
{
  "prediction": <number 1.0 to 35.0>,
  "confidence": <number 0 to 100>,
  "advice": "BET" | "SKIP" | "CAUTION",
  "reasoning": "<concise one-line explanation>",
  "suggestedCashout": <number — conservative safe cashout if betting>
}`;
    },

    async predict() {
      if (AI.loading) return null;
      AI.loading = true;
      try {
        const stats = await VDB.getStats();
        if (stats.count < CFG.MIN_ROUNDS) {
          AI.loading = false;
          return {
            prediction: 0,
            confidence: 0,
            advice: "WAIT",
            reasoning: `Collecting data — need ${CFG.MIN_ROUNDS - stats.count} more rounds`,
            suggestedCashout: 0,
          };
        }

        const recent = await VDB.getRecent(15);

        // V3: Dynamic Temperature Scaling based on standard deviation
        const recentVals = recent.map((r) => r.cv);
        const mean =
          recentVals.reduce((a, b) => a + b, 0) / (recentVals.length || 1);
        const variance =
          recentVals.reduce((a, b) => a + Math.pow(b - mean, 2), 0) /
          (recentVals.length || 1);
        const stdDev = Math.sqrt(variance);

        let dynamicTemp = 0.4; // Default
        if (stdDev > 4.0)
          dynamicTemp = 0.7; // High volatility -> High temp for creative outlier prediction
        else if (stdDev < 1.0) dynamicTemp = 0.2; // Dead streaks -> Stiff logic

        const last = recent[recent.length - 1];
        let knn = [];
        if (last) knn = await VDB.queryKNN(last.features, CFG.KNN_K);

        const prompt = AI.buildPrompt([...recent].reverse(), knn, stats);
        const result = await AI.call(prompt, dynamicTemp);
        // V2: Store the KNN IDs used for this prediction so we can reward/punish them later
        AI.lastPred = {
          ...result,
          madeAt: Date.now(),
          knnIds: knn.map((k) => k.id),
        };
        AI.loading = false;
        return result;
      } catch (err) {
        AI.loading = false;
        log("❌ AI: " + err.message, "s");
        return {
          prediction: 0,
          confidence: 0,
          advice: "ERROR",
          reasoning: err.message,
          suggestedCashout: 0,
        };
      }
    },

    trackAccuracy(actual) {
      if (
        !AI.lastPred ||
        !AI.lastPred.prediction ||
        AI.lastPred.advice === "WAIT" ||
        AI.lastPred.advice === "ERROR"
      )
        return;
      AI.total++;
      const adv = AI.lastPred.advice;
      if (
        (adv === "SKIP" && actual < 1.5) ||
        (adv === "BET" && actual >= (AI.lastPred.suggestedCashout || 1.5))
      ) {
        AI.correct++;
      } else if (adv === "CAUTION" && actual >= 1.3) {
        AI.correct++;
      }
    },
  };

  // ===================== STATE =====================
  const S = {
    ws: null,
    status: null,
    gainCoef: 25,
    coeffStartTime: null,
    mult: 1,
    isCrashed: true,
    stageId: null,
    consecutiveProfits: 0,
    profitsDeltas: [],
    lastEventTime: null,
    hasCashouts: false,
    cashoutCount: 0,
    profitCount: 0,
    roundEvents: 0,
    startTime: null,
    crashHistory: [],
    lowStreak: 0,
    highStreak: 0,
    tickLeakDetected: false,
  };

  let animId = null;

  function calcMult(ms) {
    return !ms || ms <= 0 ? 1 : Math.min((S.gainCoef / 1e9) * ms * ms + 1, 35);
  }

  function startAnim() {
    stopAnim();
    S.isCrashed = false;
    (function tick() {
      if (S.isCrashed) return;
      S.mult = calcMult(Date.now() - S.coeffStartTime);
      const el = Q("#cp-mult");
      if (el) {
        el.textContent = S.mult.toFixed(2) + "x";
        el.className = "cp-mv growing";
      }
      animId = requestAnimationFrame(tick);
    })();
  }
  function stopAnim() {
    if (animId) {
      cancelAnimationFrame(animId);
      animId = null;
    }
  }

  // ===================== UI =====================
  const Q = (s) => document.querySelector(s);

  function createUI() {
    if (Q("#cp-root")) return;
    const d = document.createElement("div");
    d.id = "cp-root";
    d.innerHTML = `
        <style>
            @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700;800&display=swap');
            #cp-root{position:fixed;top:8px;left:8px;width:300px;
              background:rgba(12,8,30,.96);border:1px solid rgba(168,85,247,.18);border-radius:14px;
              font-family:'JetBrains Mono','Cascadia Code',monospace;font-size:11px;color:#d0d0e0;
              z-index:999997;overflow:hidden;backdrop-filter:blur(16px);
              box-shadow:0 0 30px rgba(168,85,247,.06),0 4px 20px rgba(0,0,0,.4);user-select:none;resize:both}
            #cp-hd{display:flex;align-items:center;justify-content:space-between;padding:7px 12px;
              background:linear-gradient(90deg,rgba(168,85,247,.08),transparent);
              border-bottom:1px solid rgba(255,255,255,.04);cursor:move}
            #cp-hd .t{font-weight:800;font-size:11px;letter-spacing:1.2px;
              background:linear-gradient(90deg,#a855f7,#ec4899);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
            .cp-btn{cursor:pointer;color:#555;font-size:13px;padding:2px 5px;border-radius:3px;display:inline-block;background:none;border:none;font-family:inherit}
            .cp-btn:hover{background:rgba(168,85,247,.15);color:#a855f7}

            #cp-live{text-align:center;padding:6px 8px 2px}
            .cp-mv{font-size:32px;font-weight:900;font-variant-numeric:tabular-nums;line-height:1}
            .cp-mv.growing{color:#4cff4c;text-shadow:0 0 10px rgba(76,255,76,.06)}
            .cp-mv.crashed{color:#ff4444;text-shadow:0 0 10px rgba(255,68,68,.06)}
            .cp-mv.waiting{color:#444}

            #cp-ai-box{margin:6px 8px;padding:14px 12px;border-radius:12px;text-align:center;
              background:rgba(168,85,247,.04);border:2px solid rgba(168,85,247,.12);transition:all .3s}
            #cp-ai-box.predicting{border-color:rgba(168,85,247,.3);animation:cpPulse 1.5s infinite}
            #cp-ai-box.bet{background:rgba(76,255,76,.06);border-color:rgba(76,255,76,.3)}
            #cp-ai-box.skip{background:rgba(255,40,40,.06);border-color:rgba(255,50,50,.3)}
            #cp-ai-box.caution{background:rgba(255,200,0,.05);border-color:rgba(255,200,0,.25)}
            #cp-ai-box.wait{background:rgba(100,100,120,.04);border-color:rgba(100,100,120,.15)}
            #cp-ai-box.error{background:rgba(255,30,30,.04);border-color:rgba(255,50,50,.2)}
            @keyframes cpPulse{0%,100%{box-shadow:0 0 4px rgba(168,85,247,.05)}50%{box-shadow:0 0 20px rgba(168,85,247,.12)}}

            #cp-pred-val{font-size:28px;font-weight:900;letter-spacing:1px;margin-bottom:4px}
            #cp-pred-val.bet{color:#4cff4c}
            #cp-pred-val.skip{color:#ff4444}
            #cp-pred-val.caution{color:#ffc800}
            #cp-pred-val.wait{color:#666}
            #cp-pred-val.error{color:#ff4444}
            #cp-pred-val.predicting{color:#a855f7}

            #cp-advice{font-size:13px;font-weight:800;letter-spacing:.8px;margin-bottom:6px}
            #cp-advice.bet{color:#4cff4c}
            #cp-advice.skip{color:#ff4444}
            #cp-advice.caution{color:#ffc800}
            #cp-advice.wait{color:#666}

            #cp-conf-bar{height:5px;background:rgba(255,255,255,.04);border-radius:3px;overflow:hidden;margin:6px 0 4px}
            #cp-conf-fill{height:100%;width:0%;transition:width .5s,background .3s;border-radius:3px}
            #cp-conf-label{font-size:8px;color:#666;letter-spacing:.3px}

            #cp-reason{font-size:9px;color:#888;margin-top:5px;line-height:1.3;font-weight:400;
              max-height:32px;overflow:hidden;text-overflow:ellipsis}
            #cp-cashout-hint{font-size:9px;color:#a855f7;margin-top:3px;font-weight:600}

            #cp-stats{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:rgba(255,255,255,.01);margin-top:4px}
            .cp-st{background:rgba(12,8,30,.9);padding:4px 2px;text-align:center}
            .cp-stl{font-size:7px;color:#555;text-transform:uppercase;letter-spacing:.4px}
            .cp-stv{font-size:10px;font-weight:700;margin-top:1px;font-variant-numeric:tabular-nums;color:#d0d0e0}

            #cp-history{display:flex;gap:2px;padding:4px 8px;flex-wrap:wrap;border-top:1px solid rgba(255,255,255,.02)}
            .cp-rh{padding:1px 4px;border-radius:3px;font-size:8px;font-weight:700;font-variant-numeric:tabular-nums}
            .cp-rh.lo{background:rgba(255,68,68,.08);color:#f66}
            .cp-rh.mi{background:rgba(255,170,0,.08);color:#fa0}
            .cp-rh.hi{background:rgba(76,255,76,.08);color:#4c4}
            .cp-rh.now{outline:1px solid rgba(168,85,247,.5)}

            #cp-log{max-height:100px;overflow-y:auto;font-size:9px;border-top:1px solid rgba(255,255,255,.03)}
            #cp-log::-webkit-scrollbar{width:3px}#cp-log::-webkit-scrollbar-thumb{background:#333;border-radius:2px}
            .cp-le{padding:2px 8px;border-bottom:1px solid rgba(255,255,255,.01);display:flex;gap:4px;align-items:baseline}
            .cp-le.s{border-left:2px solid #ff4444}.cp-le.i{border-left:2px solid #a855f7}.cp-le.w{border-left:2px solid #ffaa00}.cp-le.g{border-left:2px solid #4cff4c}
            .cp-lt{color:#444;font-size:7px;min-width:48px;font-variant-numeric:tabular-nums}
            .cp-lx{color:#888;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}

            #cp-ft{padding:4px 8px;display:flex;justify-content:space-between;align-items:center;
              border-top:1px solid rgba(255,255,255,.03);font-size:8px;color:#444}
            .cp-dot{width:5px;height:5px;border-radius:50%;display:inline-block;margin-right:3px}
            .cp-dot.on{background:#4cff4c;box-shadow:0 0 4px #4cff4c88}
            .cp-dot.wait{background:#fa0}
            .cp-dot.off{background:#ff4444}

            /* Config modal */
            #cp-cfg-modal{display:none;position:absolute;top:0;left:0;right:0;bottom:0;
              background:rgba(0,0,0,.85);z-index:10;padding:16px;border-radius:14px}
            #cp-cfg-modal.show{display:block}
            #cp-cfg-modal h3{margin:0 0 12px;font-size:13px;color:#a855f7;font-weight:800;letter-spacing:1px}
            .cp-cfg-row{margin-bottom:8px}
            .cp-cfg-row label{display:block;font-size:8px;color:#888;text-transform:uppercase;letter-spacing:.5px;margin-bottom:2px}
            .cp-cfg-row input,.cp-cfg-row select{width:100%;padding:5px 8px;background:rgba(255,255,255,.05);border:1px solid rgba(168,85,247,.2);
              border-radius:6px;color:#e0e0e0;font-family:inherit;font-size:11px;outline:none;box-sizing:border-box}
            .cp-cfg-row input:focus{border-color:#a855f7}
            .cp-cfg-btns{display:flex;gap:6px;margin-top:12px}
            .cp-cfg-btns button{flex:1;padding:6px;border:none;border-radius:6px;font-family:inherit;font-size:11px;font-weight:700;cursor:pointer}
            .cp-cfg-save{background:#a855f7;color:#fff}
            .cp-cfg-save:hover{background:#9333ea}
            .cp-cfg-cancel{background:rgba(255,255,255,.08);color:#aaa}
            .cp-cfg-cancel:hover{background:rgba(255,255,255,.12)}
            .cp-cfg-danger{background:rgba(255,50,50,.15);color:#ff6666;font-size:9px;margin-top:8px;padding:6px;border-radius:6px;text-align:center;cursor:pointer}
            .cp-cfg-danger:hover{background:rgba(255,50,50,.25)}
        </style>
        <div id="cp-hd"><span class="t">🧠 CRASH PREDICT AI</span>
          <span><button class="cp-btn" id="cp-cfg-btn">⚙</button><button class="cp-btn" id="cp-min">—</button><button class="cp-btn" id="cp-x">✕</button></span></div>
        <div id="cp-body">
          <div id="cp-live"><div class="cp-mv waiting" id="cp-mult">—</div></div>

          <div id="cp-ai-box" class="wait">
            <div id="cp-pred-val" class="wait">COLLECTING DATA</div>
            <div id="cp-advice" class="wait">OBSERVE</div>
            <div id="cp-conf-bar"><div id="cp-conf-fill"></div></div>
            <div id="cp-conf-label">CONFIDENCE: —</div>
            <div id="cp-reason">Waiting for game data...</div>
            <div id="cp-cashout-hint"></div>
          </div>

          <div id="cp-stats">
            <div class="cp-st"><div class="cp-stl">Stored</div><div class="cp-stv" id="si-stored">0</div></div>
            <div class="cp-st"><div class="cp-stl">AI Acc</div><div class="cp-stv" id="si-acc">—</div></div>
            <div class="cp-st"><div class="cp-stl">Avg Reward</div><div class="cp-stv" id="si-api">1.00</div></div>
            <div class="cp-st"><div class="cp-stl">Avg Crash</div><div class="cp-stv" id="si-avg">—</div></div>
          </div>
          <div id="cp-history"></div>
          <div id="cp-log"></div>
          <div id="cp-ft">
            <span><span class="cp-dot wait" id="cp-wsd"></span><span id="cp-wst">Waiting...</span></span>
            <span id="cp-phase">—</span>
          </div>
        </div>

        <div id="cp-cfg-modal">
          <h3>⚙ Configuration</h3>
          <div class="cp-cfg-row"><label>Gemini API Key</label><input id="cfg-key" type="password" value="${CFG.API_KEY}"></div>
          <div class="cp-cfg-row"><label>Model</label><input id="cfg-model" value="${CFG.MODEL}"></div>
          <div class="cp-cfg-row"><label>KNN Neighbors (K)</label><input id="cfg-k" type="number" min="1" max="20" value="${CFG.KNN_K}"></div>
          <div class="cp-cfg-row"><label>Min Rounds Before Prediction</label><input id="cfg-min" type="number" min="3" max="100" value="${CFG.MIN_ROUNDS}"></div>
          <div class="cp-cfg-btns">
            <button class="cp-cfg-cancel" id="cfg-cancel">Cancel</button>
            <button class="cp-cfg-save" id="cfg-save">Save</button>
          </div>
          <div style="display:flex;gap:4px;margin-top:8px">
            <div class="cp-cfg-danger" id="cfg-export" style="flex:1;background:rgba(168,85,247,.12);color:#c084fc;margin-top:0">📦 Export</div>
            <div class="cp-cfg-danger" id="cfg-import-btn" style="flex:1;background:rgba(76,255,76,.12);color:#4ade80;margin-top:0">📥 Import</div>
            <input type="file" id="cfg-import-file" accept=".json" style="display:none">
          </div>
          <div class="cp-cfg-danger" id="cfg-clear">🗑 Clear All DB Data</div>
        </div>`;
    document.body.appendChild(d);

    // ---- Dragging ----
    let sx, sy, sl, st;
    Q("#cp-hd").onmousedown = (e) => {
      e.preventDefault();
      sx = e.clientX;
      sy = e.clientY;
      const r = d.getBoundingClientRect();
      sl = r.left;
      st = r.top;
      const mv = (e) => {
        d.style.left = sl + e.clientX - sx + "px";
        d.style.top = st + e.clientY - sy + "px";
        d.style.right = "auto";
      };
      const up = () => {
        document.removeEventListener("mousemove", mv);
        document.removeEventListener("mouseup", up);
      };
      document.addEventListener("mousemove", mv);
      document.addEventListener("mouseup", up);
    };

    // ---- Controls ----
    Q("#cp-x").onclick = destroy;
    let mini = false;
    Q("#cp-min").onclick = () => {
      mini = !mini;
      Q("#cp-body").style.display = mini ? "none" : "";
      d.style.width = mini ? "180px" : "300px";
    };

    // ---- Config modal ----
    Q("#cp-cfg-btn").onclick = () => Q("#cp-cfg-modal").classList.add("show");
    Q("#cfg-cancel").onclick = () =>
      Q("#cp-cfg-modal").classList.remove("show");
    Q("#cfg-save").onclick = () => {
      CFG.API_KEY = Q("#cfg-key").value.trim();
      CFG.MODEL = Q("#cfg-model").value.trim();
      CFG.KNN_K = parseInt(Q("#cfg-k").value) || 5;
      CFG.MIN_ROUNDS = parseInt(Q("#cfg-min").value) || 10;
      saveCfg();
      Q("#cp-cfg-modal").classList.remove("show");
      log("✅ Config saved", "g");
    };
    Q("#cfg-clear").onclick = async () => {
      if (confirm("Delete ALL stored crash data? This cannot be undone.")) {
        await VDB.clearAll();
        S.crashHistory = [];
        log("🗑 Data cleared", "w");
        updateStatsUI();
      }
    };
    Q("#cfg-export").onclick = async () => {
      const json = await VDB.exportJSON();
      const blob = new Blob([json], { type: "application/json" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = "crash-predict-data-" + Date.now() + ".json";
      a.click();
      log("📦 Data exported", "i");
    };

    Q("#cfg-import-btn").onclick = () => Q("#cfg-import-file").click();
    Q("#cfg-import-file").onchange = (e) => {
      const file = e.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = async (ev) => {
        try {
          const count = await VDB.importJSON(ev.target.result);
          log(`📥 Imported ${count} rounds`, "g");
          Q("#cp-cfg-modal").classList.remove("show");
          updateStatsUI();
        } catch (err) {
          log("❌ Import failed: " + err.message, "s");
        }
      };
      reader.readAsText(file);
    };
  }

  // ===================== UI HELPERS =====================
  function log(text, cls) {
    const c = Q("#cp-log");
    if (!c) return;
    const ts =
      new Date().toLocaleTimeString("en-GB", { hour12: false }).substring(3) +
      "." +
      String(new Date().getMilliseconds()).padStart(3, "0");
    const e = document.createElement("div");
    e.className = "cp-le " + (cls || "i");
    e.innerHTML = `<span class="cp-lt">${ts}</span><span class="cp-lx">${text}</span>`;
    c.prepend(e);
    while (c.children.length > 40) c.removeChild(c.lastChild);
  }

  function showPrediction(pred) {
    if (!pred) return;
    const box = Q("#cp-ai-box");
    const val = Q("#cp-pred-val");
    const adv = Q("#cp-advice");
    const fill = Q("#cp-conf-fill");
    const label = Q("#cp-conf-label");
    const reason = Q("#cp-reason");
    const cashout = Q("#cp-cashout-hint");
    if (!box) return;

    const advLower = (pred.advice || "wait").toLowerCase();
    box.className = advLower;
    val.className = "cp-pred-val " + advLower; // Keep id via CSS

    if (pred.advice === "WAIT" || pred.advice === "ERROR") {
      val.textContent = pred.advice === "WAIT" ? "COLLECTING DATA" : "⚠ ERROR";
    } else {
      val.textContent = "PREDICTED: " + (pred.prediction || 0).toFixed(2) + "x";
    }

    adv.textContent = pred.advice || "OBSERVE";
    adv.className = advLower;

    const conf = pred.confidence || 0;
    if (fill) {
      fill.style.width = conf + "%";
      fill.style.background =
        conf >= 70 ? "#4cff4c" : conf >= 40 ? "#ffc800" : "#ff6666";
    }
    if (label) label.textContent = "CONFIDENCE: " + conf + "%";
    if (reason) reason.textContent = pred.reasoning || "";
    if (cashout) {
      cashout.textContent =
        pred.suggestedCashout && pred.advice === "BET"
          ? "💰 Suggested cashout: " + pred.suggestedCashout.toFixed(2) + "x"
          : "";
    }
  }

  function updateHistory() {
    const el = Q("#cp-history");
    if (!el) return;
    el.innerHTML = S.crashHistory
      .slice(0, 20)
      .map((v, i) => {
        const c = v < 1.5 ? "lo" : v < 3 ? "mi" : "hi";
        return `<span class="cp-rh ${c}${i === 0 ? " now" : ""}">${v.toFixed(2)}x</span>`;
      })
      .join("");
  }

  async function updateStatsUI() {
    const stats = await VDB.getStats().catch(() => ({ count: 0, avg: 0 }));
    const all = await VDB._getAll().catch(() => []);

    const avgReward = all.length
      ? all.reduce((sum, r) => sum + (r.rewardScore || 1.0), 0) / all.length
      : 1.0;

    const st = Q("#si-stored");
    if (st) st.textContent = stats.count;
    const av = Q("#si-avg");
    if (av) av.textContent = stats.count ? stats.avg.toFixed(2) + "x" : "—";
    const api = Q("#si-api");
    if (api) api.textContent = avgReward.toFixed(2);

    const acc = Q("#si-acc");
    if (acc) {
      if (AI.total > 0) {
        const pct = ((AI.correct / AI.total) * 100).toFixed(0);
        acc.textContent = pct + "%";
        acc.style.color = pct >= 50 ? "#4cff4c" : "#ff6666";
      } else {
        acc.textContent = "—";
      }
    }
  }
  // ===================== EVENT HANDLER =====================
  function handleEvent(ev, d) {
    const now = Date.now();
    const delta = S.lastEventTime ? now - S.lastEventTime : 0;
    S.lastEventTime = now;

    switch (ev) {
      case "OnRegistration":
        S.gainCoef = d.kx ? d.kx * 1000 : 25;
        S.status = d.s;
        S.stageId = d.l;
        if (d.h) {
          S.crashHistory = d.h.slice(0, 30).map((h) => h.f || 0);
          S.lowStreak = 0;
          for (let i = 0; i < S.crashHistory.length; i++) {
            if (S.crashHistory[i] < CFG.STREAK_LOW) S.lowStreak++;
            else break;
          }
          updateHistory();
        }
        if (d.s === 3) {
          S.coeffStartTime = now - (d.t || 0);
          S.startTime = now - (d.t || 0);
          startAnim();
        }
        log(
          "📡 Registered — s:" + d.s + " rounds:" + (d.h ? d.h.length : 0),
          "i",
        );
        {
          const ph = Q("#cp-phase");
          if (ph) ph.textContent = "s:" + d.s;
        }
        VDB.init()
          .then(() => {
            log("💾 Vector DB ready", "g");
            updateStatsUI();
          })
          .catch((e) => log("❌ DB: " + e, "s"));
        break;

      case "OnStage":
        S.stageId = d.l;
        S.status = 1;
        S.consecutiveProfits = 0;
        S.profitsDeltas = [];
        S.hasCashouts = false;
        S.cashoutCount = 0;
        S.profitCount = 0;
        S.roundEvents = 0;
        S.startTime = null;
        S.tickLeakDetected = false;
        stopAnim();
        {
          const m = Q("#cp-mult");
          if (m) {
            m.textContent = "—";
            m.className = "cp-mv waiting";
          }
        }
        {
          const ph = Q("#cp-phase");
          if (ph) ph.textContent = "stage";
        }
        break;

      case "OnBetting": {
        S.status = 2;
        {
          const ph = Q("#cp-phase");
          if (ph) ph.textContent = "betting";
        }

        // Trigger AI prediction
        if (CFG.AUTO_PREDICT) {
          const box = Q("#cp-ai-box");
          const val = Q("#cp-pred-val");
          if (box) box.className = "predicting";
          if (val) {
            val.textContent = "🔮 PREDICTING...";
            val.className = "cp-pred-val predicting";
          }
          log("🧠 Calling Gemini...", "i");

          AI.predict().then((pred) => {
            if (pred) {
              showPrediction(pred);
              if (pred.advice !== "WAIT" && pred.advice !== "ERROR") {
                log(
                  `🤖 Pred: ${pred.prediction?.toFixed(2)}x [${pred.confidence}%] → ${pred.advice}`,
                  pred.advice === "BET"
                    ? "g"
                    : pred.advice === "SKIP"
                      ? "s"
                      : "w",
                );
              } else {
                log("⏳ " + (pred.reasoning || "Waiting..."), "i");
              }
              updateStatsUI();
            }
          });
        }
        break;
      }

      case "OnStart":
        S.status = 3;
        S.coeffStartTime = now;
        S.startTime = now;
        startAnim();
        {
          const ph = Q("#cp-phase");
          if (ph) ph.textContent = "flying";
        }
        break;

      case "OnProfits":
        S.consecutiveProfits++;
        S.profitCount++;
        S.profitsDeltas.push(delta);
        S.roundEvents++;
        if (delta <= CFG.P1_DELTA && S.profitsDeltas.length > 1) {
          S.tickLeakDetected = true;
        }
        break;

      case "OnCashouts":
        S.consecutiveProfits = 0;
        S.hasCashouts = true;
        S.cashoutCount++;
        S.roundEvents++;
        break;

      case "OnCrash": {
        S.status = 4;
        S.isCrashed = true;
        stopAnim();
        const crashVal = d.f || 0;
        const duration = S.startTime ? now - S.startTime : 0;

        // Show crash in multiplier
        {
          const el = Q("#cp-mult");
          if (el) {
            el.textContent = crashVal.toFixed(2) + "x";
            el.className = "cp-mv crashed";
          }
        }
        {
          const ph = Q("#cp-phase");
          if (ph) ph.textContent = "crashed";
        }

        // Track AI accuracy and process V2 reward feedback loop
        AI.trackAccuracy(crashVal);

        let accNote = "";
        if (
          AI.lastPred &&
          AI.lastPred.prediction &&
          AI.lastPred.advice !== "WAIT" &&
          AI.lastPred.advice !== "ERROR"
        ) {
          const diff = Math.abs(AI.lastPred.prediction - crashVal);
          accNote = ` | pred was ${AI.lastPred.prediction.toFixed(2)}x (off by ${diff.toFixed(2)})`;

          // V2 Feedback loop: reward or punish the KNN rounds that led to this prediction
          if (AI.lastPred.knnIds && AI.lastPred.knnIds.length > 0) {
            let multiplier = 1.0;
            const gap = crashVal - AI.lastPred.prediction; // positive means safe, negative means busted

            if (crashVal === 1.0 && AI.lastPred.prediction > 1.0) {
              // Penalty 1: Missed an instant 1.00x crash. Huge penalty to learn to predict 1.00x.
              multiplier = 0.6;
              log("🔥 Insta-crash (1.00x) missed! HUGE penalty.", "w");
            } else if (gap < 0) {
              // Penalty 2: Predicted higher than actual crash (User lost)
              multiplier = 0.8;
              log("❌ Busted pred! Heavy penalty to vectors.", "w");
            } else {
              // Reward: Predicted lower than or equal to actual crash (User won)
              if (gap <= 0.2) {
                multiplier = 1.1; // Very good
                log("🎯 Perfect pred (gap <= 0.2)! High reward.", "g");
              } else if (gap <= 0.4) {
                multiplier = 1.05; // Good
                log("✅ Good pred (gap <= 0.4). Medium reward.", "g");
              } else if (gap <= 0.8) {
                multiplier = 1.02; // Fair
                log("🆗 Fair pred (gap <= 0.8). Low reward.", "i");
              } else if (gap > 1.0) {
                // Penalty 3: Gap too big, predicted way too low
                multiplier = 0.6;
                log("📉 Huge gap (> 1.0). HUGE Penalty.", "w");
              }
            }

            if (multiplier !== 1.0) {
              VDB.updateRewards(AI.lastPred.knnIds, multiplier)
                .then(() => updateStatsUI())
                .catch((e) => log("❌ Reward Error: " + e, "s"));
            }
          }
        }

        // Update streaks
        if (crashVal < CFG.STREAK_LOW) {
          S.lowStreak++;
          S.highStreak = 0;
        } else if (crashVal >= CFG.STREAK_HIGH) {
          S.highStreak++;
          S.lowStreak = 0;
        } else {
          S.lowStreak = 0;
          S.highStreak = 0;
        }

        // Update history
        if (crashVal > 0) {
          S.crashHistory.unshift(crashVal);
          if (S.crashHistory.length > 30) S.crashHistory.pop();
        }
        updateHistory();

        // Store in vector DB
        const roundRec = {
          crashValue: crashVal,
          profitsDeltas: [...S.profitsDeltas],
          profitCount: S.profitCount,
          cashoutCount: S.cashoutCount,
          roundDuration: duration,
          lowStreak: S.lowStreak,
          highStreak: S.highStreak,
          tickLeak: S.tickLeakDetected,
        };
        VDB.addRound(roundRec)
          .then(() => {
            updateStatsUI();
          })
          .catch((e) => log("❌ Store: " + e, "s"));

        // Log result
        log(
          `💥 Crash: ${crashVal.toFixed(2)}x | dur: ${(duration / 1000).toFixed(1)}s${accNote}`,
          crashVal < 1.5 ? "s" : crashVal > 3 ? "g" : "w",
        );

        break;
      }
    }
  }

  // ===================== SIGNALR PARSER =====================
  function parseMsg(raw) {
    const m = [];
    if (typeof raw !== "string") return m;
    for (const p of raw.split("\x1e")) {
      if (!p.trim()) continue;
      try {
        const o = JSON.parse(p);
        if (o.type === 1 && o.target)
          m.push({ ev: o.target, d: o.arguments?.[0] || {} });
      } catch (e) {}
    }
    return m;
  }

  // ===================== WEBSOCKET HOOK =====================
  const OrigWS = window.WebSocket;

  function hookSocket(ws, src) {
    if (ws.__cp_hooked) return;
    ws.__cp_hooked = true;
    S.ws = ws;
    if (document.body && !Q("#cp-root")) createUI();
    {
      const dot = Q("#cp-wsd");
      if (dot) dot.className = "cp-dot on";
    }
    {
      const wst = Q("#cp-wst");
      if (wst) wst.textContent = "Live";
    }
    log("📡 Connected (" + src + ")", "g");

    ws.addEventListener("message", (e) => {
      for (const m of parseMsg(e.data)) handleEvent(m.ev, m.d);
    });
    ws.addEventListener("close", () => {
      S.ws = null;
      {
        const dot = Q("#cp-wsd");
        if (dot) dot.className = "cp-dot off";
      }
      {
        const wst = Q("#cp-wst");
        if (wst) wst.textContent = "Closed";
      }
      log("❌ Disconnected", "s");
    });
  }

  window.WebSocket = function (url, protocols) {
    const ws = protocols ? new OrigWS(url, protocols) : new OrigWS(url);
    if (url && typeof url === "string" && url.includes("sockets/crash"))
      ws.addEventListener("open", () => hookSocket(ws, "constructor"));
    return ws;
  };
  window.WebSocket.prototype = OrigWS.prototype;
  window.WebSocket.CONNECTING = OrigWS.CONNECTING;
  window.WebSocket.OPEN = OrigWS.OPEN;
  window.WebSocket.CLOSING = OrigWS.CLOSING;
  window.WebSocket.CLOSED = OrigWS.CLOSED;

  // ===================== INIT / DESTROY =====================
  function destroy() {
    window.__cpActive = false;
    window.WebSocket = OrigWS;
    stopAnim();
    const r = Q("#cp-root");
    if (r) r.remove();
  }

  window.__predict_destroy = destroy;

  window.__predict_debug = async function () {
    const stats = await VDB.getStats();
    const recent = await VDB.getRecent(5);
    return {
      config: { ...CFG, API_KEY: CFG.API_KEY.substring(0, 8) + "..." },
      dbStats: stats,
      recentRounds: recent,
      aiState: {
        callCount: AI.callCount,
        accuracy: AI.total
          ? ((AI.correct / AI.total) * 100).toFixed(1) + "%"
          : "N/A",
        lastPred: AI.lastPred,
      },
      gameState: {
        status: S.status,
        lowStreak: S.lowStreak,
        highStreak: S.highStreak,
        historyLen: S.crashHistory.length,
      },
    };
  };

  if (document.readyState === "loading")
    document.addEventListener("DOMContentLoaded", createUI);
  else createUI();
})();
