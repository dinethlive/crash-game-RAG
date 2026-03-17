# 🧠 Crash Predict AI — Gemini-Powered Prediction

![Crash Predict AI UI](screenshot/crash-predict-ai-ss-1.png)

> **⚠️ EDUCATIONAL PURPOSES ONLY — SEE DISCLAIMER SECTION**

An AI-powered crash game prediction system that combines local vector database storage with Gemini 2.5 Flash for intelligent predictions. This is a sophisticated userscript that intercepts game events, builds feature vectors from historical rounds, and uses KNN similarity search augmented with LLM reasoning.

---

## 📋 Overview

Crash Predict AI is a browser-based userscript (Tampermonkey/Violentmonkey) that:

1. **Hooks into WebSocket/SignalR** traffic to capture real-time crash game events
2. **Extracts 9-dimensional feature vectors** from each round (crash value, deltas, streaks, etc.)
3. **Stores vectors in IndexedDB** — a local vector database that persists across sessions
4. **Runs K-Nearest Neighbors (KNN)** queries to find historically similar rounds
5. **Augments prompts with retrieved context** and sends to Gemini 2.5 Flash API
6. **Displays predictions** with confidence scores, betting advice, and suggested cashout points
7. **Implements a feedback loop** — rewards/punishes historical rounds based on prediction accuracy

### Supported Platforms

- `https://melbet-srilanka.com/games-frame/games/371*`
- `https://*.melbet*.com/games-frame/games/371*`

---

## 🏗️ Architecture Breakdown

### 1. Vector Database (IndexedDB)

The script uses **IndexedDB** as a local vector database with the following schema:

```
DB Name: crash-predict-db
Store: rounds
```

Each stored record contains:

| Field          | Description                                        |
| -------------- | -------------------------------------------------- |
| `id`           | Auto-increment primary key                         |
| `crashValue`   | Actual crash multiplier (e.g., 2.45x)              |
| `features`     | 9D raw feature vector                              |
| `norm`         | Normalized feature vector (0-1 range)              |
| `lowStreak`    | Consecutive low crashes (<1.5x) before this round  |
| `highStreak`   | Consecutive high crashes (>5.0x) before this round |
| `avgDelta`     | Average time delta between profit events (ms)      |
| `minDelta`     | Minimum time delta between profit events (ms)      |
| `tickLeak`     | Boolean flag for rapid tick detection              |
| `duration`     | Round duration in milliseconds                     |
| `profitCount`  | Number of profit events in the round               |
| `cashoutCount` | Number of cashouts in the round                    |
| `rewardScore`  | Learned reward score (V2 feature, starts at 1.0)   |
| `ts`           | Unix timestamp                                     |

**Normalization Ranges:**

| Feature              | Min | Max   |
| -------------------- | --- | ----- |
| `ln(crashValue)`     | 0   | 3.6   |
| `avgDelta` (ms)      | 0   | 2000  |
| `minDelta` (ms)      | 0   | 1000  |
| `stdDelta` (ms)      | 0   | 800   |
| `cashoutRatio`       | 0   | 1     |
| `roundDuration` (ms) | 0   | 60000 |
| `lowStreak`          | 0   | 10    |
| `highStreak`         | 0   | 10    |
| `tickLeak`           | 0   | 1     |

### 2. KNN Retrieval (Cosine Similarity)

The system uses **cosine similarity** to find historically similar rounds:

```javascript
cosine(a, b) {
    let dot = 0, ma = 0, mb = 0;
    for (let i = 0; i < a.length; i++) {
        dot += a[i] * b[i];
        ma += a[i] * a[i];
        mb += b[i] * b[i];
    }
    ma = Math.sqrt(ma); mb = Math.sqrt(mb);
    return (ma && mb) ? dot / (ma * mb) : 0;
}
```

**Scoring Formula:**

```
finalScore = rawSimilarity × rewardMultiplier × timeDecay
```

- **Reward Multiplier**: Weighted by learned `rewardScore` (log-scaled, range 0.5–2.0)
- **Time Decay**: Exponential decay with ~2-hour half-life — recency bias
- **Adaptive K**: If no matches exceed 0.85 similarity threshold, returns only the single best match

### 3. Gemini API Integration

**Model**: `gemini-2.5-flash`

**API Endpoint:**

```
https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent
```

**Request Configuration:**

```javascript
{
    responseMimeType: "application/json",
    temperature: 0.2 - 0.7,  // Dynamic scaling based on volatility
    maxOutputTokens: 512,
    thinkingConfig: { thinkingBudget: 0 }  // Disabled for clean JSON
}
```

**Expected Response Format:**

```json
{
  "prediction": 2.45,
  "confidence": 68,
  "advice": "BET",
  "reasoning": "After 3 consecutive low crashes, mean reversion suggests higher multiplier",
  "suggestedCashout": 1.85
}
```

### 4. WebSocket/SignalR Hook

The script monkey-patches `window.WebSocket` to intercept SignalR messages:

1. **Message Parsing**: Splits on `\x1e` (SignalR record separator), parses JSON
2. **Event Types**:
   - `OnRegistration`: Initializes game state, loads crash history
   - `OnStage`: New round starts
   - `OnBetting`: Betting phase begins → **triggers AI prediction**
   - `OnStart`: Round goes live (multiplier rising)
   - `OnProfits`: Tracks profit events and calculates deltas
   - `OnCashouts`: Tracks cashout events
   - `OnCrash`: Round ends → stores features in DB, tracks accuracy

### 5. UI Components

- **Draggable panel** with JetBrains Mono font
- **Live multiplier display** with color-coded states (growing/crashed/waiting)
- **AI prediction box** with confidence bar and advice
- **Statistics panel**: Stored rounds, AI accuracy, avg reward score, avg crash
- **Crash history**: Visual representation of recent 20 rounds
- **Log panel**: Real-time event logging
- **Config modal**: API key, model, KNN-K, minimum rounds

---

## 🔬 How the RAG-Like System Works

### Step 1: Feature Extraction (9D Vectors)

When a round crashes, the system extracts these features:

```javascript
features = [
  Math.log(crashValue), // f0: ln(crash) - log scale for distribution
  avgDelta, // f1: average time between profit events (ms)
  minDelta, // f2: minimum time delta (ms)
  stdDelta, // f3: standard deviation of deltas
  cashoutRatio, // f4: cashouts / total events
  roundDuration, // f5: total round time (ms)
  lowStreak, // f6: consecutive <1.5x crashes
  highStreak, // f7: consecutive >5.0x crashes
  tickLeak ? 1 : 0, // f8: rapid tick detection
];
```

### Step 2: Storage in IndexedDB

Features are:

1. Built into a record object
2. Normalized (0-1 scaling)
3. Stored in IndexedDB with timestamp

### Step 3: KNN Query on New Prediction

Before each round, when betting opens:

1. Get the **last round's features** as the query vector
2. **Normalize** the query vector
3. **Compute cosine similarity** against all stored vectors
4. **Apply weighting**:
   - Reward score multiplier (learned from past predictions)
   - Time decay (2-hour half-life)
5. **Sort** by weighted score, take top-K (default 5)
6. **Adaptive filtering**: If no matches >0.85 similarity, use only the best match

### Step 4: Prompt Augmentation

The retrieved KNN rounds are injected into the prompt:

```
You are a statistical analyst for a crash/multiplier game...

RECENT CRASHES: [2.45, 1.12, 3.20, 1.05, ...]

CURRENT STATE:
- Consecutive low crashes (<1.5x): 3
- Consecutive high crashes (>5.0x): 0

ALL-TIME STATISTICS (156 rounds):
- Mean: 2.45x | Median: 1.85x
- Range: 1.00x – 25.30x

5 MOST SIMILAR ROUNDS (by latency/feature cosine similarity):
  #1 [sim=0.923] crashed=2.35x | avgΔ=450ms | minΔ=120ms | tickLeak=no | lowStrk=2 | dur=8.5s
  #2 [sim=0.891] crashed=2.55x | avgΔ=380ms | minΔ=95ms | tickLeak=no | lowStrk=3 | dur=7.2s
  ...

ANALYSIS GUIDELINES:
- The game uses a provably fair RNG — no deterministic pattern exists
- However, short-term statistical clustering and mean-reversion tendencies are observable
- Latency features (avgΔ, minΔ, tickLeak) often correlate with crash timing
- After streaks of low crashes, slightly higher results tend to follow (regression to mean)
...

Return ONLY valid JSON...
```

### Step 5: Gemini 2.5 Flash Generation

The LLM receives the augmented prompt and generates a JSON prediction. The model is instructed to:

- Be honest about uncertainty (high confidence should be rare)
- Consider mean reversion after streaks
- Factor in latency patterns
- Return structured JSON output

### Step 6: Feedback Loop (V2)

After each crash, the system evaluates the prediction:

| Scenario                                 | Multiplier    |
| ---------------------------------------- | ------------- |
| Missed 1.00x instant crash               | 0.60 (punish) |
| Predicted higher than actual (user lost) | 0.80 (punish) |
| Gap >1.0 (predicted too low)             | 0.60 (punish) |
| Gap ≤0.2 (excellent)                     | 1.10 (reward) |
| Gap ≤0.4 (good)                          | 1.05 (reward) |
| Gap ≤0.8 (fair)                          | 1.02 (reward) |

The reward/penalty applies to the `rewardScore` of the KNN rounds that contributed to that prediction, creating a **learning system** that weights more accurate historical patterns higher.

---

## 💡 Behind the Thinking

### Why Local Vector DB Instead of Server-Side?

1. **Privacy**: No data leaves the browser
2. **Latency**: Instant KNN queries without network round-trip
3. **Persistence**: Survives page refreshes via IndexedDB
4. **No infrastructure**: No backend server required

### Why KNN + LLM Hybrid?

- **KNN** provides concrete, similar historical cases — the "evidence"
- **LLM** provides reasoning and pattern recognition across multiple dimensions
- Together: grounded reasoning with statistical backing

Pure LLM-only predictions would lack historical grounding. Pure KNN-only would lack nuanced reasoning.

### Dynamic Temperature Scaling

```javascript
if (stdDev > 4.0)
  temp = 0.7; // High volatility → creative outlier prediction
else if (stdDev < 1.0)
  temp = 0.2; // Dead streaks → stiff logic
else temp = 0.4; // Default
```

- **High temp**: When recent rounds are erratic, the model takes more risks
- **Low temp**: When rounds are stable/conservative, the model follows the pattern

### Reward Scoring System

The V2 feedback loop creates a **self-improving system**:

- Historical rounds that correctly predict get boosted weight
- Poor predictors get penalized
- Over time, the system "learns" which patterns are more reliable

### Time Decay (Recency Bias)

```javascript
const hoursOld = (now - r.ts) / (1000 * 60 * 60);
const timeDecay = Math.exp(-hoursOld / 2); // ~2 hour half-life
```

- Recent rounds matter more (game dynamics may change)
- Old rounds slowly lose influence
- Prevents stale patterns from dominating

---

## ⚠️ EDUCATIONAL DISCLAIMER

### 🚨 IMPORTANT — READ BEFORE USE

> **THIS TOOL IS FOR EDUCATIONAL AND RESEARCH PURPOSES ONLY**

1. **Gambling Involves Real Financial Risk**: This tool interacts with real-money gambling platforms. Using this script may result in the loss of your deposited funds.

2. **No Guarantee of Profits**: Past performance does not guarantee future results. This script provides predictions based on statistical patterns, but crash games use provably fair RNG — no deterministic pattern can reliably predict outcomes.

3. **Past Performance ≠ Future Results**: Even if the AI shows "accuracy" on past data, this does not mean it will predict future crashes accurately.

4. **Use at Your Own Risk**: The author(s) of this code assume no responsibility for any financial losses incurred while using this tool.

5. **Legal Restrictions**:
   - Ensure that using such tools is legal in your jurisdiction
   - Many gambling jurisdictions prohibit the use of automated prediction tools
   - Check the Terms of Service of the gambling platform

6. **Responsible Gambling**:
   - Never gamble more than you can afford to lose
   - If you feel you have a gambling problem, seek help from professional organizations
   - This tool should not be used as a sole basis for financial decisions

---

## 📖 Usage Instructions

### Prerequisites

1. **Browser Extension**: Tampermonkey or Violentmonkey installed
2. **Gemini API Key**: Get one from [Google AI Studio](https://aistudio.google.com/app/apikey)

### Installation

1. Open Tampermonkey dashboard → "Create a new script"
2. Paste the contents of `crash-predict-ai.js`
3. Save (Ctrl+S)
4. Navigate to a supported Melbet crash game URL

### Initial Setup

1. Click the **⚙ gear icon** in the UI panel
2. Enter your **Gemini API Key**
3. Adjust settings:
   - **KNN K**: Number of similar rounds to retrieve (default: 5)
   - **Min Rounds**: Minimum rounds before predictions start (default: 10)
4. Click **Save**

### How It Works

1. **Data Collection Phase**: The script collects rounds automatically. Watch the "Stored" counter increase.
2. **Prediction Phase**: Once minimum rounds reached, predictions appear during the betting phase.
3. **Feedback Loop**: After each crash, the system updates reward scores based on prediction accuracy.

### Debug Functions

```javascript
// View internal state
window.__predict_debug();

// Destroy/remove the script
window.__predict_destroy();
```

### Data Management

- **Export**: Save your collected data as JSON for backup
- **Import**: Restore data from a previous export
- **Clear**: Delete all stored rounds (irreversible)

---

## 🔧 Technical Notes

### Dependencies

- None (pure vanilla JavaScript)
- IndexedDB (browser native)
- WebSocket API (browser native)
- Google Gemini API (requires key)

### Browser Compatibility

- Chrome/Edge/Firefox/Safari (modern versions)
- IndexedDB support required

### Performance Considerations

- KNN queries are O(n) where n = stored rounds
- For 1000+ rounds, consider increasing K slightly
- The script runs prediction async, doesn't block the UI

---

## 📄 License

This code is provided as-is for educational purposes. Use at your own risk.

---

_Generated for the crash-game-RAG project_
