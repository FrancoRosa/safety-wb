/* admin.js — "Usage Statistics" dashboard for /admin/
 * Reads the sessions + person sightings the tracker writes to IndexedDB
 * (stats-db.js) and charts them per day and time window with Recharts.
 * The data lives in this browser only, so open this page on the device
 * that runs the tracker.
 */

const h = React.createElement;
const { useState, useEffect, useMemo } = React;
const {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceArea,
  ScatterChart,
  Scatter,
} = Recharts;

// Column width of the activity chart: 15 min on wide screens, hourly on
// phones where 48 columns would shrink to slivers.
const NARROW_QUERY = "(max-width: 640px)";
const bucketMinutes = (narrow) => (narrow ? 60 : 15);
const bucketName = (min) => (min === 60 ? "hour" : `${min} min`);
const MAX_SCATTER_POINTS = 5000;
const REFRESH_MS = 30000;

const COLORS = {
  accent: "#ffc800",
  surface: "#131007",
  field: "#171208",
  grid: "rgba(255, 200, 20, 0.10)",
  axis: "#a89a72",
  band: "rgba(243, 236, 216, 0.07)",
  cursor: "rgba(255, 200, 0, 0.06)",
};

const VIEW_LABELS = { flat: "Normal", 360: "360° view", planet: "Little planet" };

// ---------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------

const pad = (n) => String(n).padStart(2, "0");
const toDateInput = (d) =>
  `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
const fromDateInput = (s) => {
  const [y, m, d] = s.split("-").map(Number);
  return new Date(y, m - 1, d);
};
const hhmm = (t) => {
  const d = new Date(t);
  return `${pad(d.getHours())}:${pad(d.getMinutes())}`;
};
const fmtDuration = (ms) => {
  const m = Math.round(ms / 60000);
  return m < 60 ? `${m} min` : `${Math.floor(m / 60)} h ${pad(m % 60)} min`;
};
const fmtDateTime = (t) =>
  new Date(t).toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });

// Multiple tabs / devices restarts can overlap; merge so open time isn't
// double counted.
function mergeIntervals(list) {
  const sorted = list.slice().sort((a, b) => a[0] - b[0]);
  const out = [];
  for (const [a, b] of sorted) {
    const last = out[out.length - 1];
    if (last && a <= last[1]) last[1] = Math.max(last[1], b);
    else out.push([a, b]);
  }
  return out;
}

function summarize(data, from, to, bucketMin) {
  const BUCKET_MS = bucketMin * 60000;
  const intervals = mergeIntervals(
    data.sessions
      .map((s) => [Math.max(s.start, from), Math.min(s.lastSeen, to)])
      .filter(([a, b]) => b > a),
  );
  const openMs = intervals.reduce((sum, [a, b]) => sum + (b - a), 0);

  const n = Math.ceil((to - from) / BUCKET_MS);
  const rows = Array.from({ length: n }, (_, i) => {
    const t = from + i * BUCKET_MS;
    let open = 0;
    for (const [a, b] of intervals) {
      open += Math.max(0, Math.min(b, t + BUCKET_MS) - Math.max(a, t));
    }
    return { t, end: t + BUCKET_MS, bucketMin, label: hhmm(t), persons: 0, openMin: Math.round(open / 60000), tracks: new Set() };
  });
  const perSession = new Map();
  for (const s of data.sightings) {
    const i = Math.floor((s.t - from) / BUCKET_MS);
    if (i >= 0 && i < n) rows[i].tracks.add(s.track);
    if (!perSession.has(s.session)) perSession.set(s.session, { tracks: new Set(), samples: 0 });
    const ps = perSession.get(s.session);
    ps.tracks.add(s.track);
    ps.samples += 1;
  }
  for (const r of rows) {
    r.persons = r.tracks.size;
    delete r.tracks;
  }

  // App-open bands, snapped to the 15-min categories of the bar chart.
  const bands = intervals.map(([a, b]) => ({
    x1: rows[Math.floor((a - from) / BUCKET_MS)].label,
    x2: rows[Math.min(n - 1, Math.ceil((b - from) / BUCKET_MS) - 1)].label,
  }));

  const busiest = rows.reduce((best, r) => (r.persons > (best ? best.persons : 0) ? r : best), null);
  const people = new Set(data.sightings.map((s) => s.track)).size;

  const sessions = data.sessions
    .slice()
    .sort((a, b) => a.start - b.start)
    .map((s) => ({
      ...s,
      people: perSession.get(s.id)?.tracks.size ?? 0,
      samples: perSession.get(s.id)?.samples ?? 0,
      live: Date.now() - s.lastSeen < 90000,
    }));

  let points = data.sightings;
  if (points.length > MAX_SCATTER_POINTS) {
    const step = points.length / MAX_SCATTER_POINTS;
    points = Array.from({ length: MAX_SCATTER_POINTS }, (_, i) => points[Math.floor(i * step)]);
  }

  return { rows, bands, openMs, busiest, people, sessions, points, total: data.sightings.length };
}

function exportCsv(data, dateStr) {
  const lines = ["time,session,track,x,y,score,view"];
  for (const s of data.sightings) {
    lines.push([new Date(s.t).toISOString(), s.session, s.track, s.x, s.y, s.score, s.view].join(","));
  }
  const blob = new Blob([lines.join("\n")], { type: "text/csv" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `sightings-${dateStr}.csv`;
  a.click();
  URL.revokeObjectURL(a.href);
}

// ---------------------------------------------------------------------
// Components
// ---------------------------------------------------------------------

function Tile({ label, value, sub }) {
  return h(
    "div",
    { className: "tile" },
    h("div", { className: "tile-label" }, label),
    h("div", { className: "tile-value" }, value),
    sub ? h("div", { className: "tile-sub" }, sub) : null,
  );
}

function BarTip({ active, payload }) {
  if (!active || !payload || !payload.length) return null;
  const r = payload[0].payload;
  return h(
    "div",
    { className: "tip" },
    h("div", { className: "tip-title" }, `${r.label} – ${hhmm(r.end)}`),
    h("div", null, h("b", null, r.persons), r.persons === 1 ? " person detected" : " people detected"),
    h("div", null, `App open ${r.openMin} of ${r.bucketMin} min`),
  );
}

function ScatterTip({ active, payload }) {
  if (!active || !payload || !payload.length) return null;
  const p = payload[0].payload;
  return h(
    "div",
    { className: "tip" },
    h("div", { className: "tip-title" }, fmtDateTime(p.t)),
    h("div", null, `Position ${Math.round(p.x * 100)}% across, ${Math.round(p.y * 100)}% down`),
    h("div", null, `Seen in ${VIEW_LABELS[p.view] || p.view} mode`),
  );
}

function Dot({ cx, cy }) {
  // r=4 mark with a 2px surface ring so overlapping sightings stay legible.
  return h("circle", {
    cx,
    cy,
    r: 4,
    fill: COLORS.accent,
    fillOpacity: 0.7,
    stroke: COLORS.field,
    strokeWidth: 2,
  });
}

function Filters({ date, setDate, fromHour, setFromHour, toHour, setToHour, onExport, canExport }) {
  const today = toDateInput(new Date());
  const shift = (days) => {
    const d = fromDateInput(date);
    d.setDate(d.getDate() + days);
    setDate(toDateInput(d));
  };
  const hourOpts = (min, max) =>
    Array.from({ length: max - min + 1 }, (_, i) =>
      h("option", { key: i, value: min + i }, `${pad(min + i)}:00`),
    );
  return h(
    "div",
    { className: "filters" },
    h("button", { type: "button", onClick: () => shift(-1), "aria-label": "Previous day" }, "◀"),
    h("input", {
      type: "date",
      value: date,
      max: today,
      onChange: (e) => e.target.value && setDate(e.target.value),
    }),
    h(
      "button",
      { type: "button", onClick: () => shift(1), disabled: date >= today, "aria-label": "Next day" },
      "▶",
    ),
    h("button", { type: "button", onClick: () => setDate(today), disabled: date === today }, "Today"),
    h(
      "label",
      null,
      "From",
      h(
        "select",
        { value: fromHour, onChange: (e) => setFromHour(Math.min(Number(e.target.value), toHour - 1)) },
        hourOpts(0, 23),
      ),
    ),
    h(
      "label",
      null,
      "to",
      h(
        "select",
        { value: toHour, onChange: (e) => setToHour(Math.max(Number(e.target.value), fromHour + 1)) },
        hourOpts(1, 24),
      ),
    ),
    h("div", { className: "spacer" }),
    h("button", { type: "button", className: "primary", onClick: onExport, disabled: !canExport }, "Export CSV"),
  );
}

function ActivityChart({ s, hours, bucketMin }) {
  // Wide: one tick per hour (every 2 h for long windows) on the 15-min
  // columns. Hourly columns: let Recharts drop labels that would collide.
  const interval =
    bucketMin === 60 ? "preserveStartEnd" : (hours > 12 ? 2 : 1) * (60 / bucketMin) - 1;
  const maxPersons = Math.max(...s.rows.map((r) => r.persons));
  return h(
    "section",
    { className: "card" },
    h("h2", null, "Activity over the day"),
    h(
      "p",
      { className: "sub" },
      `People detected per ${bucketName(bucketMin)}. Shaded bands show when the app was open.`,
    ),
    h(
      "div",
      { className: "legend" },
      h("span", null, h("i", { className: "swatch bar" }), "People detected"),
      h("span", null, h("i", { className: "swatch band" }), "App open"),
    ),
    h(
      "div",
      { className: "chart-box" },
      h(
        ResponsiveContainer,
        { width: "100%", height: "100%" },
        h(
          BarChart,
          { data: s.rows, margin: { top: 8, right: 8, bottom: 0, left: -16 }, barCategoryGap: 2 },
          h(CartesianGrid, { stroke: COLORS.grid, vertical: false }),
          ...s.bands.map((b, i) =>
            h(ReferenceArea, {
              key: `band-${i}`,
              x1: b.x1,
              x2: b.x2,
              fill: COLORS.band,
              fillOpacity: 1,
              strokeOpacity: 0,
              ifOverflow: "extendDomain",
            }),
          ),
          h(XAxis, {
            dataKey: "label",
            interval,
            minTickGap: 8,
            tick: { fill: COLORS.axis, fontSize: 11 },
            tickLine: false,
            axisLine: { stroke: COLORS.grid },
          }),
          h(YAxis, {
            allowDecimals: false,
            // Nice rounded ticks; keep a 0..1 scale when nobody was seen.
            domain: maxPersons > 0 ? [0, "auto"] : [0, 1],
            tick: { fill: COLORS.axis, fontSize: 11 },
            tickLine: false,
            axisLine: false,
          }),
          h(Tooltip, { content: h(BarTip), cursor: { fill: COLORS.cursor } }),
          h(Bar, {
            dataKey: "persons",
            fill: COLORS.accent,
            maxBarSize: 24,
            radius: [4, 4, 0, 0],
            isAnimationActive: false,
          }),
        ),
      ),
    ),
  );
}

function PositionsChart({ s }) {
  return h(
    "section",
    { className: "card" },
    h("h2", null, "Where people were detected"),
    h(
      "p",
      { className: "sub" },
      `Feet position of each sighting within the camera frame (one sample per person every 5 s)` +
        (s.total > s.points.length ? ` — showing ${s.points.length.toLocaleString()} of ${s.total.toLocaleString()}.` : "."),
    ),
    h(
      "div",
      { className: "scatter-box" },
      h(
        ResponsiveContainer,
        { width: "100%", height: "100%" },
        h(
          ScatterChart,
          { margin: { top: 8, right: 8, bottom: 8, left: 8 } },
          h(XAxis, { type: "number", dataKey: "x", domain: [0, 1], hide: true }),
          h(YAxis, { type: "number", dataKey: "y", domain: [0, 1], reversed: true, hide: true }),
          h(Tooltip, { content: h(ScatterTip), cursor: false }),
          h(Scatter, { data: s.points, shape: h(Dot), isAnimationActive: false }),
        ),
      ),
    ),
  );
}

function SessionsTable({ s }) {
  return h(
    "section",
    { className: "card" },
    h("h2", null, "Sessions"),
    h("p", { className: "sub" }, "Each time the app was opened in this browser."),
    h(
      "div",
      { className: "table-wrap" },
      h(
        "table",
        null,
        h(
          "thead",
          null,
          h(
            "tr",
            null,
            h("th", null, "Opened"),
            h("th", null, "Closed"),
            h("th", { className: "num" }, "Duration"),
            h("th", { className: "num" }, "People"),
            h("th", { className: "num" }, "Samples"),
          ),
        ),
        h(
          "tbody",
          null,
          s.sessions.map((x) =>
            h(
              "tr",
              { key: x.id },
              h("td", null, fmtDateTime(x.start)),
              h("td", null, x.live ? "still open" : fmtDateTime(x.lastSeen)),
              h("td", { className: "num" }, fmtDuration(x.lastSeen - x.start)),
              h("td", { className: "num" }, x.people.toLocaleString()),
              h("td", { className: "num" }, x.samples.toLocaleString()),
            ),
          ),
        ),
      ),
    ),
  );
}

function App() {
  const [date, setDate] = useState(toDateInput(new Date()));
  const [fromHour, setFromHour] = useState(6);
  const [toHour, setToHour] = useState(18);
  const [data, setData] = useState(null);
  const [narrow, setNarrow] = useState(() => window.matchMedia(NARROW_QUERY).matches);
  const bucketMin = bucketMinutes(narrow);

  useEffect(() => {
    const mq = window.matchMedia(NARROW_QUERY);
    const onChange = () => setNarrow(mq.matches);
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);
  const [error, setError] = useState(null);

  const day = fromDateInput(date);
  const from = new Date(day.getFullYear(), day.getMonth(), day.getDate(), fromHour).getTime();
  const to = new Date(day.getFullYear(), day.getMonth(), day.getDate(), toHour).getTime();

  useEffect(() => {
    let cancelled = false;
    const load = () =>
      StatsDB.query(from, to)
        .then((d) => !cancelled && (setData(d), setError(null)))
        .catch((err) => !cancelled && setError(err.message || String(err)));
    load();
    // Keep a window that includes "now" live while the tracker runs.
    const timer = Date.now() >= from && Date.now() <= to ? setInterval(load, REFRESH_MS) : null;
    return () => {
      cancelled = true;
      if (timer) clearInterval(timer);
    };
  }, [from, to]);

  const s = useMemo(
    () => (data ? summarize(data, from, to, bucketMin) : null),
    [data, from, to, bucketMin],
  );
  const hasData = s && (s.sessions.length || s.total);

  return h(
    React.Fragment,
    null,
    h(Filters, {
      date,
      setDate,
      fromHour,
      setFromHour,
      toHour,
      setToHour,
      onExport: () => exportCsv(data, date),
      canExport: Boolean(data && data.sightings.length),
    }),
    error ? h("div", { className: "card empty" }, `Could not read statistics: ${error}`) : null,
    !s
      ? h("div", { className: "card empty" }, "Loading…")
      : h(
          React.Fragment,
          null,
          h(
            "div",
            { className: "tiles" },
            h(Tile, { label: "App open", value: fmtDuration(s.openMs), sub: `of ${toHour - fromHour} h window` }),
            h(Tile, { label: "Sessions", value: s.sessions.length.toLocaleString() }),
            h(Tile, { label: "People detected", value: s.people.toLocaleString(), sub: "unique tracked persons" }),
            h(Tile, {
              label: `Busiest ${bucketName(bucketMin)}`,
              value: s.busiest ? s.busiest.label : "—",
              sub: s.busiest ? `${s.busiest.persons} ${s.busiest.persons === 1 ? "person" : "people"}` : null,
            }),
          ),
          hasData
            ? h(
                React.Fragment,
                null,
                h(ActivityChart, { s, hours: toHour - fromHour, bucketMin }),
                h(PositionsChart, { s }),
                h(SessionsTable, { s }),
              )
            : h(
                "div",
                { className: "card empty" },
                "No activity recorded in this period.",
                h("br"),
                "Statistics are stored in this browser only — open this page on the device that runs the tracker.",
              ),
        ),
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(h(App));
