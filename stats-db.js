/* stats-db.js — local usage statistics store (IndexedDB)
 * Shared by the tracker (writes) and /admin (reads). Everything stays in
 * this browser's IndexedDB for this origin: no server, no network.
 *
 *   sessions   { id, start, lastSeen }            one row per page open
 *              lastSeen is bumped by a heartbeat, so it doubles as the
 *              "closed at" time even when the tab is killed without warning.
 *   sightings  { seq, t, session, track, cls, x, y, score, view }
 *              throttled samples of confirmed person tracks. x / y are the
 *              person's feet (box bottom-centre) normalised 0..1 to the raw
 *              camera frame — in 360 modes they're mapped back through the
 *              projection, so positions line up across modes and view angles.
 */

const StatsDB = (() => {
  const DB_NAME = "safety-stats";
  const VERSION = 1;
  let dbPromise = null;

  function open() {
    if (!dbPromise) {
      dbPromise = new Promise((resolve, reject) => {
        const req = indexedDB.open(DB_NAME, VERSION);
        req.onupgradeneeded = () => {
          const db = req.result;
          const sessions = db.createObjectStore("sessions", { keyPath: "id" });
          sessions.createIndex("lastSeen", "lastSeen");
          const sightings = db.createObjectStore("sightings", {
            keyPath: "seq",
            autoIncrement: true,
          });
          sightings.createIndex("t", "t");
        };
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error);
      });
    }
    return dbPromise;
  }

  // Run `fn(stores)` in one transaction; resolves with fn's result once the
  // transaction commits.
  async function withTx(names, mode, fn) {
    const db = await open();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(names, mode);
      const stores = Object.fromEntries(names.map((n) => [n, tx.objectStore(n)]));
      let result;
      Promise.resolve(fn(stores)).then((r) => (result = r), reject);
      tx.oncomplete = () => resolve(result);
      tx.onerror = () => reject(tx.error);
      tx.onabort = () => reject(tx.error);
    });
  }

  function getAll(source, range) {
    return new Promise((resolve, reject) => {
      const req = source.getAll(range);
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
  }

  function deleteRange(index, range) {
    const req = index.openCursor(range);
    req.onsuccess = () => {
      const cur = req.result;
      if (!cur) return;
      cur.delete();
      cur.continue();
    };
  }

  return {
    putSession(session) {
      return withTx(["sessions"], "readwrite", (s) => {
        s.sessions.put(session);
      });
    },

    addSightings(rows) {
      if (!rows.length) return Promise.resolve();
      return withTx(["sightings"], "readwrite", (s) => {
        for (const row of rows) s.sightings.add(row);
      });
    },

    // Sessions overlapping [from, to] and sightings inside it (ms epochs).
    query(from, to) {
      return withTx(["sessions", "sightings"], "readonly", async (s) => {
        const [sessions, sightings] = await Promise.all([
          getAll(s.sessions.index("lastSeen"), IDBKeyRange.lowerBound(from)),
          getAll(s.sightings.index("t"), IDBKeyRange.bound(from, to)),
        ]);
        return {
          sessions: sessions.filter((x) => x.start <= to),
          sightings,
        };
      });
    },

    // Drop everything older than `before` (ms epoch).
    prune(before) {
      return withTx(["sessions", "sightings"], "readwrite", (s) => {
        deleteRange(s.sessions.index("lastSeen"), IDBKeyRange.upperBound(before, true));
        deleteRange(s.sightings.index("t"), IDBKeyRange.upperBound(before, true));
      });
    },
  };
})();
