const CACHE_NAME = "safety-wb-v4";
const APP_SHELL = [
  "./",
  "./index.html",
  "./style.css",
  "./app.js",
  "./tracker.js",
  "./ort/ort.min.js",
  "./ort/ort.js",
  "./ort/ort.wasm.js",
  "./ort/ort.wasm.mjs",
  "./ort/ort-wasm-simd-threaded.mjs",
  "./ort/ort-wasm-simd-threaded.wasm",
  "./ort/ort-wasm-simd-threaded.jsep.mjs",
  "./ort/ort-wasm-simd-threaded.jsep.wasm",
  "./manifest.webmanifest",
  "./icon.svg",
  "./yolo26n.onnx",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches
      .open(CACHE_NAME)
      .then((cache) => cache.addAll(APP_SHELL))
      .then(() => self.skipWaiting()),
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(
          keys
            .filter((key) => key !== CACHE_NAME)
            .map((key) => caches.delete(key)),
        ),
      )
      .then(() => self.clients.claim()),
  );
});

self.addEventListener("fetch", (event) => {
  const { request } = event;
  if (request.method !== "GET") return;

  if (request.mode === "navigate") {
    event.respondWith(
      fetch(request)
        .then((response) => {
          const copy = response.clone();
          caches.open(CACHE_NAME).then((cache) => cache.put(request, copy));
          return response;
        })
        .catch(() => caches.match("./index.html")),
    );
    return;
  }

  event.respondWith(
    caches.match(request).then((cached) => {
      if (cached) return cached;
      return fetch(request)
        .then((response) => {
          const copy = response.clone();
          if (request.url.startsWith(self.location.origin)) {
            caches.open(CACHE_NAME).then((cache) => cache.put(request, copy));
          }
          return response;
        })
        .catch(() => caches.match("./index.html"));
    }),
  );
});
