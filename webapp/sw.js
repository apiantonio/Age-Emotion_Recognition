const CACHE_NAME = 'facesight-cache-v18';

const ASSETS_TO_CACHE = [
    './',
    './index.html',
    './script.js',
    './manifest.json',
    './icon-192.png',
    './icon-512.png',
    // './emotion.onnx',
    // './age.onnx'
];

self.addEventListener('install', (event) => {
    self.skipWaiting();
    console.log('[Service Worker] Installazione in corso...');
    event.waitUntil(
        caches.open(CACHE_NAME).then((cache) => {
            console.log('[Service Worker] Salvataggio file locali nella cache...');
            return cache.addAll(ASSETS_TO_CACHE);
        })
    );
});

self.addEventListener('activate', (event) => {
    event.waitUntil(
        caches.keys().then((keyList) => {
            return Promise.all(keyList.map((key) => {
                if (key !== CACHE_NAME) {
                    console.log('[Service Worker] Rimozione vecchia cache', key);
                    return caches.delete(key);
                }
            }));
        })
    );
});

self.addEventListener('fetch', (event) => {
    // IGNORA le richieste delle estensioni di Chrome per evitare crash
    if (!event.request.url.startsWith('http')) {
        return;
    }

    event.respondWith(
        caches.match(event.request).then((cachedResponse) => {
            if (cachedResponse) {
                return cachedResponse;
            }
      
            return fetch(event.request).then((networkResponse) => {
                return caches.open(CACHE_NAME).then((cache) => {
                    cache.put(event.request, networkResponse.clone());
                    return networkResponse;
                });
            });
        })
    );
});