document.getElementById('listenBtn').addEventListener('click', async () => {
    const btn = document.getElementById('listenBtn');
    const resultDiv = document.getElementById('result');
    const [tab] = await browser.tabs.query({ active: true, currentWindow: true });

    btn.disabled = true;
    btn.innerText = "Grabando página...";

    try {
        // Ejecutamos la grabación dentro de la página web
        const results = await browser.scripting.executeScript({
            target: { tabId: tab.id },
            func: recordTabAudio,
        });

        const base64Audio = results[0].result;
        if (!base64Audio) throw new Error("No se pudo capturar audio");

        btn.innerText = "Identificando...";

        // Convertir Base64 a Blob para enviar al servidor
        const audioBlob = await fetch(base64Audio).then(r => r.blob());
        const formData = new FormData();
        formData.append('file', audioBlob, 'audio.webm');

        const response = await fetch('http://localhost:8000/recognize', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        
        if (data.success) {
            // Detectar si es resultado de YouTube o Shazam
            if (data.youtube_results && data.youtube_results.length > 0) {
                // Mostrar resultados de YouTube con características musicales
                let html = `<div style="text-align: left; max-height: 400px; overflow-y: auto;">`;
                
                // Mostrar características musicales si están disponibles
                if (data.musical_features) {
                    const features = data.musical_features;
                    html += `<div style="background: #e8f4f8; padding: 8px; border-radius: 5px; margin-bottom: 10px;">`;
                    html += `<b>🎵 Análisis Musical:</b><br>`;
                    html += `<small>`;
                    html += `• Tempo: <b>${features.tempo_bpm.toFixed(0)} BPM</b><br>`;
                    if (features.estimated_key) {
                        html += `• Tonalidad: <b>${features.estimated_key}</b><br>`;
                    }
                    if (features.genre_hints && features.genre_hints.length > 0) {
                        html += `• Características: ${features.genre_hints.join(', ')}<br>`;
                    }
                    html += `</small></div>`;
                }
                
                // Mostrar transcripción si existe
                if (data.transcription) {
                    html += `<div style="background: #f0f0f0; padding: 8px; border-radius: 5px; margin-bottom: 10px;">`;
                    html += `<b>📝 Letras detectadas</b>`;
                    if (data.language) {
                        html += ` <small>(${data.language})</small>`;
                    }
                    html += `:<br>`;
                    html += `<small style="color: #666;">"${data.transcription.substring(0, 100)}${data.transcription.length > 100 ? '...' : ''}"</small>`;
                    html += `</div>`;
                } else if (data.search_query) {
                    html += `<div style="background: #fff3cd; padding: 8px; border-radius: 5px; margin-bottom: 10px;">`;
                    html += `<b>⚠️ No se detectaron letras</b><br>`;
                    html += `<small>Búsqueda por características musicales: "${data.search_query}"</small>`;
                    html += `</div>`;
                }
                
                html += `<b>Posibles canciones:</b><br>`;
                
                data.youtube_results.forEach((video, i) => {
                    html += `
                        <div style="margin: 10px 0; padding: 8px; background: #f5f5f5; border-radius: 5px;">
                            <a href="${video.url}" target="_blank" style="text-decoration: none; color: #0078D7;">
                                <img src="${video.thumbnail}" style="width: 100%; border-radius: 4px; margin-bottom: 5px;">
                                <b>${i + 1}. ${video.title}</b><br>
                                <small>${video.artist}</small>
                            </a>
                        </div>
                    `;
                });
                
                html += `</div>`;
                resultDiv.innerHTML = html;
            } else {
                // Resultado de Shazam (formato original)
                let html = `<b>${data.title}</b><br>${data.subtitle}`;
                if (data.method) {
                    html += `<br><small style="color: #666;">Método: ${data.method}</small>`;
                }
                if (data.coverart) {
                    html += `<br><img src="${data.coverart}" alt="Cover">`;
                }
                resultDiv.innerHTML = html;
            }
        } else {
            resultDiv.innerHTML = "No se encontró la canción.";
        }

    } catch (error) {
        console.error(error);
        resultDiv.innerHTML = "Error: Asegúrate de que la música esté sonando y refresca la página.";
    } finally {
        btn.disabled = false;
        btn.innerText = "Escuchar (8s)";
    }
});

// Esta función se ejecuta DENTRO de la página (YouTube, etc.)
async function recordTabAudio() {
    return new Promise((resolve) => {
        const video = document.querySelector('video') || document.querySelector('audio');
        if (!video) return resolve(null);

        // Capturar el flujo del elemento multimedia
        const stream = video.captureStream ? video.captureStream() : video.mozCaptureStream();
        const recorder = new MediaRecorder(stream);
        const chunks = [];

        recorder.ondataavailable = e => chunks.push(e.data);
        recorder.onstop = async () => {
            const blob = new Blob(chunks, { type: 'audio/webm' });
            const reader = new FileReader();
            reader.onloadend = () => resolve(reader.result); // Devuelve Base64
            reader.readAsDataURL(blob);
        };

        recorder.start();
        setTimeout(() => recorder.stop(), 8000); // Aumentado a 8 segundos
    });
}