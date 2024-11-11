async function startAttentionTracking() {
    await faceapi.nets.tinyFaceDetector.loadFromUri('/models');
    await faceapi.nets.faceLandmark68Net.loadFromUri('/models');
    
    const video = document.getElementById('monitorVideo');
    const notification = document.getElementById('notification');
    
    video.addEventListener('play', () => {
        setInterval(async () => {
            const detections = await faceapi.detectSingleFace(video).withFaceLandmarks();
            if (detections) {
                const attentionScore = calculateAttentionScore(detections.landmarks);
                if (attentionScore < threshold) {
                    notification.classList.remove('hidden');
                } else {
                    notification.classList.add('hidden');
                }
            }
        }, 1000);
    });
}

function calculateAttentionScore(landmarks) {
    // Implement attention score logic based on head pose
    return Math.random() * 100;  // Placeholder for real attention score
}

startAttentionTracking();
