import React, { useRef, useState, useEffect, useCallback } from "react";


interface AlertData {
  abnormal: boolean;
  label: string;
  clipUrl?: string;
  sourceChunkIndex: number;
}

const VideoSurveillanceUI = () => {
  const [uploading, setUploading] = useState(false);
  const [preprocessing, setPreprocessing] = useState(false);
  const [videoUrl, setVideoUrl] = useState<string | null>(null); // URL for the display video
  const [isReadyToPlay, setIsReadyToPlay] = useState(false); // Controls Start button visibility
  const [isPlaying, setIsPlaying] = useState(false); // Tracks if video is playing
  const [alerts, setAlerts] = useState<AlertData[]>([]);
  const [totalChunks, setTotalChunks] = useState<number>(0);
  const [fps, setFps] = useState<number>(0);
  const [chunkDuration, setChunkDuration] = useState<number>(0);
  const [lastSignaledChunkIndex, setLastSignaledChunkIndex] = useState<number>(-1);
  const [error, setError] = useState<string | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const websocketRef = useRef<WebSocket | null>(null);

  // Calculate chunk duration once FPS is known
  useEffect(() => {
    if (fps > 0) {
      const duration = FRAMES_PER_CHUNK / fps;
      setChunkDuration(duration);
      console.log(`Calculated chunk duration: ${duration} seconds for ${FRAMES_PER_CHUNK} frames at ${fps} FPS.`);
    }
  }, [fps]);

  // Cleanup WebSocket on component unmount
  useEffect(() => {
    return () => {
      websocketRef.current?.close();
    };
  }, []);

  const handleUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setError(null); // Clear previous errors
    setUploading(true);
    setPreprocessing(true);
    setIsReadyToPlay(false); // Hide start button during upload/processing
    setVideoUrl(null); // Clear previous video
    setAlerts([]); // Clear previous alerts
    setLastSignaledChunkIndex(-1);


    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://localhost:8000/api/upload", {
        method: "POST",
        body: formData,
      });

      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.error || `HTTP error! status: ${response.status}`);
      }

      if (result.displayUrl && result.totalChunks > 0 && result.fps > 0) {
        // Prepend server origin if necessary (it's already included by backend mount)
        const fullDisplayUrl = `http://localhost:8000${result.displayUrl}`;
        console.log("Received display URL:", fullDisplayUrl);
        setVideoUrl(fullDisplayUrl);
        setTotalChunks(result.totalChunks);
        setFps(result.fps); // This triggers the useEffect to calculate chunkDuration
        setIsReadyToPlay(true); // Show the start button now
         console.log(`Upload successful. Total Chunks: ${result.totalChunks}, FPS: ${result.fps}`);
      } else {
           throw new Error(result.error || "Upload completed but received invalid data from backend.");
      }
    } catch (err: any) {
      console.error("Upload or processing failed:", err);
      setError(`Upload failed: ${err.message}`);
      setIsReadyToPlay(false); // Ensure start button remains hidden on error
    } finally {
      setUploading(false);
      setPreprocessing(false);
    }
  };

 const connectWebSocket = () => {
    // Close existing connection if any before opening a new one
    websocketRef.current?.close();

    const socket = new WebSocket("ws://localhost:8000/ws");
    websocketRef.current = socket;

    socket.onopen = () => {
      console.log("WebSocket connected!");
      // Optionally send initial state or trigger first chunk processing if needed immediately
      // processCurrentChunk(); // Or let onTimeUpdate handle the first chunk
    };

    socket.onmessage = (event) => {
      try {
          const data: AlertData = JSON.parse(event.data);
          console.log("WebSocket message received:", data);

           // Only add abnormal alerts to the list for display
           if (data.abnormal) {
              // Avoid adding duplicate alerts for the same chunk if backend sends multiple
               setAlerts((prev) => {
                   if (!prev.some(a => a.sourceChunkIndex === data.sourceChunkIndex)) {
                       return [...prev, data];
                   }
                   return prev;
               });
           } else {
                // Handle normal/error messages if needed (e.g., update status)
                console.log(`Chunk ${data.sourceChunkIndex} processed as: ${data.label}`);
           }

      } catch (error) {
           console.error("Failed to parse WebSocket message:", error);
      }

    };

    socket.onerror = (event) => {
        console.error("WebSocket error:", event);
         setError("WebSocket connection error.");
    };


    socket.onclose = (event) => {
      console.log("WebSocket connection closed.", event.reason);
      // Optionally attempt to reconnect or notify user
      setIsPlaying(false); // Stop state if connection drops
    };
 };


  const handleStart = () => {
    if (!videoRef.current || !videoUrl) return;

    setIsReadyToPlay(false); // Hide start button once started
    setIsPlaying(true);
    videoRef.current.controls = true; // Show native controls once playing
    videoRef.current.play().catch(err => {
        console.error("Video play failed:", err);
        setError(`Could not play video: ${err.message}`);
        setIsPlaying(false); // Reset state if play fails
        setIsReadyToPlay(true); // Show start button again
    });
    connectWebSocket();
  };

  const sendProcessChunkRequest = useCallback((chunkIndex: number) => {
    if (websocketRef.current?.readyState === WebSocket.OPEN) {
       console.log(`Signaling backend to process chunk index: ${chunkIndex}`);
       websocketRef.current.send(JSON.stringify({ type: "processChunk", chunkIndex: chunkIndex }));
       setLastSignaledChunkIndex(chunkIndex);
    } else {
       console.warn("WebSocket not open. Cannot send chunk request.");
       // Optional: Attempt reconnect or queue request
    }
 }, []); // No dependencies needed if it only uses refs and state setters


 const processCurrentChunk = useCallback(() => {
    if (!videoRef.current || chunkDuration <= 0 || !isPlaying) {
      return; // Don't process if video not ready/playing or duration unknown
    }

    const currentTime = videoRef.current.currentTime;
    // Ensure index calculation is safe for t=0 and doesn't exceed total chunks
    const currentChunkIndex = Math.max(0, Math.min(totalChunks - 1, Math.floor(currentTime / chunkDuration)));

    // Send only if the index has changed and is valid
    if (currentChunkIndex >= 0 && currentChunkIndex < totalChunks && currentChunkIndex !== lastSignaledChunkIndex) {
        sendProcessChunkRequest(currentChunkIndex);
    }
  }, [chunkDuration, isPlaying, lastSignaledChunkIndex, totalChunks, sendProcessChunkRequest]); // Add sendProcessChunkRequest


  // UseEffect for interval-based check as ontimeupdate can be inconsistent
//   useEffect(() => {
//      let intervalId: NodeJS.Timeout | null = null;
//      if (isPlaying) {
//          // Check more frequently than chunk duration to catch changes promptly
//          intervalId = setInterval(processCurrentChunk, 500); // Check every 500ms
//      } else {
//          if (intervalId) clearInterval(intervalId);
//      }
//      return () => {
//          if (intervalId) clearInterval(intervalId);
//      };
//   }, [isPlaying, processCurrentChunk]);

  // --- Constants ---
  const FRAMES_PER_CHUNK = 96; // Keep consistent with backend if needed


  return (
    <div className="p-4 border rounded-lg w-full max-w-2xl mx-auto bg-white shadow-lg">
      <h2 className="text-2xl font-bold mb-6 text-center text-gray-800">Intelligent Video Surveillance</h2>

      {error && (
          <div className="my-4 p-3 bg-red-100 border border-red-400 text-red-700 rounded text-center">
              <strong>Error:</strong> {error}
          </div>
      )}

      {!videoUrl && !uploading && (
        <div className="text-center">
            <label htmlFor="video-upload" className="cursor-pointer bg-indigo-600 text-white py-2 px-5 rounded-md hover:bg-indigo-700 transition font-medium">
                Upload Video (.mp4 or .avi)
            </label>
          <input
            id="video-upload"
            type="file"
            accept=".mp4,.avi"
            onChange={handleUpload}
            className="hidden" // Hide default input, use label styling
            disabled={uploading || preprocessing}
          />
        </div>
      )}

       {(uploading || preprocessing) && (
         <div className="text-center my-4 text-gray-600">
             <p>{uploading && !preprocessing ? "Uploading..." : "Preprocessing video (converting/chunking)... Please wait."}</p>
             {/* Optional: Add a spinner */}
         </div>
       )}


      {videoUrl && (
        <div className="mb-4">
          <video
            ref={videoRef}
            src={videoUrl}
            className="w-full rounded-md bg-black" // Added bg-black for letterboxing
            controls={isPlaying} // Show controls only when playing has started
            onTimeUpdate={processCurrentChunk} // Trigger processing check on time update
            onSeeked={processCurrentChunk} // Trigger processing check immediately after seek
            // Consider adding onPlay, onPause to manage isPlaying state more accurately
            onPlay={() => setIsPlaying(true)}
            onPause={() => setIsPlaying(false)}
            onEnded={() => {setIsPlaying(false); websocketRef.current?.close();}} // Close WS on end
          />
        </div>
      )}

      {isReadyToPlay && !isPlaying && (
        <button
          onClick={handleStart}
          disabled={!videoUrl} // Should always be true if button is visible
          className="w-full bg-blue-600 text-white py-2 px-4 rounded hover:bg-blue-700 transition disabled:opacity-50"
        >
          Start Analysis
        </button>
      )}

      {alerts.length > 0 && (
        <div className="mt-6 pt-4 border-t">
          <h3 className="text-lg font-semibold mb-3 text-gray-700">Detected Abnormal Events:</h3>
          <div className="space-y-3 max-h-60 overflow-y-auto pr-2"> {/* Scrollable alerts */}
            {alerts.map((alert, idx) => (
              <div key={idx} className="bg-red-100 p-3 rounded-md border border-red-300 text-sm shadow-sm">
                <p className="font-semibold text-red-800">
                  Chunk {alert.sourceChunkIndex}: {alert.label}
                </p>
                {alert.clipUrl && (
                  <div className="mt-2">
                    <video
                     // Prepend origin for static files as well
                      src={`http://localhost:8000${alert.clipUrl}`}
                      className="w-full rounded"
                      controls
                      autoPlay={false} // Don't autoplay alert clips by default maybe?
                      loop={false}
                    />
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default VideoSurveillanceUI;