import React, { useRef, useState, useEffect, useCallback, DragEvent } from "react";

// Interface for data received from backend WebSocket
interface ChunkStatusData {
  abnormal: boolean;
  label: string; // e.g., "Normal Video", "Road Accident", "Error: ..."
  clipUrl?: string; // URL only if abnormal and conversion succeeded
  sourceChunkIndex: number; // 0-based index
}

const VideoSurveillanceUI = () => {
  // --- State Variables ---
  const [fileInfo, setFileInfo] = useState<{ name: string; type: string } | null>(null); // Store basic file info
  const [uploading, setUploading] = useState(false);
  const [preprocessing, setPreprocessing] = useState(false);
  const [videoUrl, setVideoUrl] = useState<string | null>(null); // Display URL
  const [isReadyToPlay, setIsReadyToPlay] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  // Store all chunk statuses received, not just abnormal ones for the sidebar
  const [chunkStatuses, setChunkStatuses] = useState<Record<number, ChunkStatusData>>({});
  const [totalChunks, setTotalChunks] = useState<number>(0);
  const [fps, setFps] = useState<number>(0);
  const [chunkDuration, setChunkDuration] = useState<number>(0);
  const [lastSignaledChunkIndex, setLastSignaledChunkIndex] = useState<number>(-1);
  const [error, setError] = useState<string | null>(null);
  const [isDraggingOver, setIsDraggingOver] = useState(false);

  // --- Refs ---
  const videoRef = useRef<HTMLVideoElement>(null);
  const websocketRef = useRef<WebSocket | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const sidebarRef = useRef<HTMLElement>(null); // Ref for the sidebar

  // --- Constants ---
  const FRAMES_PER_CHUNK = 96;
  const BACKEND_URL = "http://localhost:8000";

  // --- Effects ---

  // Calculate chunk duration
  useEffect(() => {
    if (fps > 0) {
      const duration = FRAMES_PER_CHUNK / fps;
      setChunkDuration(duration);
      console.log(`Calculated chunk duration: ${duration.toFixed(3)}s`);
    }
  }, [fps]);

  // WebSocket cleanup
  useEffect(() => {
    return () => {
      websocketRef.current?.close();
    };
  }, []);

  // --- Core Logic Functions ---

  // Centralized Upload Handler
  const uploadAndProcessFile = useCallback(async (selectedFile: File | null) => {
    if (!selectedFile) {
      setError("No file selected.");
      return;
    }
    if (!['video/mp4', 'video/avi', 'video/x-msvideo'].includes(selectedFile.type)) {
      setError("Invalid file type. Please upload MP4 or AVI.");
      return;
    }

    // Reset states for new upload
    setError(null);
    setFileInfo({ name: selectedFile.name, type: selectedFile.type });
    setUploading(true);
    setPreprocessing(true);
    setIsReadyToPlay(false);
    setVideoUrl(null);
    setChunkStatuses({}); // Clear previous statuses
    setLastSignaledChunkIndex(-1);
    setIsPlaying(false);
    setFps(0);
    setChunkDuration(0);
    setTotalChunks(0);
    websocketRef.current?.close(); // Close any existing WS connection


    const formData = new FormData();
    formData.append("file", selectedFile);
    console.log(`Starting upload for: ${selectedFile.name}`);

    try {
      const response = await fetch(`${BACKEND_URL}/api/upload`, { method: "POST", body: formData });
      setUploading(false); // Upload binary transfer complete
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || `Upload failed (HTTP ${response.status})`);

      console.log("Backend response:", result);
      if (result.displayUrl && result.totalChunks > 0 && result.fps > 0) {
        setVideoUrl(`${BACKEND_URL}${result.displayUrl}`);
        setTotalChunks(result.totalChunks);
        setFps(result.fps);
        setIsReadyToPlay(true);
        console.log(`Preprocessing successful. Chunks: ${result.totalChunks}, FPS: ${result.fps.toFixed(2)}`);
      } else {
        throw new Error(result.error || "Preprocessing failed: Invalid data received.");
      }
    } catch (err: any) {
      console.error("Upload or processing failed:", err);
      setError(`Operation failed: ${err.message}`);
      setIsReadyToPlay(false);
      setUploading(false);
    } finally {
      setPreprocessing(false); // Preprocessing phase finished
    }
  }, []); // Empty dependencies - Function is self-contained

  // WebSocket Connect
  const connectWebSocket = useCallback(() => {
    websocketRef.current?.close();
    const socket = new WebSocket(`ws://${window.location.hostname}:8000/ws`);
    websocketRef.current = socket;

    socket.onopen = () => console.log("WebSocket connected!");
    socket.onerror = (event) => {
      console.error("WebSocket error:", event);
      setError("WebSocket connection error. Ensure backend is running.");
      setIsPlaying(false);
      if(videoUrl) setIsReadyToPlay(true);
    };
    socket.onclose = (event) => {
      console.log("WebSocket connection closed.", event.code, event.reason);
      if (!event.wasClean && isPlaying) {
          setError("WebSocket connection lost during processing.");
          setIsPlaying(false);
          if(videoUrl) setIsReadyToPlay(true);
      }
    };
    socket.onmessage = (event) => {
       console.log("Raw WebSocket message:", event.data);
      try {
          const data: ChunkStatusData = JSON.parse(event.data);
           console.log("Parsed WebSocket data:", data);
          // Update the status for this specific chunk index
          setChunkStatuses(prev => ({
              ...prev,
              [data.sourceChunkIndex]: data
          }));
      } catch (error) {
           console.error("Failed to parse WebSocket message:", error, "Raw data:", event.data);
      }
    };
  }, [videoUrl, isPlaying]); // Added isPlaying dependency for onclose logic

  // Send Chunk Request
  const sendProcessChunkRequest = useCallback((chunkIndex: number) => {
    const ws = websocketRef.current;
    if (ws?.readyState === WebSocket.OPEN) {
      // console.log(`Signaling backend for chunk index: ${chunkIndex}`);
      ws.send(JSON.stringify({ type: "processChunk", chunkIndex }));
      setLastSignaledChunkIndex(chunkIndex);
    } else {
      const state = ws?.readyState;
      if (state === WebSocket.CONNECTING) {
        console.warn("WebSocket is still connecting. Chunk request deferred.");
      } else {
        console.error(`WebSocket not open (state: ${state}). Cannot send chunk request.`);
        setError("WebSocket disconnected. Cannot send processing request.");
        setIsPlaying(false);
         if(videoUrl) setIsReadyToPlay(true);
      }
    }
  }, [videoUrl]); // Include videoUrl dependency

  // Process Current Chunk based on video time
  const processCurrentChunk = useCallback((fromSeek = false) => {
    if (!videoRef.current || chunkDuration <= 0) return;
    if (!isPlaying && !fromSeek) return;

    const video = videoRef.current;
    const currentTime = video.currentTime;
    const currentChunkIndex = Math.max(0, Math.min(totalChunks - 1, Math.floor(currentTime / chunkDuration)));

    // console.log(`[${fromSeek ? 'Seek' : 'TimeUpdate'}] time: ${currentTime.toFixed(2)}, chunkIdx: ${currentChunkIndex}, lastSignaled: ${lastSignaledChunkIndex}`);

    if (currentChunkIndex >= 0 && currentChunkIndex < totalChunks && currentChunkIndex !== lastSignaledChunkIndex) {
      sendProcessChunkRequest(currentChunkIndex);
    }
  }, [chunkDuration, isPlaying, lastSignaledChunkIndex, totalChunks, sendProcessChunkRequest]);

  // Start Button Handler
  const handleStart = () => {
    if (!videoRef.current || !videoUrl) return;
    setError(null);
    setIsReadyToPlay(false);
    connectWebSocket(); // Connect WebSocket
    videoRef.current.play() // Attempt to play
        .then(() => {
            setIsPlaying(true);
            videoRef.current!.controls = true;
            // Let connectWebSocket's onopen trigger the first chunk check
        })
        .catch(err => {
            console.error("Video play failed:", err);
            setError(`Could not play video: ${err.message}.`);
            setIsPlaying(false);
            setIsReadyToPlay(true);
            websocketRef.current?.close();
        });
  };

   // Seek Handler
   const handleSeeked = useCallback(() => {
      if (!videoRef.current) return;
      console.log(`Seeked to: ${videoRef.current.currentTime.toFixed(2)}`);
      // Reset last signaled index to force send for the new position
      setLastSignaledChunkIndex(-1);
      processCurrentChunk(true); // Trigger check after seek
   }, [processCurrentChunk]);

  // --- Drag and Drop Handlers ---
  const handleDragOver = useCallback((event: DragEvent<HTMLDivElement>) => {
    event.preventDefault(); event.stopPropagation(); setIsDraggingOver(true);
  }, []);
  const handleDragEnter = useCallback((event: DragEvent<HTMLDivElement>) => {
    event.preventDefault(); event.stopPropagation(); setIsDraggingOver(true);
  }, []);
  const handleDragLeave = useCallback((event: DragEvent<HTMLDivElement>) => {
    event.preventDefault(); event.stopPropagation();
    const relatedTarget = event.relatedTarget as Node;
     if (!event.currentTarget.contains(relatedTarget)) setIsDraggingOver(false);
  }, []);
  const handleDrop = useCallback((event: DragEvent<HTMLDivElement>) => {
    event.preventDefault(); event.stopPropagation(); setIsDraggingOver(false);
    if (uploading || preprocessing) return;
    const files = event.dataTransfer.files;
    if (files && files.length > 0) {
      uploadAndProcessFile(files[0]);
      if (event.dataTransfer.items) event.dataTransfer.items.clear();
      else event.dataTransfer.clearData();
    }
  }, [uploadAndProcessFile, uploading, preprocessing]);
  const triggerFileInput = () => {
    if (uploading || preprocessing) return;
    fileInputRef.current?.click();
  };


  // --- JSX Rendering ---
  return (
    <div className="w-full max-w-7xl mx-auto bg-white rounded-xl shadow-2xl overflow-hidden my-8 border border-gray-200">
      {/* Header */}
      <div className="bg-gradient-to-r from-slate-700 to-slate-900 p-5">
        <h2 className="text-2xl sm:text-3xl font-bold text-white text-center tracking-tight">
          Intelligent Video Surveillance
        </h2>
      </div>

      <div className="flex flex-col lg:flex-row"> {/* Main Content Area */}

        {/* --- Left Column (Main Interaction Area) --- */}
        <div className="flex-grow lg:w-2/3 p-6 sm:p-8 border-b lg:border-b-0 lg:border-r border-gray-200">
          {/* Error Display */}
          {error && (
            <div className="mb-4 p-4 bg-red-50 border border-red-300 text-red-800 rounded-lg text-center shadow-sm">
              <span className="font-semibold">Error:</span> {error}
            </div>
          )}

          {/* 1. Upload Stage */}
          {!videoUrl && !preprocessing && (
            <div
              className={`border-2 border-dashed rounded-lg p-8 sm:p-16 text-center transition-colors duration-200 ease-in-out cursor-pointer min-h-[300px] flex flex-col justify-center items-center ${isDraggingOver ? 'border-indigo-500 bg-indigo-50' : 'border-gray-300 hover:border-gray-400 bg-gray-50'} ${uploading ? 'opacity-50 cursor-not-allowed' : ''}`}
              onDragEnter={handleDragEnter} onDragLeave={handleDragLeave} onDragOver={handleDragOver} onDrop={handleDrop}
              onClick={triggerFileInput} role="button" tabIndex={0} aria-label="Video upload drop zone"
            >
              <input ref={fileInputRef} id="video-upload" type="file" accept=".mp4,.avi,video/mp4,video/avi,video/x-msvideo" onChange={(e) => uploadAndProcessFile(e.target.files ? e.target.files[0] : null)} className="hidden" disabled={uploading || preprocessing} />
              <div className="space-y-3 text-gray-600">
                 <svg className={`w-16 h-16 mx-auto ${isDraggingOver ? 'text-indigo-600' : 'text-gray-400'}`} fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path></svg>
                <p className="text-lg font-medium">{isDraggingOver ? "Drop video file here!" : "Drag & drop video file or click"}</p>
                <p className="text-sm text-gray-500">MP4 or AVI formats supported</p>
              </div>
              {uploading && <p className="mt-4 text-blue-600 font-medium">Uploading...</p>}
            </div>
          )}

          {/* 2. Preprocessing Stage */}
          {preprocessing && (
            <div className="text-center py-16 text-gray-700 space-y-4 min-h-[300px] flex flex-col justify-center items-center">
               <svg className="animate-spin h-10 w-10 text-indigo-600 mx-auto" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
              <p className="text-xl font-semibold">Preprocessing Video...</p>
              <p className="text-base text-gray-500">Converting & splitting chunks. Please wait.</p>
            </div>
          )}

          {/* 3. Ready/Playing Stage */}
          {videoUrl && !preprocessing && (
            <div className="space-y-5">
              {/* Video Player */}
              <div>
                <h3 className="text-xl font-semibold mb-3 text-gray-800 capitalize">
                  {fileInfo?.name || 'Video Analysis'}
                </h3>
                <video
                  ref={videoRef}
                  src={videoUrl}
                  className="w-full rounded-lg bg-black shadow-md aspect-video"
                  controls={isPlaying} // Only show controls once playing started
                  onTimeUpdate={() => processCurrentChunk(false)}
                  onSeeked={handleSeeked} // Use specific handler
                  onPlay={() => setIsPlaying(true)}
                  onPause={() => setIsPlaying(false)}
                  onEnded={() => { setIsPlaying(false); websocketRef.current?.close(); }}
                  onError={(e) => { console.error("Main video playback error:", e); setError("Video playback error."); }}
                  preload="auto" // Help ensure video metadata is loaded
                />
              </div>

              {/* Start Button */}
              {isReadyToPlay && !isPlaying && (
                <div className="text-center">
                    <button
                        onClick={handleStart}
                        className="w-full sm:w-auto inline-flex justify-center items-center px-8 py-3 border border-transparent text-base font-medium rounded-md shadow-sm text-white bg-gradient-to-r from-blue-600 to-indigo-700 hover:from-blue-700 hover:to-indigo-800 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transition duration-150 ease-in-out disabled:opacity-50"
                        disabled={!videoUrl || uploading || preprocessing}
                    >
                        <svg className="-ml-1 mr-3 h-5 w-5" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true"><path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z" clipRule="evenodd" /></svg>
                        Start Analysis
                    </button>
                </div>
              )}
            </div>
          )}
        </div> {/* End Left Column */}


        {/* --- Right Column (Sidebar / Events Panel) --- */}
        <aside ref={sidebarRef} className="lg:w-1/3 p-6 bg-slate-50 border-l border-gray-200 max-h-[calc(100vh-150px)] lg:max-h-none overflow-y-auto custom-scrollbar">
           <h3 className="text-xl font-semibold mb-4 text-slate-700 border-b border-slate-300 pb-3 sticky top-0 bg-slate-50 z-10">
             Analysis Log
           </h3>
           <div className="space-y-3">
             {totalChunks > 0 ? (
                // Display status for each chunk as it comes in, with newest at the top
                 Object.values(chunkStatuses) // Get array of status objects
                   .sort((a, b) => b.sourceChunkIndex - a.sourceChunkIndex) // Sort in reverse order (newest first)
                   .map((status) => (
                     <div
                        key={status.sourceChunkIndex}
                        className={`border rounded-md p-3 shadow-sm transition hover:shadow ${
                          status.abnormal
                             ? 'bg-red-50 border-red-200'
                             : status.label.startsWith("Error:")
                             ? 'bg-yellow-50 border-yellow-200'
                             : 'bg-white border-gray-200'
                        }`}
                      >
                         <p className={`text-sm font-medium flex justify-between items-center ${
                              status.abnormal ? 'text-red-700' : status.label.startsWith("Error:") ? 'text-yellow-700' : 'text-gray-700'
                            }`}>
                             <span>Chunk {status.sourceChunkIndex + 1}</span> {/* Display 1-based */}
                             <span className={`px-2 py-0.5 rounded-full text-xs font-semibold ${
                                status.abnormal ? 'bg-red-200 text-red-800' : status.label.startsWith("Error:") ? 'bg-yellow-200 text-yellow-800' : 'bg-green-100 text-green-800'
                             }`}>
                               {status.label}
                             </span>
                         </p>
                         {/* Only show video player here if abnormal and clip exists */}
                         {status.abnormal && status.clipUrl && (
                            <div className="mt-2 pt-2 border-t border-red-200">
                             <video
                                src={`${BACKEND_URL}${status.clipUrl}`}
                                className="w-full rounded border border-red-200"
                                controls
                                preload="metadata" // Load only metadata initially
                                onError={(e) => console.error('Error loading alert clip:', status.clipUrl, e)}
                             />
                           </div>
                         )}
                     </div>
                   ))
             ) : (
                 <p className="text-sm text-gray-500 italic text-center py-10">
                   {videoUrl ? 'Analysis not started yet.' : 'Upload a video to begin.'}
                 </p>
             )}
              {/* Add a placeholder at the bottom to ensure scroll works */}
             <div className="h-1"></div>
           </div>
        </aside> {/* End Right Column */}

      </div> {/* End Main Content Area flex */}
    </div> // End Outer container
  );
};

export default VideoSurveillanceUI;