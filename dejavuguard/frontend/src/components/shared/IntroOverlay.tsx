import { useEffect, useRef, useState } from "react";

const INTRO_SEEN_KEY = "dejavuguard_intro_seen";

export default function IntroOverlay() {
  const [show, setShow] = useState(() => {
    return !sessionStorage.getItem(INTRO_SEEN_KEY);
  });
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    if (show && videoRef.current) {
      videoRef.current.play().catch(() => {
        // Autoplay blocked — dismiss immediately
        setShow(false);
        sessionStorage.setItem(INTRO_SEEN_KEY, "1");
      });
    }
  }, [show]);

  const handleEnded = () => {
    setShow(false);
    sessionStorage.setItem(INTRO_SEEN_KEY, "1");
  };

  const handleClick = () => {
    setShow(false);
    sessionStorage.setItem(INTRO_SEEN_KEY, "1");
  };

  if (!show) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black cursor-pointer"
      onClick={handleClick}
      data-testid="intro-overlay"
    >
      <video
        ref={videoRef}
        src="/assets/intro_logo.mp4"
        onEnded={handleEnded}
        playsInline
        className="max-h-[80vh] max-w-[80vw]"
      />
      <p className="absolute bottom-8 text-terminal-dim text-xs font-mono animate-pulse">
        Click anywhere to skip
      </p>
    </div>
  );
}
