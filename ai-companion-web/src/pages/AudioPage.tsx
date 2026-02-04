import { useState, useCallback, useRef } from 'react';
import {
  Mic,
  Volume2,
  Play,
  Pause,
  Download,
  Trash2,
  Loader2,
  Settings2,
  ChevronDown,
  ChevronUp,
  Square,
} from 'lucide-react';

const VOICE_MODELS = [
  { id: 'edge-tts', name: 'Edge TTS', provider: 'Microsoft' },
  { id: 'openai-tts', name: 'OpenAI TTS', provider: 'OpenAI' },
  { id: 'elevenlabs', name: 'ElevenLabs', provider: 'ElevenLabs' },
  { id: 'local-tts', name: 'Local TTS', provider: 'Local' },
];

const VOICES = {
  'edge-tts': [
    { id: 'ko-KR-SunHiNeural', name: 'SunHi (Korean Female)' },
    { id: 'ko-KR-InJoonNeural', name: 'InJoon (Korean Male)' },
    { id: 'en-US-JennyNeural', name: 'Jenny (English Female)' },
    { id: 'en-US-GuyNeural', name: 'Guy (English Male)' },
    { id: 'ja-JP-NanamiNeural', name: 'Nanami (Japanese Female)' },
    { id: 'zh-CN-XiaoxiaoNeural', name: 'Xiaoxiao (Chinese Female)' },
  ],
  'openai-tts': [
    { id: 'alloy', name: 'Alloy' },
    { id: 'echo', name: 'Echo' },
    { id: 'fable', name: 'Fable' },
    { id: 'onyx', name: 'Onyx' },
    { id: 'nova', name: 'Nova' },
    { id: 'shimmer', name: 'Shimmer' },
  ],
  'elevenlabs': [
    { id: 'rachel', name: 'Rachel' },
    { id: 'clyde', name: 'Clyde' },
    { id: 'domi', name: 'Domi' },
    { id: 'bella', name: 'Bella' },
  ],
  'local-tts': [
    { id: 'default', name: 'Default Voice' },
  ],
};

interface AudioParams {
  speed: number;
  pitch: number;
  volume: number;
}

interface GeneratedAudio {
  id: string;
  text: string;
  voice: string;
  model: string;
  url: string;
  duration: number;
  timestamp: string;
}

const DEFAULT_PARAMS: AudioParams = {
  speed: 1.0,
  pitch: 1.0,
  volume: 1.0,
};

export function AudioPage() {
  const [text, setText] = useState('');
  const [selectedModel, setSelectedModel] = useState('edge-tts');
  const [selectedVoice, setSelectedVoice] = useState('ko-KR-SunHiNeural');
  const [params, setParams] = useState<AudioParams>(DEFAULT_PARAMS);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedAudios, setGeneratedAudios] = useState<GeneratedAudio[]>([]);
  const [playingId, setPlayingId] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  const handleModelChange = useCallback((modelId: string) => {
    setSelectedModel(modelId);
    const voices = VOICES[modelId as keyof typeof VOICES];
    if (voices && voices.length > 0) {
      setSelectedVoice(voices[0].id);
    }
  }, []);

  const handleParamChange = useCallback(
    <K extends keyof AudioParams>(key: K, value: AudioParams[K]) => {
      setParams((prev) => ({ ...prev, [key]: value }));
    },
    []
  );

  const handleGenerate = useCallback(async () => {
    if (!text.trim()) return;

    setIsGenerating(true);

    try {
      // TODO: Replace with actual API call
      await new Promise((resolve) => setTimeout(resolve, 1500));

      // Simulated response
      const newAudio: GeneratedAudio = {
        id: `audio_${Date.now()}`,
        text: text.slice(0, 100),
        voice: selectedVoice,
        model: selectedModel,
        url: 'https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3',
        duration: Math.random() * 10 + 2,
        timestamp: new Date().toISOString(),
      };

      setGeneratedAudios((prev) => [newAudio, ...prev]);
    } catch (error) {
      console.error('Failed to generate audio:', error);
    } finally {
      setIsGenerating(false);
    }
  }, [text, selectedModel, selectedVoice]);

  const handlePlay = useCallback((audio: GeneratedAudio) => {
    if (playingId === audio.id) {
      audioRef.current?.pause();
      setPlayingId(null);
    } else {
      if (audioRef.current) {
        audioRef.current.src = audio.url;
        audioRef.current.play();
        setPlayingId(audio.id);
      }
    }
  }, [playingId]);

  const handleStop = useCallback(() => {
    audioRef.current?.pause();
    if (audioRef.current) {
      audioRef.current.currentTime = 0;
    }
    setPlayingId(null);
  }, []);

  const handleDelete = useCallback((id: string) => {
    if (playingId === id) {
      handleStop();
    }
    setGeneratedAudios((prev) => prev.filter((a) => a.id !== id));
  }, [playingId, handleStop]);

  const handleDownload = useCallback((audio: GeneratedAudio) => {
    const link = document.createElement('a');
    link.href = audio.url;
    link.download = `tts_${audio.id}.mp3`;
    link.click();
  }, []);

  const availableVoices = VOICES[selectedModel as keyof typeof VOICES] || [];

  return (
    <div className="h-full flex bg-bg-primary">
      {/* Hidden Audio Element */}
      <audio
        ref={audioRef}
        onEnded={() => setPlayingId(null)}
        className="hidden"
      />

      {/* Left Panel - Parameters */}
      <div className="w-80 flex-shrink-0 border-r border-border-primary overflow-y-auto">
        <div className="p-4 space-y-4">
          {/* Model Selection */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-text-secondary">TTS Model</label>
            <select
              value={selectedModel}
              onChange={(e) => handleModelChange(e.target.value)}
              className="w-full px-3 py-2 bg-bg-secondary border border-border-primary rounded-lg text-text-primary focus:outline-none focus:border-accent-primary"
            >
              {VOICE_MODELS.map((model) => (
                <option key={model.id} value={model.id}>
                  {model.name} ({model.provider})
                </option>
              ))}
            </select>
          </div>

          {/* Voice Selection */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-text-secondary">Voice</label>
            <select
              value={selectedVoice}
              onChange={(e) => setSelectedVoice(e.target.value)}
              className="w-full px-3 py-2 bg-bg-secondary border border-border-primary rounded-lg text-text-primary focus:outline-none focus:border-accent-primary"
            >
              {availableVoices.map((voice) => (
                <option key={voice.id} value={voice.id}>
                  {voice.name}
                </option>
              ))}
            </select>
          </div>

          {/* Text Input */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-text-secondary">Text to Speech</label>
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Enter text to convert to speech..."
              className="w-full px-3 py-2 bg-bg-secondary border border-border-primary rounded-lg text-text-primary placeholder-text-muted focus:outline-none focus:border-accent-primary resize-none"
              rows={6}
            />
            <div className="text-xs text-text-muted text-right">
              {text.length} characters
            </div>
          </div>

          {/* Basic Parameters */}
          <div className="space-y-3">
            {/* Speed */}
            <div className="space-y-1">
              <div className="flex justify-between">
                <label className="text-sm text-text-secondary">Speed</label>
                <span className="text-sm text-text-muted">{params.speed.toFixed(1)}x</span>
              </div>
              <input
                type="range"
                min={0.5}
                max={2.0}
                step={0.1}
                value={params.speed}
                onChange={(e) => handleParamChange('speed', Number(e.target.value))}
                className="w-full accent-accent-primary"
              />
            </div>

            {/* Volume */}
            <div className="space-y-1">
              <div className="flex justify-between">
                <label className="text-sm text-text-secondary">Volume</label>
                <span className="text-sm text-text-muted">{Math.round(params.volume * 100)}%</span>
              </div>
              <input
                type="range"
                min={0}
                max={1}
                step={0.1}
                value={params.volume}
                onChange={(e) => handleParamChange('volume', Number(e.target.value))}
                className="w-full accent-accent-primary"
              />
            </div>
          </div>

          {/* Advanced Settings Toggle */}
          <button
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="flex items-center gap-2 text-sm text-text-secondary hover:text-text-primary transition-colors"
          >
            <Settings2 size={16} />
            Advanced Settings
            {showAdvanced ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
          </button>

          {/* Advanced Settings */}
          {showAdvanced && (
            <div className="space-y-3 pt-2 border-t border-border-primary">
              {/* Pitch */}
              <div className="space-y-1">
                <div className="flex justify-between">
                  <label className="text-sm text-text-secondary">Pitch</label>
                  <span className="text-sm text-text-muted">{params.pitch.toFixed(1)}</span>
                </div>
                <input
                  type="range"
                  min={0.5}
                  max={2.0}
                  step={0.1}
                  value={params.pitch}
                  onChange={(e) => handleParamChange('pitch', Number(e.target.value))}
                  className="w-full accent-accent-primary"
                />
              </div>
            </div>
          )}

          {/* Generate Button */}
          <button
            onClick={handleGenerate}
            disabled={isGenerating || !text.trim()}
            className="w-full py-2.5 bg-accent-primary hover:bg-accent-hover disabled:opacity-50 disabled:cursor-not-allowed text-white font-medium rounded-lg transition-colors flex items-center justify-center gap-2"
          >
            {isGenerating ? (
              <>
                <Loader2 size={18} className="animate-spin" />
                Generating...
              </>
            ) : (
              <>
                <Volume2 size={18} />
                Generate Speech
              </>
            )}
          </button>
        </div>
      </div>

      {/* Main Content - Audio List */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <div className="p-4 border-b border-border-primary">
          <h2 className="text-lg font-semibold text-text-primary">Generated Audio</h2>
          <p className="text-sm text-text-muted">
            {generatedAudios.length} audio file{generatedAudios.length !== 1 ? 's' : ''} generated
          </p>
        </div>

        {/* Audio List */}
        <div className="flex-1 overflow-y-auto p-4">
          {generatedAudios.length > 0 ? (
            <div className="space-y-3">
              {generatedAudios.map((audio) => (
                <div
                  key={audio.id}
                  className="bg-bg-secondary rounded-lg border border-border-primary p-4"
                >
                  <div className="flex items-start gap-4">
                    {/* Play Button */}
                    <button
                      onClick={() => handlePlay(audio)}
                      className={`w-12 h-12 rounded-full flex items-center justify-center flex-shrink-0 transition-colors ${
                        playingId === audio.id
                          ? 'bg-accent-primary text-white'
                          : 'bg-bg-tertiary text-text-primary hover:bg-bg-elevated'
                      }`}
                    >
                      {playingId === audio.id ? (
                        <Pause size={24} />
                      ) : (
                        <Play size={24} className="ml-1" />
                      )}
                    </button>

                    {/* Audio Info */}
                    <div className="flex-1 min-w-0">
                      <p className="text-text-primary line-clamp-2">{audio.text}</p>
                      <div className="flex items-center gap-2 mt-2 text-xs text-text-muted">
                        <span>{VOICE_MODELS.find((m) => m.id === audio.model)?.name}</span>
                        <span>•</span>
                        <span>
                          {
                            VOICES[audio.model as keyof typeof VOICES]?.find(
                              (v) => v.id === audio.voice
                            )?.name
                          }
                        </span>
                        <span>•</span>
                        <span>{audio.duration.toFixed(1)}s</span>
                      </div>
                    </div>

                    {/* Actions */}
                    <div className="flex items-center gap-1">
                      {playingId === audio.id && (
                        <button
                          onClick={handleStop}
                          className="p-2 text-text-secondary hover:text-text-primary hover:bg-bg-tertiary rounded-lg transition-colors"
                          title="Stop"
                        >
                          <Square size={18} />
                        </button>
                      )}
                      <button
                        onClick={() => handleDownload(audio)}
                        className="p-2 text-text-secondary hover:text-text-primary hover:bg-bg-tertiary rounded-lg transition-colors"
                        title="Download"
                      >
                        <Download size={18} />
                      </button>
                      <button
                        onClick={() => handleDelete(audio.id)}
                        className="p-2 text-text-secondary hover:text-status-error hover:bg-status-error/10 rounded-lg transition-colors"
                        title="Delete"
                      >
                        <Trash2 size={18} />
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="h-full flex items-center justify-center">
              <div className="text-center text-text-muted">
                <Mic size={48} className="mx-auto mb-4 opacity-50" />
                <p className="text-lg">No audio generated yet</p>
                <p className="text-sm mt-1">Enter text and click generate to create speech</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default AudioPage;
