import { useState, useCallback } from 'react';
import {
  BookOpen,
  Sparkles,
  Copy,
  Download,
  Save,
  Loader2,
  ChevronDown,
  ChevronUp,
  Settings2,
  Wand2,
} from 'lucide-react';

const GENRES = [
  { id: 'fantasy', name: 'Fantasy', emoji: '🧙' },
  { id: 'scifi', name: 'Science Fiction', emoji: '🚀' },
  { id: 'romance', name: 'Romance', emoji: '💕' },
  { id: 'mystery', name: 'Mystery', emoji: '🔍' },
  { id: 'horror', name: 'Horror', emoji: '👻' },
  { id: 'adventure', name: 'Adventure', emoji: '⚔️' },
  { id: 'comedy', name: 'Comedy', emoji: '😄' },
  { id: 'drama', name: 'Drama', emoji: '🎭' },
];

const STORY_LENGTHS = [
  { id: 'short', name: 'Short', description: '~500 words' },
  { id: 'medium', name: 'Medium', description: '~1500 words' },
  { id: 'long', name: 'Long', description: '~3000 words' },
];

const WRITING_STYLES = [
  { id: 'descriptive', name: 'Descriptive' },
  { id: 'dialogue-heavy', name: 'Dialogue Heavy' },
  { id: 'action-packed', name: 'Action Packed' },
  { id: 'poetic', name: 'Poetic' },
  { id: 'minimalist', name: 'Minimalist' },
];

interface StoryParams {
  genre: string;
  length: string;
  style: string;
  temperature: number;
  includeDialogue: boolean;
  includeNarration: boolean;
}

interface GeneratedStory {
  id: string;
  title: string;
  content: string;
  prompt: string;
  params: StoryParams;
  wordCount: number;
  timestamp: string;
}

const DEFAULT_PARAMS: StoryParams = {
  genre: 'fantasy',
  length: 'medium',
  style: 'descriptive',
  temperature: 0.8,
  includeDialogue: true,
  includeNarration: true,
};

export function StorytellerPage() {
  const [prompt, setPrompt] = useState('');
  const [params, setParams] = useState<StoryParams>(DEFAULT_PARAMS);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [currentStory, setCurrentStory] = useState<GeneratedStory | null>(null);
  const [savedStories, setSavedStories] = useState<GeneratedStory[]>([]);
  const [showSaved, setShowSaved] = useState(false);

  const handleParamChange = useCallback(
    <K extends keyof StoryParams>(key: K, value: StoryParams[K]) => {
      setParams((prev) => ({ ...prev, [key]: value }));
    },
    []
  );

  const handleGenerate = useCallback(async () => {
    if (!prompt.trim()) return;

    setIsGenerating(true);

    try {
      // TODO: Replace with actual API call
      await new Promise((resolve) => setTimeout(resolve, 2000));

      const genre = GENRES.find((g) => g.id === params.genre);

      // Simulated response
      const storyContent = `# The ${genre?.emoji} ${prompt}

Once upon a time, in a world where magic and technology intertwined, there lived a young adventurer named Luna. She had always dreamed of exploring the ancient ruins that lay beyond the Crystalline Mountains.

## Chapter 1: The Beginning

The morning sun cast long shadows across the cobblestone streets as Luna gathered her supplies. Her backpack was worn but sturdy, filled with provisions for the journey ahead.

"Are you sure about this?" asked her mentor, an elderly mage named Theron.

Luna nodded firmly. "I've been preparing for this my whole life. The legends speak of a powerful artifact hidden within those ruins—one that could change everything."

Theron sighed, his wrinkled face softening. "Very well. But remember: the greatest treasures are often not the ones we seek, but the ones we discover along the way."

## Chapter 2: The Journey

The path through the Whispering Woods was treacherous. Ancient trees loomed overhead, their branches creating a canopy so thick that barely any sunlight penetrated through. Strange sounds echoed from the shadows—creatures that Luna had only read about in dusty old tomes.

She pressed on, guided by the compass her father had given her before he disappeared on his own expedition years ago. The needle spun wildly at first, then settled, pointing toward a direction that no ordinary compass would indicate.

"Trust in the magic," she whispered to herself, remembering her father's words.

## Chapter 3: The Discovery

After three days of travel, Luna finally reached the entrance to the ruins. Massive stone pillars, covered in luminescent moss, framed a doorway that seemed to pulse with an otherworldly energy.

As she stepped inside, the world around her transformed. What had appeared to be ancient decay was actually a perfectly preserved sanctuary, filled with wonders beyond imagination.

And there, at the center of it all, was not the artifact she had expected—but something far more precious: a letter from her father, and a map to where he waited for her.

---

*To be continued...*`;

      const newStory: GeneratedStory = {
        id: `story_${Date.now()}`,
        title: `The ${genre?.name} Tale of ${prompt.slice(0, 30)}`,
        content: storyContent,
        prompt,
        params: { ...params },
        wordCount: storyContent.split(/\s+/).length,
        timestamp: new Date().toISOString(),
      };

      setCurrentStory(newStory);
    } catch (error) {
      console.error('Failed to generate story:', error);
    } finally {
      setIsGenerating(false);
    }
  }, [prompt, params]);

  const handleContinue = useCallback(async () => {
    if (!currentStory) return;

    setIsGenerating(true);

    try {
      // TODO: Replace with actual API call for continuation
      await new Promise((resolve) => setTimeout(resolve, 1500));

      const continuation = `

## Chapter 4: The Reunion

Luna's heart raced as she followed the map through winding corridors. Each step brought her closer to the truth she had sought for so long. The air grew warmer, and a soft golden light began to emanate from somewhere ahead.

She turned the final corner and stopped. There, sitting at a wooden desk covered in scrolls and artifacts, was her father—older now, with gray streaks in his once-dark hair, but unmistakably him.

"Luna," he said, his voice catching. "You found me."

*The adventure continues...*`;

      setCurrentStory((prev) =>
        prev
          ? {
              ...prev,
              content: prev.content + continuation,
              wordCount: (prev.content + continuation).split(/\s+/).length,
            }
          : null
      );
    } catch (error) {
      console.error('Failed to continue story:', error);
    } finally {
      setIsGenerating(false);
    }
  }, [currentStory]);

  const handleSave = useCallback(() => {
    if (!currentStory) return;

    setSavedStories((prev) => {
      const exists = prev.find((s) => s.id === currentStory.id);
      if (exists) {
        return prev.map((s) => (s.id === currentStory.id ? currentStory : s));
      }
      return [currentStory, ...prev];
    });
  }, [currentStory]);

  const handleCopy = useCallback(() => {
    if (!currentStory) return;
    navigator.clipboard.writeText(currentStory.content);
  }, [currentStory]);

  const handleDownload = useCallback(() => {
    if (!currentStory) return;

    const blob = new Blob([currentStory.content], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${currentStory.title.replace(/\s+/g, '_')}.md`;
    link.click();
    URL.revokeObjectURL(url);
  }, [currentStory]);

  const handleLoadStory = useCallback((story: GeneratedStory) => {
    setCurrentStory(story);
    setPrompt(story.prompt);
    setParams(story.params);
    setShowSaved(false);
  }, []);

  return (
    <div className="h-full flex bg-bg-primary">
      {/* Left Panel - Parameters */}
      <div className="w-80 flex-shrink-0 border-r border-border-primary overflow-y-auto">
        <div className="p-4 space-y-4">
          {/* Saved Stories Toggle */}
          <button
            onClick={() => setShowSaved(!showSaved)}
            className="w-full flex items-center justify-between px-3 py-2 bg-bg-secondary border border-border-primary rounded-lg text-text-primary hover:bg-bg-tertiary transition-colors"
          >
            <span className="flex items-center gap-2">
              <BookOpen size={16} />
              Saved Stories ({savedStories.length})
            </span>
            {showSaved ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
          </button>

          {/* Saved Stories List */}
          {showSaved && savedStories.length > 0 && (
            <div className="space-y-2 max-h-48 overflow-y-auto">
              {savedStories.map((story) => (
                <button
                  key={story.id}
                  onClick={() => handleLoadStory(story)}
                  className="w-full text-left px-3 py-2 bg-bg-tertiary rounded-lg hover:bg-bg-elevated transition-colors"
                >
                  <p className="text-sm text-text-primary truncate">{story.title}</p>
                  <p className="text-xs text-text-muted">{story.wordCount} words</p>
                </button>
              ))}
            </div>
          )}

          {/* Story Prompt */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-text-secondary">Story Idea</label>
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Describe your story idea..."
              className="w-full px-3 py-2 bg-bg-secondary border border-border-primary rounded-lg text-text-primary placeholder-text-muted focus:outline-none focus:border-accent-primary resize-none"
              rows={4}
            />
          </div>

          {/* Genre Selection */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-text-secondary">Genre</label>
            <div className="grid grid-cols-2 gap-2">
              {GENRES.map((genre) => (
                <button
                  key={genre.id}
                  onClick={() => handleParamChange('genre', genre.id)}
                  className={`px-3 py-2 text-sm rounded-lg transition-colors flex items-center gap-2 ${
                    params.genre === genre.id
                      ? 'bg-accent-primary text-white'
                      : 'bg-bg-tertiary text-text-secondary hover:bg-bg-elevated'
                  }`}
                >
                  <span>{genre.emoji}</span>
                  <span className="truncate">{genre.name}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Story Length */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-text-secondary">Length</label>
            <div className="grid grid-cols-3 gap-2">
              {STORY_LENGTHS.map((length) => (
                <button
                  key={length.id}
                  onClick={() => handleParamChange('length', length.id)}
                  className={`px-2 py-2 text-xs rounded-lg transition-colors ${
                    params.length === length.id
                      ? 'bg-accent-primary text-white'
                      : 'bg-bg-tertiary text-text-secondary hover:bg-bg-elevated'
                  }`}
                >
                  <div>{length.name}</div>
                  <div className="opacity-75">{length.description}</div>
                </button>
              ))}
            </div>
          </div>

          {/* Writing Style */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-text-secondary">Writing Style</label>
            <select
              value={params.style}
              onChange={(e) => handleParamChange('style', e.target.value)}
              className="w-full px-3 py-2 bg-bg-secondary border border-border-primary rounded-lg text-text-primary focus:outline-none focus:border-accent-primary"
            >
              {WRITING_STYLES.map((style) => (
                <option key={style.id} value={style.id}>
                  {style.name}
                </option>
              ))}
            </select>
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
              {/* Creativity */}
              <div className="space-y-1">
                <div className="flex justify-between">
                  <label className="text-sm text-text-secondary">Creativity</label>
                  <span className="text-sm text-text-muted">
                    {params.temperature.toFixed(1)}
                  </span>
                </div>
                <input
                  type="range"
                  min={0.1}
                  max={1.5}
                  step={0.1}
                  value={params.temperature}
                  onChange={(e) => handleParamChange('temperature', Number(e.target.value))}
                  className="w-full accent-accent-primary"
                />
              </div>

              {/* Include Options */}
              <div className="space-y-2">
                <label className="flex items-center gap-2 text-sm text-text-secondary">
                  <input
                    type="checkbox"
                    checked={params.includeDialogue}
                    onChange={(e) => handleParamChange('includeDialogue', e.target.checked)}
                    className="rounded accent-accent-primary"
                  />
                  Include Dialogue
                </label>
                <label className="flex items-center gap-2 text-sm text-text-secondary">
                  <input
                    type="checkbox"
                    checked={params.includeNarration}
                    onChange={(e) => handleParamChange('includeNarration', e.target.checked)}
                    className="rounded accent-accent-primary"
                  />
                  Include Narration
                </label>
              </div>
            </div>
          )}

          {/* Generate Button */}
          <button
            onClick={handleGenerate}
            disabled={isGenerating || !prompt.trim()}
            className="w-full py-2.5 bg-accent-primary hover:bg-accent-hover disabled:opacity-50 disabled:cursor-not-allowed text-white font-medium rounded-lg transition-colors flex items-center justify-center gap-2"
          >
            {isGenerating ? (
              <>
                <Loader2 size={18} className="animate-spin" />
                Writing...
              </>
            ) : (
              <>
                <Sparkles size={18} />
                Generate Story
              </>
            )}
          </button>
        </div>
      </div>

      {/* Main Content - Story Display */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {currentStory ? (
          <>
            {/* Story Header */}
            <div className="p-4 border-b border-border-primary flex items-center justify-between">
              <div>
                <h2 className="text-lg font-semibold text-text-primary">{currentStory.title}</h2>
                <p className="text-sm text-text-muted">
                  {currentStory.wordCount} words • {GENRES.find((g) => g.id === currentStory.params.genre)?.name}
                </p>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={handleContinue}
                  disabled={isGenerating}
                  className="p-2 text-text-secondary hover:text-text-primary hover:bg-bg-tertiary rounded-lg transition-colors"
                  title="Continue Story"
                >
                  <Wand2 size={20} />
                </button>
                <button
                  onClick={handleCopy}
                  className="p-2 text-text-secondary hover:text-text-primary hover:bg-bg-tertiary rounded-lg transition-colors"
                  title="Copy"
                >
                  <Copy size={20} />
                </button>
                <button
                  onClick={handleDownload}
                  className="p-2 text-text-secondary hover:text-text-primary hover:bg-bg-tertiary rounded-lg transition-colors"
                  title="Download"
                >
                  <Download size={20} />
                </button>
                <button
                  onClick={handleSave}
                  className="p-2 text-text-secondary hover:text-accent-secondary hover:bg-accent-secondary/10 rounded-lg transition-colors"
                  title="Save"
                >
                  <Save size={20} />
                </button>
              </div>
            </div>

            {/* Story Content */}
            <div className="flex-1 overflow-y-auto p-6">
              <div className="max-w-3xl mx-auto prose prose-invert prose-lg">
                {currentStory.content.split('\n').map((line, index) => {
                  if (line.startsWith('# ')) {
                    return (
                      <h1 key={index} className="text-3xl font-bold text-text-primary mt-0 mb-6">
                        {line.slice(2)}
                      </h1>
                    );
                  }
                  if (line.startsWith('## ')) {
                    return (
                      <h2 key={index} className="text-xl font-semibold text-text-primary mt-8 mb-4">
                        {line.slice(3)}
                      </h2>
                    );
                  }
                  if (line.startsWith('---')) {
                    return <hr key={index} className="border-border-primary my-8" />;
                  }
                  if (line.startsWith('*') && line.endsWith('*') && !line.startsWith('**')) {
                    return (
                      <p key={index} className="text-text-muted italic text-center my-6">
                        {line.slice(1, -1)}
                      </p>
                    );
                  }
                  if (line.trim() === '') {
                    return <br key={index} />;
                  }
                  return (
                    <p key={index} className="text-text-secondary leading-relaxed mb-4">
                      {line}
                    </p>
                  );
                })}
              </div>
            </div>
          </>
        ) : (
          <div className="h-full flex items-center justify-center">
            <div className="text-center text-text-muted">
              <BookOpen size={48} className="mx-auto mb-4 opacity-50" />
              <p className="text-lg">No story yet</p>
              <p className="text-sm mt-1">Enter a story idea and click generate</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default StorytellerPage;
