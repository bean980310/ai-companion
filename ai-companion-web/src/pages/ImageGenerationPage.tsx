import { useState, useCallback } from 'react';
import {
  Image,
  Upload,
  Download,
  Trash2,
  Settings2,
  ChevronDown,
  ChevronUp,
  Loader2,
} from 'lucide-react';
import type { ImageGenerationParams } from '../types';

const SAMPLERS = [
  'Euler',
  'Euler a',
  'DPM++ 2M',
  'DPM++ 2M Karras',
  'DPM++ SDE',
  'DPM++ SDE Karras',
  'DDIM',
  'UniPC',
];

const SCHEDULERS = ['Normal', 'Karras', 'Exponential', 'SGM Uniform'];

const ASPECT_RATIOS = [
  { label: '1:1', width: 512, height: 512 },
  { label: '1:1 HD', width: 1024, height: 1024 },
  { label: '3:4', width: 768, height: 1024 },
  { label: '4:3', width: 1024, height: 768 },
  { label: '9:16', width: 576, height: 1024 },
  { label: '16:9', width: 1024, height: 576 },
];

const DEFAULT_PARAMS: ImageGenerationParams = {
  width: 512,
  height: 512,
  steps: 20,
  cfgScale: 7.0,
  seed: -1,
  randomSeed: true,
  sampler: 'Euler a',
  scheduler: 'Normal',
  clipSkip: 1,
  batchSize: 1,
  batchCount: 1,
  denoiseStrength: 0.75,
};

interface GeneratedImage {
  id: string;
  url: string;
  prompt: string;
  negativePrompt: string;
  params: ImageGenerationParams;
  timestamp: string;
}

export function ImageGenerationPage() {
  const [prompt, setPrompt] = useState('');
  const [negativePrompt, setNegativePrompt] = useState('');
  const [params, setParams] = useState<ImageGenerationParams>(DEFAULT_PARAMS);
  const [isGenerating, setIsGenerating] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [generatedImages, setGeneratedImages] = useState<GeneratedImage[]>([]);
  const [selectedImage, setSelectedImage] = useState<GeneratedImage | null>(null);

  const handleParamChange = useCallback(
    <K extends keyof ImageGenerationParams>(key: K, value: ImageGenerationParams[K]) => {
      setParams((prev) => ({ ...prev, [key]: value }));
    },
    []
  );

  const handleAspectRatioChange = useCallback((width: number, height: number) => {
    setParams((prev) => ({ ...prev, width, height }));
  }, []);

  const handleGenerate = useCallback(async () => {
    if (!prompt.trim()) return;

    setIsGenerating(true);

    try {
      // TODO: Replace with actual API call
      await new Promise((resolve) => setTimeout(resolve, 2000));

      // Simulated response
      const newImage: GeneratedImage = {
        id: `img_${Date.now()}`,
        url: `https://picsum.photos/seed/${Date.now()}/${params.width}/${params.height}`,
        prompt,
        negativePrompt,
        params: { ...params },
        timestamp: new Date().toISOString(),
      };

      setGeneratedImages((prev) => [newImage, ...prev]);
      setSelectedImage(newImage);
    } catch (error) {
      console.error('Failed to generate image:', error);
    } finally {
      setIsGenerating(false);
    }
  }, [prompt, negativePrompt, params]);

  const handleDelete = useCallback((id: string) => {
    setGeneratedImages((prev) => prev.filter((img) => img.id !== id));
    setSelectedImage((prev) => (prev?.id === id ? null : prev));
  }, []);

  const handleDownload = useCallback((image: GeneratedImage) => {
    const link = document.createElement('a');
    link.href = image.url;
    link.download = `generated_${image.id}.png`;
    link.click();
  }, []);

  return (
    <div className="h-full flex bg-bg-primary">
      {/* Left Panel - Parameters */}
      <div className="w-80 flex-shrink-0 border-r border-border-primary overflow-y-auto">
        <div className="p-4 space-y-4">
          {/* Model Selection */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-text-secondary">Model</label>
            <select className="w-full px-3 py-2 bg-bg-secondary border border-border-primary rounded-lg text-text-primary focus:outline-none focus:border-accent-primary">
              <option value="sd15">Stable Diffusion 1.5</option>
              <option value="sdxl">Stable Diffusion XL</option>
              <option value="sd3">Stable Diffusion 3</option>
              <option value="flux">FLUX</option>
            </select>
          </div>

          {/* Prompt */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-text-secondary">Prompt</label>
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Describe your image..."
              className="w-full px-3 py-2 bg-bg-secondary border border-border-primary rounded-lg text-text-primary placeholder-text-muted focus:outline-none focus:border-accent-primary resize-none"
              rows={4}
            />
          </div>

          {/* Negative Prompt */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-text-secondary">Negative Prompt</label>
            <textarea
              value={negativePrompt}
              onChange={(e) => setNegativePrompt(e.target.value)}
              placeholder="What to avoid..."
              className="w-full px-3 py-2 bg-bg-secondary border border-border-primary rounded-lg text-text-primary placeholder-text-muted focus:outline-none focus:border-accent-primary resize-none"
              rows={2}
            />
          </div>

          {/* Aspect Ratio */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-text-secondary">Aspect Ratio</label>
            <div className="grid grid-cols-3 gap-2">
              {ASPECT_RATIOS.map((ratio) => (
                <button
                  key={ratio.label}
                  onClick={() => handleAspectRatioChange(ratio.width, ratio.height)}
                  className={`px-2 py-1.5 text-xs rounded-md transition-colors ${
                    params.width === ratio.width && params.height === ratio.height
                      ? 'bg-accent-primary text-white'
                      : 'bg-bg-tertiary text-text-secondary hover:bg-bg-elevated'
                  }`}
                >
                  {ratio.label}
                </button>
              ))}
            </div>
          </div>

          {/* Basic Parameters */}
          <div className="space-y-3">
            {/* Steps */}
            <div className="space-y-1">
              <div className="flex justify-between">
                <label className="text-sm text-text-secondary">Steps</label>
                <span className="text-sm text-text-muted">{params.steps}</span>
              </div>
              <input
                type="range"
                min={1}
                max={100}
                value={params.steps}
                onChange={(e) => handleParamChange('steps', Number(e.target.value))}
                className="w-full accent-accent-primary"
              />
            </div>

            {/* CFG Scale */}
            <div className="space-y-1">
              <div className="flex justify-between">
                <label className="text-sm text-text-secondary">CFG Scale</label>
                <span className="text-sm text-text-muted">{params.cfgScale.toFixed(1)}</span>
              </div>
              <input
                type="range"
                min={1}
                max={20}
                step={0.5}
                value={params.cfgScale}
                onChange={(e) => handleParamChange('cfgScale', Number(e.target.value))}
                className="w-full accent-accent-primary"
              />
            </div>

            {/* Seed */}
            <div className="space-y-1">
              <div className="flex justify-between items-center">
                <label className="text-sm text-text-secondary">Seed</label>
                <label className="flex items-center gap-1 text-xs text-text-muted">
                  <input
                    type="checkbox"
                    checked={params.randomSeed}
                    onChange={(e) => handleParamChange('randomSeed', e.target.checked)}
                    className="rounded accent-accent-primary"
                  />
                  Random
                </label>
              </div>
              <input
                type="number"
                value={params.seed}
                onChange={(e) => handleParamChange('seed', Number(e.target.value))}
                disabled={params.randomSeed}
                className="w-full px-3 py-1.5 bg-bg-secondary border border-border-primary rounded-lg text-text-primary text-sm focus:outline-none focus:border-accent-primary disabled:opacity-50"
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
              {/* Sampler */}
              <div className="space-y-1">
                <label className="text-sm text-text-secondary">Sampler</label>
                <select
                  value={params.sampler}
                  onChange={(e) => handleParamChange('sampler', e.target.value)}
                  className="w-full px-3 py-1.5 bg-bg-secondary border border-border-primary rounded-lg text-text-primary text-sm focus:outline-none focus:border-accent-primary"
                >
                  {SAMPLERS.map((sampler) => (
                    <option key={sampler} value={sampler}>
                      {sampler}
                    </option>
                  ))}
                </select>
              </div>

              {/* Scheduler */}
              <div className="space-y-1">
                <label className="text-sm text-text-secondary">Scheduler</label>
                <select
                  value={params.scheduler}
                  onChange={(e) => handleParamChange('scheduler', e.target.value)}
                  className="w-full px-3 py-1.5 bg-bg-secondary border border-border-primary rounded-lg text-text-primary text-sm focus:outline-none focus:border-accent-primary"
                >
                  {SCHEDULERS.map((scheduler) => (
                    <option key={scheduler} value={scheduler}>
                      {scheduler}
                    </option>
                  ))}
                </select>
              </div>

              {/* Clip Skip */}
              <div className="space-y-1">
                <div className="flex justify-between">
                  <label className="text-sm text-text-secondary">Clip Skip</label>
                  <span className="text-sm text-text-muted">{params.clipSkip}</span>
                </div>
                <input
                  type="range"
                  min={1}
                  max={12}
                  value={params.clipSkip}
                  onChange={(e) => handleParamChange('clipSkip', Number(e.target.value))}
                  className="w-full accent-accent-primary"
                />
              </div>

              {/* Batch Size */}
              <div className="space-y-1">
                <div className="flex justify-between">
                  <label className="text-sm text-text-secondary">Batch Size</label>
                  <span className="text-sm text-text-muted">{params.batchSize}</span>
                </div>
                <input
                  type="range"
                  min={1}
                  max={8}
                  value={params.batchSize}
                  onChange={(e) => handleParamChange('batchSize', Number(e.target.value))}
                  className="w-full accent-accent-primary"
                />
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
                Generating...
              </>
            ) : (
              <>
                <Image size={18} />
                Generate
              </>
            )}
          </button>
        </div>
      </div>

      {/* Main Content - Image Display */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Selected Image Preview */}
        <div className="flex-1 flex items-center justify-center p-4">
          {selectedImage ? (
            <div className="relative max-w-full max-h-full">
              <img
                src={selectedImage.url}
                alt={selectedImage.prompt}
                className="max-w-full max-h-[calc(100vh-16rem)] object-contain rounded-lg shadow-lg"
              />
              <div className="absolute bottom-4 right-4 flex gap-2">
                <button
                  onClick={() => handleDownload(selectedImage)}
                  className="p-2 bg-bg-elevated/80 backdrop-blur-sm rounded-lg hover:bg-bg-tertiary transition-colors"
                  title="Download"
                >
                  <Download size={20} className="text-text-primary" />
                </button>
                <button
                  onClick={() => handleDelete(selectedImage.id)}
                  className="p-2 bg-bg-elevated/80 backdrop-blur-sm rounded-lg hover:bg-status-error/20 transition-colors"
                  title="Delete"
                >
                  <Trash2 size={20} className="text-status-error" />
                </button>
              </div>
            </div>
          ) : (
            <div className="text-center text-text-muted">
              <Upload size={48} className="mx-auto mb-4 opacity-50" />
              <p className="text-lg">No image selected</p>
              <p className="text-sm mt-1">Generate an image to see it here</p>
            </div>
          )}
        </div>

        {/* Image Info */}
        {selectedImage && (
          <div className="p-4 border-t border-border-primary bg-bg-secondary">
            <div className="text-sm text-text-secondary space-y-1">
              <p>
                <span className="text-text-muted">Prompt:</span> {selectedImage.prompt}
              </p>
              <p>
                <span className="text-text-muted">Size:</span> {selectedImage.params.width} ×{' '}
                {selectedImage.params.height}
              </p>
              <p>
                <span className="text-text-muted">Steps:</span> {selectedImage.params.steps} |{' '}
                <span className="text-text-muted">CFG:</span> {selectedImage.params.cfgScale} |{' '}
                <span className="text-text-muted">Sampler:</span> {selectedImage.params.sampler}
              </p>
            </div>
          </div>
        )}

        {/* Generated Images Gallery */}
        {generatedImages.length > 0 && (
          <div className="h-32 border-t border-border-primary bg-bg-secondary p-2 overflow-x-auto">
            <div className="flex gap-2 h-full">
              {generatedImages.map((image) => (
                <button
                  key={image.id}
                  onClick={() => setSelectedImage(image)}
                  className={`h-full aspect-square rounded-lg overflow-hidden flex-shrink-0 border-2 transition-colors ${
                    selectedImage?.id === image.id
                      ? 'border-accent-primary'
                      : 'border-transparent hover:border-border-primary'
                  }`}
                >
                  <img
                    src={image.url}
                    alt={image.prompt}
                    className="w-full h-full object-cover"
                  />
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default ImageGenerationPage;
