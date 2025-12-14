# WebAssembly (WASM) Support

The Arcade Learning Environment can be compiled to WebAssembly, enabling Atari 2600 emulation directly in web browsers without any server-side components or native installations.

## Overview

WebAssembly support brings ALE to the browser, opening up new possibilities for:

- **Interactive Demonstrations** - Showcase RL algorithms with live, interactive visualizations
- **Educational Applications** - Teach reinforcement learning concepts with hands-on browser-based examples
- **Research Tools** - Rapid prototyping and visual debugging of RL algorithms

Be aware that vectorized environments, JAX integration and the direct python bindings are not supported.
Below we describe how to install and use the WASM module for your web apps.

An [example html file](wasm/example.html) is available for users to develop from.

## Installation

### Option 1: npm Package

Install ALE via npm:

```bash
npm install @farama/ale
```

Then import in your JavaScript:

```javascript
import createALEModule from '@farama/ale';

const ALE = await createALEModule();
const ale = new ALE.ALEInterface();
```

### Option 2: GitHub Release Bundle

Download the standalone bundle from [GitHub Releases](https://github.com/Farama-Foundation/Arcade-Learning-Environment/releases):

1. Download `ale-wasm.zip` from the latest release
2. Extract `ale.js`, `ale.wasm`, and optionally `ale.data` (preloaded ROMs)
3. Serve the files from your web server
4. Load in your HTML:

```html
<script src="ale.js"></script>
<script>
  // createALEModule is available as a global after loading ale.js
  async function init() {
    const ALE = await createALEModule();
    const ale = new ALE.ALEInterface();
  }
  init();
</script>
```

## Usage

### Initializing the Module

```javascript
import createALEModule from './ale.js';

// Initialize ALE
const ALE = await createALEModule();
const ale = new ALE.ALEInterface();

console.log('ALE Version:', ALE.ALEInterface.getVersion());
```

### Loading ROMs

ROMs can be loaded in several ways, from preloaded bundles to dynamic loading:

#### 1. Preloaded ROMs

If the WASM module was built with preloaded ROMs (default in releases), they're available in the virtual filesystem at `/roms/`:

```javascript
const ALE = await createALEModule();
const ale = new ALE.ALEInterface();

// List available ROMs
const files = ALE.FS.readdir('/roms');
const roms = files.filter(f => f !== '.' && f !== '..' && f.endsWith('.bin'));

// Load a ROM
const romPath = '/roms/' + roms[0];  // or choose any ROM from the list
ale.loadROM(romPath);
```

#### 2. File Upload (User Selection)

```html
<input type="file" id="romFile" accept=".bin">
```

```javascript
document.getElementById('romFile').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    await ale.loadROMFromFile(file);
});
```

#### 3. URL Fetch (Remote Loading)

```javascript
await ale.loadROMFromURL('https://example.com/roms/breakout.bin');
console.log('ROM loaded from URL');
```

### Basic Game Loop

```javascript
// Configure ALE
ale.setBool('display_screen', false);
ale.setInt('random_seed', 42);

// Load ROM
ale.loadROM('/roms/breakout.bin');

// Get action space
const actions = ale.getMinimalActionSet();

// Game loop
function step() {
    if (ale.gameOver()) {
        ale.resetGame();
    }

    // Select action (random for this example)
    const action = actions[Math.floor(Math.random() * actions.length)];

    // Execute action
    const reward = ale.act(action);
    const screen = ale.getScreenRGB();
    console.log(`Reward: ${reward}, Lives: ${ale.lives()}`);

    requestAnimationFrame(step);
}

step();
```

### Canvas Rendering

The WASM module provides helper methods for easy canvas rendering:

```javascript
function renderFrame() {
    // renderToCanvas takes a canvas element ID (string)
    ale.renderToCanvas('gameCanvas');
}
```

Or manually using ImageData:

```javascript
function renderFrame() {
    const canvas = document.getElementById('gameCanvas');
    const ctx = canvas.getContext('2d');
    const imageData = ale.getScreenImageData();

    // Render directly to canvas
    ctx.putImageData(imageData, 0, 0);
}
```

## JavaScript API Reference

```javascript
// Configuration
ale.setBool(key: string, value: boolean): void
ale.getBool(key: string): boolean
ale.setInt(key: string, value: number): void
ale.getInt(key: string): number
ale.setFloat(key: string, value: number): void
ale.getFloat(key: string): number
ale.setString(key: string, value: string): void
ale.getString(key: string): string

// ROM Management
ale.loadROM(path: string): void
ale.loadROMFromURL(url: string, filename?: string): Promise<string>
ale.loadROMFromFile(file: File): Promise<string>

// Game Loop
ale.act(action: number): number          // Returns reward
ale.resetGame(): void
ale.gameOver(): boolean
ale.gameTruncated(): boolean
ale.lives(): number
ale.getFrameNumber(): number
ale.getEpisodeFrameNumber(): number

// Observation
ale.getScreenRGB(): Uint8ClampedArray        // Interleaved RGB: [R,G,B,R,G,B,...]
ale.getScreenGrayscale(): Uint8ClampedArray  // Grayscale: [Y,Y,Y,...]
ale.getScreenImageData(): ImageData          // Get screen as RGBA ImageData ready for Canvas
ale.renderToCanvas(canvasId: string): void   // Render directly to canvas by element ID
ale.getScreenWidth(): number
ale.getScreenHeight(): number
ale.getRAM(): Uint8Array
ale.setRAM(index: number, value: number): void

// Action Space
ale.getLegalActionSet(): number[]
ale.getMinimalActionSet(): number[]

// Mode & Difficulty
ale.getAvailableModes(): number[]
ale.setMode(mode: number): void
ale.getMode(): number
ale.getAvailableDifficulties(): number[]
ale.setDifficulty(difficulty: number): void
ale.getDifficulty(): number

// State Management
ale.saveState(): Uint8Array
ale.loadState(state: Uint8Array): void

// Static Methods
ALE.ALEInterface.getVersion(): string
```

## Browser Compatibility

| Browser        | Minimum Version | Notes        |
|----------------|-----------------|--------------|
| Chrome         | 57+             | Full support |
| Firefox        | 52+             | Full support |
| Safari         | 11+             | Full support |
| Edge           | 16+             | Full support |
| Mobile Safari  | 11+             | Full support |
| Chrome Android | 57+             | Full support |

**Requirements:**
- WebAssembly support
- ES6 modules support
- File API support (for ROM uploads)
- Canvas API support

## Troubleshooting

### Module Loading Fails

**Symptom:** `Failed to fetch ale.wasm`

**Causes & Solutions:**
- ❌ Using `file://` protocol → ✅ Use HTTP server
- ❌ CORS policy blocking → ✅ Serve WASM from same origin
- ❌ MIME type incorrect → ✅ Serve `.wasm` as `application/wasm`

### Memory Errors

**Symptom:** `RuntimeError: memory access out of bounds`

**Default Configuration:**
- Initial memory: 32MB (33,554,432 bytes)
- Maximum memory: 128MB (134,217,728 bytes)
- Memory growth: enabled

**Solutions:**
- Check for memory leaks in game loop
- Reduce screen buffer allocations

### Slow Performance

**Causes & Solutions:**
- ❌ Large canvas size → ✅ Use native resolution (160x210)
- ❌ Browser throttling → ✅ Keep tab focused or use Web Workers

### ROM Loading Issues

**Symptom:** `Failed to load ROM`

**Causes & Solutions:**
- ❌ Invalid ROM file → ✅ Verify ROM is valid Atari 2600 ROM
- ❌ ROM not in virtual filesystem → ✅ Use helper methods
- ❌ Path incorrect → ✅ Use absolute paths

## Advanced Usage

### Machine Learning Integration

ALE works seamlessly with TensorFlow.js and other ML libraries. Example preprocessing for neural network policies:

```javascript
import * as tf from '@tensorflow/tfjs';

function preprocessScreen(ale) {
    const screen = ale.getScreenGrayscale();
    const width = ale.getScreenWidth();
    const height = ale.getScreenHeight();

    // Normalize to [0, 1]
    const normalized = new Float32Array(screen.length);
    for (let i = 0; i < screen.length; i++) {
        normalized[i] = screen[i] / 255.0;
    }

    return tf.tensor(normalized).reshape([1, height, width, 1]);
}
```

### State Save/Restore

Save and restore complete emulator states:

```javascript
// Save current state (returns Uint8Array)
const savedStateView = ale.saveState();
// Copy the state to preserve it (important since it's a view into WASM memory)
const savedState = new Uint8Array(savedStateView);

// Perform some actions
ale.act(1);
ale.act(2);

// Restore to saved state
ale.loadState(savedState);
```
