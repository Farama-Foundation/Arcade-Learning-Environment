# WebAssembly (WASM) Support

The Arcade Learning Environment can be compiled to WebAssembly, enabling Atari 2600 emulation directly in web browsers without any server-side components or native installations.

## Overview

WebAssembly support brings ALE to the browser, opening up new possibilities for:

- **Interactive Demonstrations** - Showcase RL algorithms with live, interactive visualizations
- **Educational Applications** - Teach reinforcement learning concepts with hands-on browser-based examples
- **Research Tools** - Rapid prototyping and visual debugging of RL algorithms
- **Web-based Training** - Run lightweight RL training directly in the browser
- **Accessibility** - Make ALE available to anyone with a web browser, no installation required

### What Works in WASM

✅ **Full ALE Core API** - Complete access to all `ALEInterface` methods
✅ **ROM Loading** - Multiple methods: File upload, URL fetch, pre-bundled
✅ **Screen Rendering** - Direct rendering to HTML5 Canvas
✅ **Game Loop** - Full game execution with action/reward/observation cycle
✅ **Configuration** - All ALE settings accessible
✅ **State Management** - Save/restore game states
✅ **Action Spaces** - Query legal and minimal action sets
✅ **Modes & Difficulties** - Full support for game variations

### What Doesn't Work in WASM

❌ **Vectorized Environments** - C++ threading limitations in WASM
❌ **XLA/JAX Integration** - Browser-only environment
❌ **Python Bindings** - JavaScript bindings instead
❌ **Multi-threading** - Limited by browser security model

## Architecture

### Build Process

```
C++ ALE Core
    ↓
Emscripten Compiler (emcc)
    ↓
WebAssembly Binary (.wasm) + JavaScript Glue (.js)
    ↓
Browser Runtime
```

### Components

1. **Emscripten Toolchain** - Compiles C++ to WebAssembly
2. **Embind** - Generates JavaScript bindings from C++ classes
3. **vcpkg** - Manages C++ dependencies (SDL2, zlib)
4. **SDL2 Compatibility Layer** - Emscripten provides SDL2 → HTML5 Canvas mapping

### Memory Model

```
JavaScript Heap
    ↓
WebAssembly Linear Memory (32MB initial, 128MB max)
    ├── ALE Core (~5-10MB)
    ├── ROM Data (~8KB per ROM)
    ├── Screen Buffers (~200KB)
    └── Emulator State (~1-2MB)
```

## Building WASM

### Prerequisites

1. **Emscripten SDK (emsdk)**
   ```bash
   git clone https://github.com/emscripten-core/emsdk.git
   cd emsdk
   ./emsdk install latest
   ./emsdk activate latest
   source ./emsdk_env.sh
   ```

2. **vcpkg**
   ```bash
   git clone https://github.com/microsoft/vcpkg.git
   ./vcpkg/bootstrap-vcpkg.sh
   export VCPKG_ROOT=$(pwd)/vcpkg
   ```

### Build Commands

```bash
# Set the target triplet
export VCPKG_DEFAULT_TRIPLET=wasm32-emscripten

# Configure
mkdir build-wasm-examples && cd build-wasm-examples
emcmake cmake ../ \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_WASM_LIB=ON

# Build
emmake make -j$(nproc)

# Output: build-wasm-examples/src/ale/wasm-examples/ale.js and ale.wasm-examples
```

See [WASM_BUILD_INSTRUCTIONS.md](../WASM_BUILD_INSTRUCTIONS.md) for detailed build instructions.

## Usage

### Loading the Module

```javascript
import createALEModule from './ale.js';

// Initialize ALE
const ALE = await createALEModule();
const ale = new ALE.ALEInterface();

console.log('ALE Version:', ALE.ALEInterface.getVersion());
```

### Loading ROMs

Three methods are supported:

#### 1. File Upload (User Selection)

```javascript
document.getElementById('romFile').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    await ale.loadROMFromFile(file);
    console.log('ROM loaded from file upload');
});
```

```html
<input type="file" id="romFile" accept=".bin,.a26">
```

#### 2. URL Fetch (Remote Loading)

```javascript
await ale.loadROMFromURL('https://example.com/roms/breakout.bin');
console.log('ROM loaded from URL');
```

#### 3. Pre-bundled (Module Initialization)

```javascript
import breakoutROM from './roms/breakout.bin';  // As ArrayBuffer

const ALE = await createALEModule({
    roms: {
        'breakout.bin': new Uint8Array(breakoutROM)
    }
});

const ale = new ALE.ALEInterface();
ale.loadROM('/roms/breakout.bin');
```

### Basic Game Loop

```javascript
// Configure ALE
ale.setBool('display_screen', false);
ale.setInt('random_seed', 42);

// Load ROM
await ale.loadROMFromURL('breakout.bin');

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

    // Get observation
    const screen = ale.getScreenRGB();
    const ram = ale.getRAM();

    console.log(`Reward: ${reward}, Lives: ${ale.lives()}`);

    requestAnimationFrame(step);
}

step();
```

### Canvas Rendering

```javascript
const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');

function renderFrame() {
    const width = ale.getScreenWidth();
    const height = ale.getScreenHeight();
    const rgb = ale.getScreenRGB();

    // Convert planar RGB to interleaved RGBA
    const imageData = new Uint8ClampedArray(width * height * 4);
    for (let i = 0; i < width * height; i++) {
        imageData[i * 4] = rgb[i];                         // R
        imageData[i * 4 + 1] = rgb[width * height + i];    // G
        imageData[i * 4 + 2] = rgb[width * height * 2 + i];// B
        imageData[i * 4 + 3] = 255;                        // A
    }

    // Create temporary canvas at native resolution
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = width;
    tempCanvas.height = height;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.putImageData(new ImageData(imageData, width, height), 0, 0);

    // Scale to display canvas (pixelated for retro look)
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(tempCanvas, 0, 0, canvas.width, canvas.height);
}
```

## Examples

Interactive examples are provided in [`docs/wasm/examples/`](wasm-examples/examples/):

### 1. Basic Example (`01-basic.html`)

Demonstrates module initialization and basic API usage without ROM loading.

**Features:**
- Loading the WASM module
- Creating ALEInterface instances
- Configuration API (setBool, setInt, etc.)
- Version information

**Use case:** Understanding the API structure

### 2. ROM Upload Example (`02-rom-upload.html`)

Shows how to load and inspect ROM files.

**Features:**
- File upload handling
- ROM metadata inspection
- Action space queries
- Mode/difficulty enumeration
- Basic game state monitoring

**Use case:** Exploring ROM structure and capabilities

### 3. Canvas Rendering Example (`03-canvas-render.html`)

Full interactive gameplay with visual rendering.

**Features:**
- Real-time Canvas rendering
- Keyboard controls (arrows + space)
- FPS monitoring and control
- Random AI mode
- Game statistics display
- Auto-reset functionality

**Use case:** Interactive gameplay and demonstrations

### 4. RL Training Example (`04-rl-training.html`)

Reinforcement learning training interface with visualization.

**Features:**
- Training loop implementation
- Episode statistics tracking
- Real-time reward charts
- Configurable parameters (episodes, epsilon, max steps)
- Performance metrics
- Training logs

**Use case:** Running RL experiments in the browser

See the [WASM Examples README](wasm/README.md) for detailed documentation of each example.

## JavaScript API Reference

### Core Methods

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
ale.getScreenRGB(): Uint8ClampedArray
ale.getScreenGrayscale(): Uint8ClampedArray
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
ale.saveState(): string
ale.loadState(state: string): void

// Static Methods
ALE.ALEInterface.getVersion(): string
```

### Helper Methods

Additional convenience methods added by `post.js`:

```javascript
// Get screen as ImageData for direct Canvas use
ale.getScreenImageData(): ImageData

// Render directly to a canvas element
ale.renderToCanvas(canvas: HTMLCanvasElement): void
```

## Performance Optimization

### 1. Disable Rendering During Training

```javascript
ale.setBool('display_screen', false);
ale.setBool('sound', false);
```

Can improve training speed by 10-100x.

### 2. Use Grayscale Instead of RGB

```javascript
const screen = ale.getScreenGrayscale();  // 1/3 the data size
```

### 3. Limit Frame Rate

```javascript
const targetFPS = 30;
const frameDelay = 1000 / targetFPS;

function gameLoop() {
    // ... game logic ...

    setTimeout(() => {
        requestAnimationFrame(gameLoop);
    }, frameDelay);
}
```

### 4. Batch UI Updates

```javascript
// Update UI every N frames instead of every frame
if (frameCount % 10 === 0) {
    updateDisplay();
}
```

### 5. Use Web Workers (Advanced)

Run training in a background thread:

```javascript
// main.js
const worker = new Worker('training-worker.js');
worker.postMessage({ type: 'start', episodes: 100 });

worker.onmessage = (e) => {
    if (e.data.type === 'episode_complete') {
        updateChart(e.data.reward);
    }
};

// training-worker.js
importScripts('ale.js');

self.onmessage = async (e) => {
    if (e.data.type === 'start') {
        const ALE = await createALEModule();
        const ale = new ALE.ALEInterface();
        // ... training loop ...
    }
};
```

## Build Configuration

### CMake Options

The WASM build is controlled by the `BUILD_WASM_LIB` option in CMake:

```cmake
option(BUILD_WASM_LIB "Build WebAssembly Interface" OFF)
```

When enabled, it automatically:
- Disables Python bindings (`BUILD_PYTHON_LIB=OFF`)
- Disables vectorization (`BUILD_VECTOR_LIB=OFF`)
- Disables XLA support (`BUILD_VECTOR_XLA_LIB=OFF`)
- Enables SDL support (`SDL_SUPPORT=ON`)

### Emscripten Flags

Key compilation flags (in `src/ale/wasm/CMakeLists.txt`):

```cmake
-sWASM=1                          # Generate WebAssembly
-sMODULARIZE=1                    # Export as ES6 module
-sEXPORT_NAME='createALEModule'   # Module function name
-sALLOW_MEMORY_GROWTH=1           # Dynamic memory allocation
-sINITIAL_MEMORY=33554432         # 32MB initial
-sMAXIMUM_MEMORY=134217728        # 128MB maximum
-sUSE_SDL=2                       # Enable SDL2 support
-lembind                          # Link Embind library
--bind                            # Enable bindings
```

### vcpkg Triplet

Custom triplet at `cmake/custom-triplets/wasm32-emscripten.cmake`:

```cmake
set(VCPKG_TARGET_ARCHITECTURE wasm32)
set(VCPKG_CMAKE_SYSTEM_NAME Emscripten)
set(VCPKG_LIBRARY_LINKAGE static)
set(VCPKG_CRT_LINKAGE dynamic)
```

## Continuous Integration

WASM builds are automated via GitHub Actions (`.github/workflows/wasm-build.yml`):

**Triggers:**
- Push to `master` or `two_player_support` branches
- Pull requests to `master`
- Manual workflow dispatch

**Outputs:**
- `ale.js` - JavaScript glue code
- `ale.wasm` - WebAssembly binary
- Artifacts available for 30 days

**Workflow steps:**
1. Setup Emscripten SDK
2. Configure vcpkg with caching
3. Run CMake with WASM configuration
4. Build with emmake
5. Verify output files
6. Upload artifacts
7. Generate build summary

## File Size Reference

Typical output sizes (Release build with compression):

| File | Uncompressed | gzip | brotli |
|------|-------------|------|--------|
| ale.js | ~1.5-2.5 MB | ~400-600 KB | ~300-500 KB |
| ale.wasm | ~1-2 MB | ~500-800 KB | ~400-600 KB |
| **Total** | **~3-4.5 MB** | **~1 MB** | **~700-1100 KB** |

**Recommendation:** Serve with Brotli compression for optimal load times.

## Browser Compatibility

| Browser | Minimum Version | Notes |
|---------|----------------|-------|
| Chrome | 57+ | Full support |
| Firefox | 52+ | Full support |
| Safari | 11+ | Full support |
| Edge | 16+ | Full support |
| Mobile Safari | 11+ | Full support |
| Chrome Android | 57+ | Full support |

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

**Solutions:**
- Increase memory limits in `src/ale/wasm/CMakeLists.txt`
- Check for memory leaks in game loop
- Reduce ROM count or screen buffer allocations

### Slow Performance

**Causes & Solutions:**
- ❌ Rendering every frame → ✅ Disable rendering during training
- ❌ Large canvas size → ✅ Use native resolution (160x210)
- ❌ Browser throttling → ✅ Keep tab focused or use Web Workers
- ❌ Debug build → ✅ Use Release build (`-O3`)

### ROM Loading Issues

**Symptom:** `Failed to load ROM`

**Causes & Solutions:**
- ❌ Invalid ROM file → ✅ Verify ROM is valid Atari 2600 ROM
- ❌ ROM not in virtual filesystem → ✅ Use helper methods
- ❌ Path incorrect → ✅ Use absolute paths in virtual filesystem

## Advanced Topics

### Custom ROM Management with IndexedDB

```javascript
class ROMCache {
    constructor() {
        this.db = null;
    }

    async init() {
        return new Promise((resolve, reject) => {
            const request = indexedDB.open('ale-roms', 1);

            request.onupgradeneeded = (e) => {
                const db = e.target.result;
                if (!db.objectStoreNames.contains('roms')) {
                    db.createObjectStore('roms', { keyPath: 'name' });
                }
            };

            request.onsuccess = () => {
                this.db = request.result;
                resolve();
            };

            request.onerror = () => reject(request.error);
        });
    }

    async saveROM(name, data) {
        const transaction = this.db.transaction(['roms'], 'readwrite');
        const store = transaction.objectStore('roms');
        return store.put({ name, data, timestamp: Date.now() });
    }

    async getROM(name) {
        const transaction = this.db.transaction(['roms'], 'readonly');
        const store = transaction.objectStore('roms');
        const request = store.get(name);

        return new Promise((resolve, reject) => {
            request.onsuccess = () => resolve(request.result?.data);
            request.onerror = () => reject(request.error);
        });
    }
}

// Usage
const cache = new ROMCache();
await cache.init();

// Save ROM
const romData = new Uint8Array(await file.arrayBuffer());
await cache.saveROM('breakout.bin', romData);

// Load from cache
const cachedData = await cache.getROM('breakout.bin');
if (cachedData) {
    Module.FS.writeFile('/roms/breakout.bin', cachedData);
    ale.loadROM('/roms/breakout.bin');
}
```

### Integration with ML Libraries

Use with TensorFlow.js for neural network policies:

```javascript
import * as tf from '@tensorflow/tfjs';

// Preprocess observation
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

// DQN-style action selection
async function selectAction(model, ale, epsilon) {
    if (Math.random() < epsilon) {
        // Explore
        const actions = ale.getMinimalActionSet();
        return actions[Math.floor(Math.random() * actions.length)];
    } else {
        // Exploit
        const state = preprocessScreen(ale);
        const qValues = await model.predict(state);
        const action = qValues.argMax(-1).dataSync()[0];
        state.dispose();
        qValues.dispose();
        return action;
    }
}
```

### State Serialization for Checkpointing

```javascript
// Save checkpoint
function saveCheckpoint(episode, totalReward) {
    const checkpoint = {
        episode: episode,
        totalReward: totalReward,
        state: ale.saveState(),
        timestamp: Date.now()
    };

    localStorage.setItem('ale_checkpoint', JSON.stringify(checkpoint));
}

// Load checkpoint
function loadCheckpoint() {
    const data = localStorage.getItem('ale_checkpoint');
    if (!data) return null;

    const checkpoint = JSON.parse(data);
    ale.loadState(checkpoint.state);

    return checkpoint;
}
```

## Future Enhancements

Potential improvements for future versions:

- [ ] **Threading Support** - Use SharedArrayBuffer for vectorized environments
- [ ] **SIMD Optimization** - Leverage WebAssembly SIMD for faster screen processing
- [ ] **Streaming Compilation** - Faster module initialization
- [ ] **NPM Package** - Published package for easy integration
- [ ] **TypeScript Definitions** - Complete type definitions
- [ ] **React Components** - Pre-built React components
- [ ] **Service Worker Caching** - Offline support
- [ ] **Audio Support** - Enable sound in browser

## Resources

- **Build Instructions:** [WASM_BUILD_INSTRUCTIONS.md](../WASM_BUILD_INSTRUCTIONS.md)
- **Implementation Plan:** [WASM_IMPLEMENTATION_PLAN.md](../WASM_IMPLEMENTATION_PLAN.md)
- **Examples:** [docs/wasm/examples/](wasm-examples/examples/)
- **API Documentation:** [docs/wasm/README.md](wasm/README.md)

**External Resources:**
- [Emscripten Documentation](https://emscripten.org/)
- [WebAssembly Documentation](https://webassembly.org/)
- [Embind Documentation](https://emscripten.org/docs/porting/connecting_cpp_and_javascript/embind.html)

## Contributing

Contributions are welcome! Areas that need help:

- Additional example applications
- Performance optimizations
- Browser compatibility testing
- Documentation improvements
- Bug fixes

Please open issues or pull requests on GitHub.

## License

WASM support is provided under the same GPL-2.0 license as ALE.
