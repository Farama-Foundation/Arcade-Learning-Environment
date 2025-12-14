# @farama/ale-wasm

Arcade Learning Environment compiled to WebAssembly for browser-based Atari 2600 emulation.

## Installation

```bash
npm install @farama/ale-wasm
```

## Usage

```javascript
import createALEModule from '@farama/ale-wasm';

const ALE = await createALEModule();
const ale = new ALE.ALEInterface();

// Load ROM from URL
await ale.loadROMFromURL('https://example.com/breakout.bin');

// Or from file input
const fileInput = document.querySelector('input[type="file"]');
await ale.loadROMFromFile(fileInput.files[0]);

// Play
ale.resetGame();
while (!ale.gameOver()) {
  const actions = ale.getMinimalActionSet();
  const action = actions[Math.floor(Math.random() * actions.length)];
  const reward = ale.act(action);
}
```

## API

- `ale.loadROM(path)` - Load ROM from virtual filesystem
- `ale.loadROMFromURL(url)` - Load ROM from URL
- `ale.loadROMFromFile(file)` - Load ROM from File object
- `ale.act(action)` - Take action, returns reward
- `ale.resetGame()` - Reset game
- `ale.gameOver()` - Check if game is over
- `ale.getScreenRGB()` - Get screen as RGB array
- `ale.getMinimalActionSet()` - Get valid actions

## Documentation

https://github.com/Farama-Foundation/Arcade-Learning-Environment
