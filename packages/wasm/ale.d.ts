export interface ALEInterface {
  // Configuration
  setBool(key: string, value: boolean): void;
  getBool(key: string): boolean;
  setInt(key: string, value: number): void;
  getInt(key: string): number;
  setFloat(key: string, value: number): void;
  getFloat(key: string): number;
  setString(key: string, value: string): void;
  getString(key: string): string;

  // ROM Management
  loadROM(path: string): void;
  loadROMFromURL(url: string, filename?: string): Promise<string>;
  loadROMFromFile(file: File): Promise<string>;

  // Game Loop
  act(action: number): number;
  resetGame(): void;
  gameOver(): boolean;
  gameTruncated(): boolean;
  lives(): number;
  getFrameNumber(): number;
  getEpisodeFrameNumber(): number;

  // Observation
  getScreenRGB(): Uint8ClampedArray;
  getScreenGrayscale(): Uint8ClampedArray;
  getScreenImageData(): ImageData;
  renderToCanvas(canvasId: string): void;
  getScreenWidth(): number;
  getScreenHeight(): number;
  getRAM(): Uint8Array;
  setRAM(index: number, value: number): void;

  // Action Space
  getLegalActionSet(): number[];
  getMinimalActionSet(): number[];

  // Mode & Difficulty
  getAvailableModes(): number[];
  setMode(mode: number): void;
  getMode(): number;
  getAvailableDifficulties(): number[];
  setDifficulty(difficulty: number): void;
  getDifficulty(): number;

  // State Management
  saveState(): Uint8Array;
  loadState(state: Uint8Array): void;
}

export interface ALEInterfaceConstructor {
  new (): ALEInterface;
  getVersion(): string;
}

export interface ALEModule {
  ALEInterface: ALEInterfaceConstructor;
  FS: any;
}

declare function createALEModule(options?: object): Promise<ALEModule>;
export default createALEModule;
