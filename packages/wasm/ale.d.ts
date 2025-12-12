export interface ALEInterface {
  setBool(key: string, value: boolean): void;
  getBool(key: string): boolean;
  setInt(key: string, value: number): void;
  getInt(key: string): number;
  setFloat(key: string, value: number): void;
  getFloat(key: string): number;
  setString(key: string, value: string): void;
  getString(key: string): string;
  loadROM(path: string): void;
  act(action: number): number;
  resetGame(): void;
  gameOver(): boolean;
  lives(): number;
  getScreenWidth(): number;
  getScreenHeight(): number;
  getScreenRGB(): Uint8ClampedArray;
  getScreenGrayscale(): Uint8ClampedArray;
  getRAM(): Uint8Array;
  getLegalActionSet(): number[];
  getMinimalActionSet(): number[];
  getEpisodeFrameNumber(): number;
  loadROMFromURL(url: string, filename?: string): Promise<void>;
  loadROMFromFile(file: File): Promise<void>;
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
