// Post-run script for ALE WASM
// This runs after the WASM module is initialized and bindings are ready

// Helper method to fetch and load ROMs from URLs
if (Module.ALEInterface) {
    Module.ALEInterface.prototype.loadROMFromURL = async function(url, filename) {
        // Fetch ROM data
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error('Failed to fetch ROM from ' + url + ': ' + response.statusText);
        }

        const arrayBuffer = await response.arrayBuffer();
        const data = new Uint8Array(arrayBuffer);

        // Ensure /roms directory exists
        try {
            Module.FS.mkdir('/roms');
        } catch (e) {
            // Directory might already exist
        }

        // Write ROM to virtual filesystem
        const romPath = '/roms/' + (filename || url.split('/').pop());
        Module.FS.writeFile(romPath, data);

        // Load the ROM
        this.loadROM(romPath);

        return romPath;
    };

    // Helper to load ROM from File API (user upload)
    Module.ALEInterface.prototype.loadROMFromFile = function(file) {
        const self = this;
        return new Promise((resolve, reject) => {
            const reader = new FileReader();

            reader.onload = (e) => {
                const data = new Uint8Array(e.target.result);

                try {
                    Module.FS.mkdir('/roms');
                } catch (err) {
                    // Ignore if exists
                }

                const romPath = '/roms/' + file.name;
                Module.FS.writeFile(romPath, data);
                self.loadROM(romPath);
                resolve(romPath);
            };

            reader.onerror = () => reject(reader.error);
            reader.readAsArrayBuffer(file);
        });
    };

    // Helper to get screen as ImageData for direct canvas rendering
    Module.ALEInterface.prototype.getScreenImageData = function() {
        const width = this.getScreenWidth();
        const height = this.getScreenHeight();
        const rgb = this.getScreenRGB();

        // Convert RGB to RGBA (RGB data is in interleaved format: RGBRGBRGB...)
        const rgba = new Uint8ClampedArray(width * height * 4);
        for (let i = 0; i < width * height; i++) {
            rgba[i * 4] = rgb[i * 3];         // R
            rgba[i * 4 + 1] = rgb[i * 3 + 1]; // G
            rgba[i * 4 + 2] = rgb[i * 3 + 2]; // B
            rgba[i * 4 + 3] = 255;            // A
        }

        return new ImageData(rgba, width, height);
    };

    // Helper to render directly to a canvas
    Module.ALEInterface.prototype.renderToCanvas = function(canvas) {
        const ctx = canvas.getContext('2d');
        const imageData = this.getScreenImageData();

        // Create temporary canvas at native resolution
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = imageData.width;
        tempCanvas.height = imageData.height;
        const tempCtx = tempCanvas.getContext('2d');
        tempCtx.putImageData(imageData, 0, 0);

        // Scale to target canvas
        ctx.imageSmoothingEnabled = false;  // Pixelated scaling for retro look
        ctx.drawImage(tempCanvas, 0, 0, canvas.width, canvas.height);
    };

    console.log('[ALE] ALE Interface ready. Version: ' + Module.ALEInterface.getVersion());
} else {
    console.warn('[ALE] ALEInterface not found. Bindings may not be loaded correctly.');
}
