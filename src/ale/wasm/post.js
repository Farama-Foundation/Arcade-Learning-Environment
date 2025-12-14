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

    console.log('[ALE] ALE Interface ready. Version: ' + Module.ALEInterface.getVersion());
} else {
    console.warn('[ALE] ALEInterface not found. Bindings may not be loaded correctly.');
}
