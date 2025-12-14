// Pre-run script for ALE WASM
// This runs before the WASM module is initialized

var Module = Module || {};

Module.preRun = Module.preRun || [];

Module.preRun.push(function() {
    // If ROMs are provided via Module.roms, load them into virtual filesystem
    // Note: Don't create /roms directory here if using --preload-file,
    // as Emscripten will automatically create it when mounting ale.data
    if (Module.FS && Module.roms) {
        // Create directory only if we have custom ROMs to load
        try {
            Module.FS.mkdir('/roms');
        } catch (e) {
            // Directory might already exist from preloaded data
        }

        Object.keys(Module.roms).forEach(function(romName) {
            var romData = Module.roms[romName];
            if (romData instanceof ArrayBuffer) {
                romData = new Uint8Array(romData);
            }
            try {
                Module.FS.writeFile('/roms/' + romName, romData);
                console.log('[ALE] Loaded ROM: ' + romName);
            } catch (e) {
                console.error('[ALE] Failed to load ROM ' + romName + ':', e);
            }
        });
    }
});

// Canvas configuration
if (typeof Module.canvas === 'undefined' && typeof document !== 'undefined') {
    // Try to find a canvas with id 'canvas' by default
    Module.canvas = document.getElementById('canvas');
}

// Print function for debugging
Module.print = Module.print || function(text) {
    console.log('[ALE]', text);
};

Module.printErr = Module.printErr || function(text) {
    console.error('[ALE Error]', text);
};
