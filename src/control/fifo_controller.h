/* *****************************************************************************
 * A.L.E (Arcade Learning Environment)
 * Copyright (c) 2009-2012 by Yavar Naddaf, Joel Veness, Marc G. Bellemare and 
 *   the Reinforcement Learning and Artificial Intelligence Laboratory
 * Released under the GNU General Public License; see License.txt for details. 
 *
 * Based on: Stella  --  "An Atari 2600 VCS Emulator"
 * Copyright (c) 1995-2007 by Bradford W. Mott and the Stella team
 *
 * *****************************************************************************
 *  fifo_controller.h
 *
 *  The implementation of the FIFOController class, which is a subclass of
 * GameController, and is resposible for sending the Screen/RAM/RL content to
 * whatever external program we are using through FIFO pipes, and apply the
 * actions that are sent back
 **************************************************************************** */

#ifndef __FIFO_CONTROLLER_H__
#define __FIFO_CONTROLLER_H__

#include "../common/Constants.h"
#include "game_controller.h"

class RomSettings;

class FIFOController : public GameController {

    public:

        FIFOController(OSystem* _osystem, bool named_pipes = false);
        virtual ~FIFOController();


        // This is called on every iteration of the main loop. It is responsible
        // passing the framebuffer and the RAM content to whatever AI module we
        // are using, and applying the returned actions.
        virtual void update();

        virtual bool has_terminated();

    protected:

        // Returns whether we have reached the maximum number of frames for this run 
        bool hasMaxFrames();

        void phosphorBlend();
        void makeAveragePalette();
        uInt8 getPhosphor(uInt8 v1, uInt8 v2);
        uInt32 makeRGB(uInt8 r, uInt8 g, uInt8 b);
        /** Converts a RGB value to an 8-bit format */
        uInt8 rgbToNTSC(uInt32 rgb);
    
    protected:
        uInt8 rgb_ntsc[64][64][64];

        uInt32* pi_old_frame_buffer;   // Copy of frame buffer. Used to detect and
                                    // only send the changed pixels
        uInt32* pi_curr_frame_buffer;
        
        uInt32 my_avg_palette[256][256];
        uInt8 phosphor_blend_ratio;

        int i_max_num_frames_per_episode;
        int i_max_num_frames;
        int i_current_frame_number;

        bool b_run_length_encoding;
        bool b_disable_color_averaging;

        FILE* p_fout;               // Output Pipe
        FILE* p_fin;                // Input Pipe
};

#endif  // __FIFO_CONTROLLER_H__


