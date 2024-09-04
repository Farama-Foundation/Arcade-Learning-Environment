//============================================================================
//
//   SSSS    tt          lll  lll
//  SS  SS   tt           ll   ll
//  SS     tttttt  eeee   ll   ll   aaaa
//   SSSS    tt   ee  ee  ll   ll      aa
//      SS   tt   eeeeee  ll   ll   aaaaa  --  "An Atari 2600 VCS Emulator"
//  SS  SS   tt   ee      ll   ll  aa  aa
//   SSSS     ttt  eeeee llll llll  aaaaa
//
// Copyright (c) 1995-2007 by Bradford W. Mott and the Stella team
//
// See the file "license" for information on usage and redistribution of
// this file, and for a DISCLAIMER OF ALL WARRANTIES.
//
// $Id: TIA.hxx,v 1.42 2007/02/22 02:15:46 stephena Exp $
//============================================================================

#ifndef TIA_HXX
#define TIA_HXX

namespace ale {
namespace stella {

class Console;
class System;
class Serializer;
class Deserializer;
class Settings;

}  // namespace stella
}  // namespace ale

#include "ale/emucore/Sound.hxx"
#include "ale/emucore/Device.hxx"
#include "ale/emucore/MediaSrc.hxx"

namespace ale {
namespace stella {

/**
  This class is a device that emulates the Television Interface Adapator
  found in the Atari 2600 and 7800 consoles.  The Television Interface
  Adapator is an integrated circuit designed to interface between an
  eight bit microprocessor and a television video modulator. It converts
  eight bit parallel data into serial outputs for the color, luminosity,
  and composite sync required by a video modulator.

  This class outputs the serial data into a frame buffer which can then
  be displayed on screen.

  @author  Bradford W. Mott
  @version $Id: TIA.hxx,v 1.42 2007/02/22 02:15:46 stephena Exp $
*/
class TIA : public Device , public MediaSource
{
  public:
    friend class TIADebug;

    /**
      Create a new TIA for the specified console

      @param console  The console the TIA is associated with
      @param settings The settings object for this TIA device
    */
    TIA(const Console& console, Settings& settings);

    /**
      Destructor
    */
    virtual ~TIA();

  public:
    /**
      Get a null terminated string which is the device's name (i.e. "M6532")

      @return The name of the device
    */
    virtual const char* name() const;

    /**
      Reset device to its power-on state
    */
    virtual void reset();

    /**
      Reset frame to change XStart/YStart/Width/Height properties
    */
    virtual void frameReset();

    /**
      Notification method invoked by the system right before the
      system resets its cycle counter to zero.  It may be necessary
      to override this method for devices that remember cycle counts.
    */
    virtual void systemCyclesReset();

    /**
      Install TIA in the specified system.  Invoked by the system
      when the TIA is attached to it.

      @param system The system the device should install itself in
    */
    virtual void install(System& system);

    /**
      Saves the current state of this device to the given Serializer.

      @param out The serializer device to save to.
      @return The result of the save.  True on success, false on failure.
    */
    virtual bool save(Serializer& out);

    /**
      Loads the current state of this device from the given Deserializer.

      @param in The deserializer device to load from.
      @return The result of the load.  True on success, false on failure.
    */
    virtual bool load(Deserializer& in);

  public:
    /**
      Get the byte at the specified address

      @return The byte at the specified address
    */
    virtual uint8_t peek(uint16_t address);

    /**
      Change the byte at the specified address to the given value

      @param address The address where the value should be stored
      @param value The value to be stored at the address
    */
    virtual void poke(uint16_t address, uint8_t value);

  public:
    /**
      This method should be called at an interval corresponding to
      the desired frame rate to update the media source.
    */
    virtual void update();

    /**
      Answers the current frame buffer

      @return Pointer to the current frame buffer
    */
    uint8_t* currentFrameBuffer() const { return myCurrentFrameBuffer; }

    /**
      Answers the previous frame buffer

      @return Pointer to the previous frame buffer
    */
    uint8_t* previousFrameBuffer() const { return myPreviousFrameBuffer; }

    /**
      Answers the height of the frame buffer

      @return The frame's height
    */
    uint32_t height() const;

    /**
      Answers the width of the frame buffer

      @return The frame's width
    */
    uint32_t width() const;

    /**
      Answers the total number of scanlines the media source generated
      in producing the current frame buffer. For partial frames, this
      will be the current scanline.

      @return The total number of scanlines generated
    */
    uint32_t scanlines() const;

    /**
      Answers the current color clock we've gotten to on this scanline.

      @return The current color clock
    */
    uint32_t clocksThisLine() const;

    /**
      Sets the sound device for the TIA.
    */
    void setSound(Sound& sound);

    enum TIABit {
      P0,   // Descriptor for Player 0 Bit
      P1,   // Descriptor for Player 1 Bit
      M0,   // Descriptor for Missle 0 Bit
      M1,   // Descriptor for Missle 1 Bit
      BL,   // Descriptor for Ball Bit
      PF    // Descriptor for Playfield Bit
    };

    /**
      Enables/disables the specified TIA bit.

      @return  Whether the bit was enabled or disabled
    */
    bool enableBit(TIABit b, bool mode) { myBitEnabled[b] = mode; return mode; }

    /**
      Toggles the specified TIA bit.

      @return  Whether the bit was enabled or disabled
    */
    bool toggleBit(TIABit b) { myBitEnabled[b] = !myBitEnabled[b]; return myBitEnabled[b]; }

    /**
      Enables/disables all TIABit bits.

      @param mode  Whether to enable or disable all bits
    */
    void enableBits(bool mode) { for(uint8_t i = 0; i < 6; ++i) myBitEnabled[i] = mode; }

  private:
    // Compute the ball mask table
    static void computeBallMaskTable();

    // Compute the collision decode table
    static void computeCollisionTable();

    // Compute the missle mask table
    static void computeMissleMaskTable();

    // Compute the player mask table
    static void computePlayerMaskTable();

    // Compute the player position reset when table
    static void computePlayerPositionResetWhenTable();

    // Compute the player reflect table
    static void computePlayerReflectTable();

    // Compute playfield mask table
    static void computePlayfieldMaskTable();

  private:
    // Update the current frame buffer up to one scanline
    void updateFrameScanline(uint32_t clocksToUpdate, uint32_t hpos);

    // Update the current frame buffer to the specified color clock
    void updateFrame(int clock);

    // Waste cycles until the current scanline is finished
    void waitHorizontalSync();

    // Grey out current framebuffer from current scanline to bottom
    void greyOutFrame();

    // Clear both internal TIA buffers to black (palette color 0)
    void clearBuffers();

    // Set up bookkeeping for the next frame
    void startFrame();

    // Update bookkeeping at end of frame
    void endFrame();

  private:
    // Console the TIA is associated with
    const Console& myConsole;

    // Settings object the TIA is associated with
    const Settings& mySettings;

    // Sound object the TIA is associated with
    Sound* mySound;

  private:
    // Indicates if color loss should be enabled or disabled.  Color loss
    // occurs on PAL (and maybe SECAM) systems when the previous frame
    // contains an odd number of scanlines.
    bool myColorLossEnabled;

    // Indicates whether we're done with the current frame. poke() clears this
    // when VSYNC is strobed or the max scanlines/frame limit is hit.
    bool myPartialFrameFlag;

  private:
    // Number of frames displayed by this TIA
    int myFrameCounter;

    // Pointer to the current frame buffer
    uint8_t* myCurrentFrameBuffer;

    // Pointer to the previous frame buffer
    uint8_t* myPreviousFrameBuffer;

    // Pointer to the next pixel that will be drawn in the current frame buffer
    uint8_t* myFramePointer;

    // Indicates where the scanline should start being displayed
    uint32_t myFrameXStart;

    // Indicates the width of the scanline
    uint32_t myFrameWidth;

    // Indicated what scanline the frame should start being drawn at
    uint32_t myFrameYStart;

    // Indicates the height of the frame in scanlines
    uint32_t myFrameHeight;

  private:
    // Indicates offset in scanlines when display should begin
     // (aka the Display.YStart property)
    uint32_t myYStart;

     // Height of display (aka Display.Height)
    uint32_t myHeight;

    // Indicates offset in color clocks when display should begin
    uint32_t myStartDisplayOffset;

    // Indicates offset in color clocks when display should stop
    uint32_t myStopDisplayOffset;

  private:
    // Indicates color clocks when the current frame began
    int myClockWhenFrameStarted;

    // Indicates color clocks when frame should begin to be drawn
    int myClockStartDisplay;

    // Indicates color clocks when frame should stop being drawn
    int myClockStopDisplay;

    // Indicates color clocks when the frame was last updated
    int myClockAtLastUpdate;

    // Indicates how many color clocks remain until the end of
    // current scanline.  This value is valid during the
    // displayed portion of the frame.
    int myClocksToEndOfScanLine;

    // Indicates the total number of scanlines generated by the last frame
    int myScanlineCountForLastFrame;

    // Indicates the current scanline during a partial frame.
    int myCurrentScanline;

    // Indicates the maximum number of scanlines to be generated for a frame
    int myMaximumNumberOfScanlines;

  private:
    // Color clock when VSYNC ending causes a new frame to be started
    int myVSYNCFinishClock;

  private:
    enum
    {
      myP0Bit = 0x01,         // Bit for Player 0
      myM0Bit = 0x02,         // Bit for Missle 0
      myP1Bit = 0x04,         // Bit for Player 1
      myM1Bit = 0x08,         // Bit for Missle 1
      myBLBit = 0x10,         // Bit for Ball
      myPFBit = 0x20,         // Bit for Playfield
      ScoreBit = 0x40,        // Bit for Playfield score mode
      PriorityBit = 0x080     // Bit for Playfield priority
    };

    // Bitmap of the objects that should be considered while drawing
    uint8_t myEnabledObjects;

  private:
    uint8_t myVSYNC;        // Holds the VSYNC register value
    uint8_t myVBLANK;       // Holds the VBLANK register value

    uint8_t myNUSIZ0;       // Number and size of player 0 and missle 0
    uint8_t myNUSIZ1;       // Number and size of player 1 and missle 1

    uint8_t myPlayfieldPriorityAndScore;
    uint32_t myColor[4];
    uint8_t myPriorityEncoder[2][256];

    uint32_t& myCOLUBK;       // Background color register (replicated 4 times)
    uint32_t& myCOLUPF;       // Playfield color register (replicated 4 times)
    uint32_t& myCOLUP0;       // Player 0 color register (replicated 4 times)
    uint32_t& myCOLUP1;       // Player 1 color register (replicated 4 times)

    uint8_t myCTRLPF;       // Playfield control register

    bool myREFP0;         // Indicates if player 0 is being reflected
    bool myREFP1;         // Indicates if player 1 is being reflected

    uint32_t myPF;          // Playfield graphics (19-12:PF2 11-4:PF1 3-0:PF0)

    uint8_t myGRP0;         // Player 0 graphics register
    uint8_t myGRP1;         // Player 1 graphics register

    uint8_t myDGRP0;        // Player 0 delayed graphics register
    uint8_t myDGRP1;        // Player 1 delayed graphics register

    bool myENAM0;         // Indicates if missle 0 is enabled
    bool myENAM1;         // Indicates if missle 0 is enabled

    bool myENABL;         // Indicates if the ball is enabled
    bool myDENABL;        // Indicates if the virtically delayed ball is enabled

    int8_t myHMP0;          // Player 0 horizontal motion register
    int8_t myHMP1;          // Player 1 horizontal motion register
    int8_t myHMM0;          // Missle 0 horizontal motion register
    int8_t myHMM1;          // Missle 1 horizontal motion register
    int8_t myHMBL;          // Ball horizontal motion register

    bool myVDELP0;        // Indicates if player 0 is being virtically delayed
    bool myVDELP1;        // Indicates if player 1 is being virtically delayed
    bool myVDELBL;        // Indicates if the ball is being virtically delayed

    bool myRESMP0;        // Indicates if missle 0 is reset to player 0
    bool myRESMP1;        // Indicates if missle 1 is reset to player 1

    uint16_t myCollision;    // Collision register

    // Note that these position registers contain the color clock
    // on which the object's serial output should begin (0 to 159)
    int16_t myPOSP0;         // Player 0 position register
    int16_t myPOSP1;         // Player 1 position register
    int16_t myPOSM0;         // Missle 0 position register
    int16_t myPOSM1;         // Missle 1 position register
    int16_t myPOSBL;         // Ball position register

  private:
    // Graphics for Player 0 that should be displayed.  This will be
    // reflected if the player is being reflected.
    uint8_t myCurrentGRP0;

    // Graphics for Player 1 that should be displayed.  This will be
    // reflected if the player is being reflected.
    uint8_t myCurrentGRP1;

    // It's VERY important that the BL, M0, M1, P0 and P1 current
    // mask pointers are always on a uint32_t boundary.  Otherwise,
    // the TIA code will fail on a good number of CPUs.

    // Pointer to the currently active mask array for the ball
    const uint8_t* myCurrentBLMask;

    // Pointer to the currently active mask array for missle 0
    const uint8_t* myCurrentM0Mask;

    // Pointer to the currently active mask array for missle 1
    const uint8_t* myCurrentM1Mask;

    // Pointer to the currently active mask array for player 0
    const uint8_t* myCurrentP0Mask;

    // Pointer to the currently active mask array for player 1
    const uint8_t* myCurrentP1Mask;

    // Pointer to the currently active mask array for the playfield
    const uint32_t* myCurrentPFMask;

    // Audio values. Only used by TIADebug.
    uint8_t myAUDV0;
    uint8_t myAUDV1;
    uint8_t myAUDC0;
    uint8_t myAUDC1;
    uint8_t myAUDF0;
    uint8_t myAUDF1;

  private:
    // Indicates when the dump for paddles was last set
    int myDumpDisabledCycle;

    // Indicates if the dump is current enabled for the paddles
    bool myDumpEnabled;

  private:
    // Color clock when last HMOVE occured
    int myLastHMOVEClock;

    // Indicates if HMOVE blanks are currently enabled
    bool myHMOVEBlankEnabled;

    // Indicates if we're allowing HMOVE blanks to be enabled
    bool myAllowHMOVEBlanks;

    // TIA M0 "bug" used for stars in Cosmic Ark flag
    bool myM0CosmicArkMotionEnabled;

    // Counter used for TIA M0 "bug"
    uint32_t myM0CosmicArkCounter;

    // Answers whether specified bits (from TIABit) are enabled or disabled
    bool myBitEnabled[6];

     // Has current frame been "greyed out" (has updateScanline() been run?)
     bool myFrameGreyed;

  private:
    // Ball mask table (entries are true or false)
    static uint8_t ourBallMaskTable[4][4][320];

    // Used to set the collision register to the correct value
    static uint16_t ourCollisionTable[64];

    // A mask table which can be used when an object is disabled
    static const uint8_t ourDisabledMaskTable[640];

    // Indicates the update delay associated with poking at a TIA address
    static const int16_t ourPokeDelayTable[64];

    // Missle mask table (entries are true or false)
    static uint8_t ourMissleMaskTable[4][8][4][320];

    // Used to convert value written in a motion register into
    // its internal representation
    static const int ourCompleteMotionTable[76][16];

    // Indicates if HMOVE blanks should occur for the corresponding cycle
    static const bool ourHMOVEBlankEnableCycles[76];

    // Player mask table
    static uint8_t ourPlayerMaskTable[4][2][8][320];

    // Indicates if player is being reset during delay, display or other times
    static int8_t ourPlayerPositionResetWhenTable[8][160][160];

    // Used to reflect a players graphics
    static uint8_t ourPlayerReflectTable[256];

    // Playfield mask table for reflected and non-reflected playfields
    static uint32_t ourPlayfieldTable[2][160];

  private:
    // Copy constructor isn't supported by this class so make it private
    TIA(const TIA&);

    // Assignment operator isn't supported by this class so make it private
    TIA& operator = (const TIA&);

  /** ALE-specific */
  private:
    bool fastUpdate;

    // Updates the frame's scanline but not the frame buffer
    void updateFrameScanlineFast(uint32_t clocksToUpdate, uint32_t hpos);

};

}  // namespace stella
}  // namespace ale

#endif
