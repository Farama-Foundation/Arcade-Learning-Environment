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
// $Id: Event.hxx,v 1.28 2007/01/30 17:13:07 stephena Exp $
//============================================================================

#ifndef EVENT_HXX
#define EVENT_HXX

namespace ale {
namespace stella {

class Event;

/**
  @author  Bradford W. Mott
  @version $Id: Event.hxx,v 1.28 2007/01/30 17:13:07 stephena Exp $
*/
class Event
{
  public:
    /**
      Enumeration of all possible events in Stella, including both
      console and controller event types as well as events that aren't
      technically part of the emulation core
    */
    enum Type
    {
      ConsoleColor, ConsoleBlackWhite,
      ConsoleLeftDifficultyA, ConsoleLeftDifficultyB,
      ConsoleRightDifficultyA, ConsoleRightDifficultyB,
      ConsoleSelect, ConsoleReset,

      JoystickZeroUp, JoystickZeroDown, JoystickZeroLeft,
      JoystickZeroRight, JoystickZeroFire,
      JoystickOneUp, JoystickOneDown, JoystickOneLeft,
      JoystickOneRight, JoystickOneFire,

      PaddleZeroResistance, PaddleZeroFire,
        PaddleZeroDecrease, PaddleZeroIncrease, PaddleZeroAnalog,
      PaddleOneResistance, PaddleOneFire,
        PaddleOneDecrease, PaddleOneIncrease, PaddleOneAnalog,
      PaddleTwoResistance, PaddleTwoFire,
        PaddleTwoDecrease, PaddleTwoIncrease, PaddleTwoAnalog,
      PaddleThreeResistance, PaddleThreeFire,
        PaddleThreeDecrease, PaddleThreeIncrease, PaddleThreeAnalog,

      LastType
    };

  public:
    /**
      Create a new event object and use the given eventstreamer
    */
    Event();

    /**
      Destructor
    */
    virtual ~Event();

  public:
    /**
      Get the value associated with the event of the specified type
    */
    virtual int get(Type type) const;

    /**
      Set the value associated with the event of the specified type
    */
    virtual void set(Type type, int value);

    /**
      Clears the event array (resets to initial state)
    */
    virtual void clear();

  protected:
    // Number of event types there are
    const int myNumberOfTypes;

    // Array of values associated with each event type
    int myValues[LastType];
};

}  // namespace stella
}  // namespace ale

#endif
