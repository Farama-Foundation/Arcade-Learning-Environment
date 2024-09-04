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
// $Id: Cart.cxx,v 1.34 2007/06/14 13:47:50 stephena Exp $
//============================================================================

#include <string>
#include <cstring>

#include <cassert>
#include <sstream>

#include "ale/emucore/Cart.hxx"
#include "ale/emucore/Cart0840.hxx"
#include "ale/emucore/Cart2K.hxx"
#include "ale/emucore/Cart3E.hxx"
#include "ale/emucore/Cart3F.hxx"
#include "ale/emucore/Cart4A50.hxx"
#include "ale/emucore/Cart4K.hxx"
#include "ale/emucore/CartAR.hxx"
#include "ale/emucore/CartDPC.hxx"
#include "ale/emucore/CartE0.hxx"
#include "ale/emucore/CartE7.hxx"
#include "ale/emucore/CartF4.hxx"
#include "ale/emucore/CartF4SC.hxx"
#include "ale/emucore/CartF6.hxx"
#include "ale/emucore/CartF6SC.hxx"
#include "ale/emucore/CartF8.hxx"
#include "ale/emucore/CartF8SC.hxx"
#include "ale/emucore/CartFASC.hxx"
#include "ale/emucore/CartFE.hxx"
#include "ale/emucore/CartMC.hxx"
#include "ale/emucore/CartMB.hxx"
#include "ale/emucore/CartCV.hxx"
#include "ale/emucore/CartUA.hxx"
#include "ale/emucore/MD5.hxx"
#include "ale/emucore/Props.hxx"
#include "ale/emucore/Settings.hxx"

namespace ale {
namespace stella {

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Cartridge* Cartridge::create(const uint8_t* image, uint32_t size,
    const Properties& properties, const Settings& settings)
{
  Cartridge* cartridge = nullptr;

  // Get the type of the cartridge we're creating
  const std::string& md5 = properties.get(Cartridge_MD5);
  std::string type = properties.get(Cartridge_Type);

  // First consider the ROMs that are special and don't have a properties entry
  // Hopefully this list will be very small
  if(md5 == "bc24440b59092559a1ec26055fd1270e" ||
     md5 == "75ee371ccfc4f43e7d9b8f24e1266b55")
  {
    // These two ROMs are normal 8K images, except they must be initialized
    // from the opposite bank compared to normal ones
    type = "F8 swapped";
  }

  // Collect some info about the ROM
  std::ostringstream buf;
  buf << "  ROM Size:        " << size << std::endl
      << "  Bankswitch Type: " << type;

  // See if we should try to auto-detect the cartridge type
  // If we ask for extended info, always do an autodetect
  if(type == "AUTO-DETECT" || settings.getBool("rominfo"))
  {
    std::string detected = autodetectType(image, size);
    buf << " ==> " << detected;
    if(type != "AUTO-DETECT" && type != detected)
      buf << " (auto-detection not consistent)";

    type = detected;
  }
  buf << std::endl;

  // We should know the cart's type by now so let's create it
  if(type == "2K")
    cartridge = new Cartridge2K(image);
  else if(type == "3E")
    cartridge = new Cartridge3E(image, size);
  else if(type == "3F")
    cartridge = new Cartridge3F(image, size);
  else if(type == "4A50")
    cartridge = new Cartridge4A50(image);
  else if(type == "4K")
    cartridge = new Cartridge4K(image);
  else if(type == "AR")
    cartridge = new CartridgeAR(image, size, true);
  else if(type == "DPC")
    cartridge = new CartridgeDPC(image, size);
  else if(type == "E0")
    cartridge = new CartridgeE0(image);
  else if(type == "E7")
    cartridge = new CartridgeE7(image);
  else if(type == "F4")
    cartridge = new CartridgeF4(image);
  else if(type == "F4SC")
    cartridge = new CartridgeF4SC(image);
  else if(type == "F6")
    cartridge = new CartridgeF6(image);
  else if(type == "F6SC")
    cartridge = new CartridgeF6SC(image);
  else if(type == "F8")
    cartridge = new CartridgeF8(image, false);
  else if(type == "F8 swapped")
    cartridge = new CartridgeF8(image, true);
  else if(type == "F8SC")
    cartridge = new CartridgeF8SC(image);
  else if(type == "FASC")
    cartridge = new CartridgeFASC(image);
  else if(type == "FE")
    cartridge = new CartridgeFE(image);
  else if(type == "MC")
    cartridge = new CartridgeMC(image, size);
  else if(type == "MB")
    cartridge = new CartridgeMB(image);
  else if(type == "CV")
    cartridge = new CartridgeCV(image, size);
  else if(type == "UA")
    cartridge = new CartridgeUA(image);
  else if(type == "0840")
    cartridge = new Cartridge0840(image);
  else
    ale::Logger::Error << "ERROR: Invalid cartridge type " << type << " ..." << std::endl;

  if (cartridge != nullptr)
    cartridge->myAboutString = buf.str();

  return cartridge;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Cartridge::Cartridge()
{
  unlockBank();
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Cartridge::~Cartridge()
{
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
bool Cartridge::save(std::ofstream& out)
{
  int size = -1;

  uint8_t* image = getImage(size);
  if(image == 0 || size <= 0)
  {
    ale::Logger::Error << "save not supported" << std::endl;
    return false;
  }

  for(int i=0; i<size; i++)
    out << image[i];

  return true;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
std::string Cartridge::autodetectType(const uint8_t* image, uint32_t size)
{
  // Guess type based on size
  const char* type = 0;

  if(size != 0 && size % 8448 == 0)
  {
    type = "AR";
  }
  else if((size == 2048) ||
          (size == 4096 && memcmp(image, image + 2048, 2048) == 0))
  {
    if(isProbablyCV(image, size))
      type = "CV";
    else
      type = "2K";
  }
  else if(size == 4096)
  {
    if(isProbablyCV(image, size))
      type = "CV";
    else
      type = "4K";
  }
  else if(size == 8192)  // 8K
  {
    if(isProbablySC(image, size))
      type = "F8SC";
    else if(memcmp(image, image + 4096, 4096) == 0)
      type = "4K";
    else if(isProbablyE0(image, size))
      type = "E0";
    else if(isProbably3E(image, size))
      type = "3E";
    else if(isProbably3F(image, size))
      type = "3F";
    else if(isProbablyUA(image, size))
      type = "UA";
    else if(isProbablyFE(image, size))
      type = "FE";
    else
      type = "F8";
  }
  else if((size == 10495) || (size == 10496) || (size == 10240))  // 10K - Pitfall2
  {
    type = "DPC";
  }
  else if(size == 12288)  // 12K
  {
    // TODO - this should really be in a method that checks the first
    // 512 bytes of ROM and finds if either the lower 256 bytes or
    // higher 256 bytes are all the same.  For now, we assume that
    // all carts of 12K are CBS RAM Plus/FASC.
    type = "FASC";
  }
  else if(size == 16384)  // 16K
  {
    if(isProbablySC(image, size))
      type = "F6SC";
    else if(isProbablyE7(image, size))
      type = "E7";
    else if(isProbably3E(image, size))
      type = "3E";
    else if(isProbably3F(image, size))
      type = "3F";
    else
      type = "F6";
  }
  else if(size == 32768)  // 32K
  {
    if(isProbablySC(image, size))
      type = "F4SC";
    else if(isProbably3E(image, size))
      type = "3E";
    else if(isProbably3F(image, size))
      type = "3F";
    else
      type = "F4";
  }
  else if(size == 65536)  // 64K
  {
    // TODO - autodetect 4A50
    if(isProbably3E(image, size))
      type = "3E";
    else if(isProbably3F(image, size))
      type = "3F";
    else
      type = "MB";
  }
  else if(size == 131072)  // 128K
  {
    // TODO - autodetect 4A50
    if(isProbably3E(image, size))
      type = "3E";
    else if(isProbably3F(image, size))
      type = "3F";
    else
      type = "MC";
  }
  else  // what else can we do?
  {
    if(isProbably3E(image, size))
      type = "3E";
    else if(isProbably3F(image, size))
      type = "3F";
    else
      return "Unrecognized cartridge; ROM size was " + std::to_string(size);
  }

  return type;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
bool Cartridge::searchForBytes(const uint8_t* image, uint32_t imagesize,
                               const uint8_t* signature, uint32_t sigsize,
                               uint32_t minhits)
{
  if (sigsize > imagesize) return false;

  uint32_t count = 0;
  for(uint32_t i = 0; i < imagesize - sigsize; ++i)
  {
    uint32_t matches = 0;
    for(uint32_t j = 0; j < sigsize; ++j)
    {
      if(image[i+j] == signature[j])
        ++matches;
      else
        break;
    }
    if(matches == sigsize)
    {
      ++count;
      i += sigsize;  // skip past this signature 'window' entirely
    }
    if(count >= minhits)
      break;
  }

  return (count >= minhits);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
bool Cartridge::isProbablySC(const uint8_t* image, uint32_t size)
{
  // We assume a Superchip cart contains the same bytes for its entire
  // RAM area; obviously this test will fail if it doesn't
  // The RAM area will be the first 256 bytes of each 4K bank
  uint32_t banks = size / 4096;
  for(uint32_t i = 0; i < banks; ++i)
  {
    uint8_t first = image[i*4096];
    for(uint32_t j = 0; j < 256; ++j)
    {
      if(image[i*4096+j] != first)
        return false;
    }
  }
  return true;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
bool Cartridge::isProbably3F(const uint8_t* image, uint32_t size)
{
  // 3F cart bankswitching is triggered by storing the bank number
  // in address 3F using 'STA $3F'
  // We expect it will be present at least 2 times, since there are
  // at least two banks
  uint8_t signature[] = { 0x85, 0x3F };  // STA $3F
  return searchForBytes(image, size, signature, 2, 2);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
bool Cartridge::isProbably3E(const uint8_t* image, uint32_t size)
{
  // 3E cart bankswitching is triggered by storing the bank number
  // in address 3E using 'STA $3E', commonly followed by an
  // immediate mode LDA
  uint8_t signature[] = { 0x85, 0x3E, 0xA9, 0x00 };  // STA $3E; LDA #$00
  return searchForBytes(image, size, signature, 4, 1);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
bool Cartridge::isProbablyE0(const uint8_t* image, uint32_t size)
{
  // E0 cart bankswitching is triggered by accessing addresses
  // $FE0 to $FF9 using absolute non-indexed addressing
  // To eliminate false positives (and speed up processing), we
  // search for only certain known signatures
  // Thanks to "stella@casperkitty.com" for this advice
  // These signatures are attributed to the MESS project
  uint8_t signature[6][3] = {
   { 0x8D, 0xE0, 0x1F },  // STA $1FE0
   { 0x8D, 0xE0, 0x5F },  // STA $5FE0
   { 0x8D, 0xE9, 0xFF },  // STA $FFE9
   { 0xAD, 0xE9, 0xFF },  // LDA $FFE9
   { 0xAD, 0xED, 0xFF },  // LDA $FFED
   { 0xAD, 0xF3, 0xBF }   // LDA $BFF3
  };
  for(uint32_t i = 0; i < 6; ++i)
  {
    if(searchForBytes(image, size, signature[i], 3, 1))
      return true;
  }
  return false;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
bool Cartridge::isProbablyE7(const uint8_t* image, uint32_t size)
{
  // E7 carts map their second 1K block of RAM at addresses
  // $800 to $8FF.  However, since this occurs in the upper 2K address
  // space, and the last 2K in the cart always points to the last 2K of the
  // ROM image, the RAM area should fall in addresses $3800 to $38FF
  // Similar to the Superchip cart, we assume this RAM block contains
  // the same bytes for its entire area
  // Also, we want to distinguish between ROMs that have large blocks
  // of the same amount of (probably unused) data by making sure that
  // something differs in the previous 32 or next 32 bytes
  uint8_t first = image[0x3800];
  for(uint32_t i = 0x3800; i < 0x3A00; ++i)
  {
    if(first != image[i])
      return false;
  }

  // OK, now scan the surrounding 32 byte blocks
  uint32_t count1 = 0, count2 = 0;
  for(uint32_t i = 0x3800 - 32; i < 0x3800; ++i)
  {
    if(first != image[i])
      ++count1;
  }
  for(uint32_t i = 0x3A00; i < 0x3A00 + 32; ++i)
  {
    if(first != image[i])
      ++count2;
  }

  return (count1 > 0 || count2 > 0);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
bool Cartridge::isProbablyUA(const uint8_t* image, uint32_t size)
{
  // UA cart bankswitching switches to bank 1 by accessing address 0x240
  // using 'STA $240'
  uint8_t signature[] = { 0x8D, 0x40, 0x02 };  // STA $240
  return searchForBytes(image, size, signature, 3, 1);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
bool Cartridge::isProbablyCV(const uint8_t* image, uint32_t size)
{
  // CV RAM access occurs at addresses $f3ff and $f400
  // These signatures are attributed to the MESS project
  uint8_t signature[2][3] = {
    { 0x9D, 0xFF, 0xF3 },  // STA $F3FF
    { 0x99, 0x00, 0xF4 }   // STA $F400
  };
  if(searchForBytes(image, size, signature[0], 3, 1))
    return true;
  else
    return searchForBytes(image, size, signature[1], 3, 1);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
bool Cartridge::isProbablyFE(const uint8_t* image, uint32_t size)
{
  // FE bankswitching is very weird, but always seems to include a
  // 'JSR $xxxx'
  // These signatures are attributed to the MESS project
  uint8_t signature[4][5] = {
    { 0x20, 0x00, 0xD0, 0xC6, 0xC5 },  // JSR $D000; DEC $C5
    { 0x20, 0xC3, 0xF8, 0xA5, 0x82 },  // JSR $F8C3; LDA $82
    { 0xD0, 0xFB, 0x20, 0x73, 0xFE },  // BNE $FB; JSR $FE73
    { 0x20, 0x00, 0xF0, 0x84, 0xD6 }   // JSR $F000; STY $D6
  };
  for(uint32_t i = 0; i < 4; ++i)
  {
    if(searchForBytes(image, size, signature[i], 5, 1))
      return true;
  }
  return false;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Cartridge::Cartridge(const Cartridge&)
{
  assert(false);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Cartridge& Cartridge::operator = (const Cartridge&)
{
  assert(false);
  return *this;
}

}  // namespace stella
}  // namespace ale
