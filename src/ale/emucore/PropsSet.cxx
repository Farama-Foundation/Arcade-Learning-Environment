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
// $Id: PropsSet.cxx,v 1.34 2007/07/31 15:46:20 stephena Exp $
//============================================================================

#include <string>
#include <sstream>
#include <cstring>
#include <iostream>

#include "ale/emucore/DefProps.hxx"
#include "ale/emucore/Props.hxx"
#include "ale/emucore/PropsSet.hxx"

namespace ale {
namespace stella {

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
PropertiesSet::PropertiesSet()
  : myRoot(NULL),
    mySize(0)
{
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
PropertiesSet::~PropertiesSet()
{
  deleteNode(myRoot);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void PropertiesSet::getMD5(const std::string& md5, Properties& properties,
                           bool useDefaults) const
{
  properties.setDefaults();
  bool found = false;

  // First check our dynamic BST for the object
  if(!useDefaults && myRoot != 0)
  {
    TreeNode* current = myRoot;
    while(current)
    {
      const std::string& currentMd5 = current->props->get(Cartridge_MD5);
      if(currentMd5 == md5)
      {
        // We only report a node as found if it's been marked as valid.
        // Invalid nodes are those that should be removed, and are
        // essentially treated as if they're not present.
        found = current->valid;
        break;
      }
      else if(md5 < currentMd5)
        current = current->left;
      else
        current = current->right;
    }

    if(found)
      properties = *(current->props);
  }

  // Otherwise, search the internal database using binary search
  if(!found)
  {
    int low = 0, high = DEF_PROPS_SIZE - 1;
    while(low <= high)
    {
      int i = (low + high) / 2;
      int cmp = std::strncmp(md5.c_str(), DefProps[i][Cartridge_MD5], 32);

      if(cmp == 0)  // found it
      {
        for(int p = 0; p < LastPropType; ++p)
          if(DefProps[i][p][0] != 0)
            properties.set((PropertyType)p, DefProps[i][p]);

        found = true;
        break;
      }
      else if(cmp < 0)
        high = i - 1; // look at lower range
      else
        low = i + 1;  // look at upper range
    }
  }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void PropertiesSet::insert(const Properties& properties, bool save)
{
  insertNode(myRoot, properties, save);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void PropertiesSet::removeMD5(const std::string& md5)
{
  // We only remove from the dynamic BST
  if(myRoot != 0)
  {
    TreeNode* current = myRoot;
    while(current)
    {
      const std::string& currentMd5 = current->props->get(Cartridge_MD5);
      if(currentMd5 == md5)
      {
        current->valid = false;  // Essentially, this node doesn't exist
        break;
      }
      else if(md5 < currentMd5)
        current = current->left;
      else
        current = current->right;
    }
  }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void PropertiesSet::insertNode(TreeNode* &t, const Properties& properties,
                               bool save)
{
  if(t)
  {
    std::string md5 = properties.get(Cartridge_MD5);
    std::string currentMd5 = t->props->get(Cartridge_MD5);

    if(md5 < currentMd5)
      insertNode(t->left, properties, save);
    else if(md5 > currentMd5)
      insertNode(t->right, properties, save);
    else
    {
      delete t->props;
      t->props = new Properties(properties);
      t->save = save;
      t->valid = true;
    }
  }
  else
  {
    t = new TreeNode;
    t->props = new Properties(properties);
    t->left = 0;
    t->right = 0;
    t->save = save;
    t->valid = true;

    ++mySize;
  }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void PropertiesSet::deleteNode(TreeNode *node)
{
  if(node)
  {
    deleteNode(node->left);
    deleteNode(node->right);
    delete node->props;
    delete node;
  }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void PropertiesSet::print() const
{
  std::cerr << size() << std::endl;
  printNode(myRoot);  // FIXME - print out internal properties as well
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void PropertiesSet::printNode(TreeNode *node) const
{
  if(node)
  {
    if(node->valid && node->save)
      node->props->print();
    printNode(node->left);
    printNode(node->right);
  }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
uint32_t PropertiesSet::size() const
{
  return mySize;
}

}  // namespace stella
}  // namespace ale
