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

#include <sstream>
#include <string.h>

#include "OSystem.hxx"
#include "DefProps.hxx"
#include "Props.hxx"
#include "PropsSet.hxx"
#include "bspf.hxx"
using namespace std;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
PropertiesSet::PropertiesSet(OSystem* osystem)
  : myOSystem(osystem),
    myRoot(NULL),
    mySize(0)
{
  const string& props = myOSystem->propertiesFile();
  load(props, true);    // do save these properties

  if(myOSystem->settings().getBool("showinfo"))
    cerr << "User game properties: \'" << props << "\'\n";
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
PropertiesSet::~PropertiesSet()
{
  deleteNode(myRoot);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void PropertiesSet::getMD5(const string& md5, Properties& properties,
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
      const string& currentMd5 = current->props->get(Cartridge_MD5);
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
      int cmp = strncmp(md5.c_str(), DefProps[i][Cartridge_MD5], 32);

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
void PropertiesSet::removeMD5(const string& md5)
{
  // We only remove from the dynamic BST
  if(myRoot != 0)
  {
    TreeNode* current = myRoot;
    while(current)
    {
      const string& currentMd5 = current->props->get(Cartridge_MD5);
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
    string md5 = properties.get(Cartridge_MD5);
    string currentMd5 = t->props->get(Cartridge_MD5);

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
void PropertiesSet::load(const string& filename, bool save)
{
  ifstream in(filename.c_str(), ios::in);

  // Loop reading properties
  for(;;)
  {
    // Make sure the stream is still good or we're done 
    if(!in)
      break;

    // Get the property list associated with this profile
    Properties prop;
    prop.load(in);

    // If the stream is still good then insert the properties
    if(in)
      insert(prop, save);
  }
  if(in)
    in.close();
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
bool PropertiesSet::save(const string& filename) const
{
  ofstream out(filename.c_str(), ios::out);
  if(!out)
    return false;

  saveNode(out, myRoot);
  out.close();
  return true;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void PropertiesSet::print() const
{
  cerr << size() << endl;
  printNode(myRoot);  // FIXME - print out internal properties as well
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void PropertiesSet::saveNode(ostream& out, TreeNode *node) const
{
  if(node)
  {
    if(node->valid && node->save)
      node->props->save(out);
    saveNode(out, node->left);
    saveNode(out, node->right);
  }
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
uInt32 PropertiesSet::size() const
{
  return mySize;
}
