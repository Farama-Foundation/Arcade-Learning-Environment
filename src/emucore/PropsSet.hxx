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
// $Id: PropsSet.hxx,v 1.18 2007/01/01 18:04:49 stephena Exp $
//============================================================================

#ifndef PROPERTIES_SET_HXX
#define PROPERTIES_SET_HXX

#include <fstream>

#include "m6502/src/bspf/src/bspf.hxx"

class OSystem;
class Properties;

/**
  This class maintains a sorted collection of properties.  The objects
  are maintained in a binary search tree sorted by md5, since this is
  the attribute most likely to be present in each entry in stella.pro
  and least likely to change.  A change in MD5 would mean a change in
  the game rom image (essentially a different game) and this would
  necessitate a new entry in the stella.pro file anyway.
  
  @author  Stephen Anthony
*/
class PropertiesSet
{
  public:
    /**
      Create an empty properties set object using the md5 as the
      key to the BST.
    */
    PropertiesSet(OSystem* osystem);

    /**
      Destructor
    */
    virtual ~PropertiesSet();

  public:
    /**
      Get the property from the set with the given MD5.

      @param md5         The md5 of the property to get
      @param properties  The property with the given MD5, or the default
                         properties if not found
      @param defaults    Use the built-in defaults, ignoring any external properties
    */
    void getMD5(const std::string& md5, Properties& properties,
                bool useDefaults = false) const;

    /** 
      Load properties from the specified file.  Use the given 
      defaults properties as the defaults for any properties loaded.

      @param filename  Full pathname of input file to use
      @param save      Indicates whether to set the 'save' tag for
                       these properties
    */
    void load(const std::string& filename, bool save);

    /**
      Save properties to the specified file.

      @param filename  Full pathname of output file to use

      @return  True on success, false on failure
               Failure occurs if file couldn't be opened for writing
    */
    bool save(const std::string& filename) const;

    /**
      Insert the properties into the set.  If a duplicate is inserted
      the old properties are overwritten with the new ones.

      @param properties  The collection of properties
      @param save        Indicates whether to set the 'save' tag for
                         this property
    */
    void insert(const Properties& properties, bool save);

    /**
      Marks the property with the given MD5 as being removed.

      @param md5  The md5 of the property to remove
    */
    void removeMD5(const std::string& md5);

    /**
      Get the number of properties in the collection.

      @return  The number of properties in the collection
    */
    uInt32 size() const;

    /**
      Prints the contents of the PropertiesSet as a flat file.
    */
    void print() const;

  private:
    struct TreeNode {
      Properties* props;
      TreeNode* left;
      TreeNode* right;
      bool save;
      bool valid;
    };

    /**
      Insert a node in the bst, keeping the tree sorted.

      @param node        The current subroot of the tree
      @param properties  The collection of properties
      @param save        Indicates whether to set the 'save' tag for
                         this property
    */
    void insertNode(TreeNode* &node, const Properties& properties, bool save);

    /**
      Deletes a node from the bst.  Does not preserve bst sorting.

      @param node  The current subroot of the tree
    */
    void deleteNode(TreeNode *node);

    /**
      Save current node properties to the specified output stream 

      @param out   The output stream to use
      @param node  The current subroot of the tree
    */
    void saveNode(std::ostream& out, TreeNode* node) const;

    /**
      Prints the current node properties

      @param node  The current subroot of the tree
    */
    void printNode(TreeNode* node) const;

  private:
    // The parent system for this object
    OSystem* myOSystem;

    // The root of the BST
    TreeNode* myRoot;

    // The size of the properties bst (i.e. the number of properties in it)
    uInt32 mySize;
};

#endif
