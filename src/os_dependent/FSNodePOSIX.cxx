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
// $Id: FSNodePOSIX.cxx,v 1.11 2007/08/07 14:38:52 stephena Exp $
//
//   Based on code from ScummVM - Scumm Interpreter
//   Copyright (C) 2002-2004 The ScummVM project
//============================================================================

#include "FSNode.hxx"

#ifdef MACOSX
  #include <sys/types.h>
#endif

#include <sys/param.h>
#include <sys/stat.h>
#include <dirent.h>

#include <stdio.h>
#include <unistd.h>

#include <sstream>

/*
 * Implementation of the Stella file system API based on POSIX (for Linux and OSX)
 */

class POSIXFilesystemNode : public AbstractFilesystemNode
{
  public:
    POSIXFilesystemNode();
    POSIXFilesystemNode(const string& path);
    POSIXFilesystemNode(const POSIXFilesystemNode* node);

    virtual string displayName() const { return _displayName; }
    virtual bool isValid() const { return _isValid; }
    virtual bool isDirectory() const { return _isDirectory; }
    virtual string path() const { return _path; }

    virtual FSList listDir(ListMode mode = kListDirectoriesOnly) const;
    virtual AbstractFilesystemNode* parent() const;

  protected:
    string _displayName;
    bool _isDirectory;
    bool _isValid;
    string _path;
};

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
static const char* lastPathComponent(const string& str)
{
  const char *start = str.c_str();
  const char *cur = start + str.size() - 2;
    
  while (cur > start && *cur != '/')
    --cur;
    
  return cur+1;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
static string validatePath(const string& p)
{
  string path = p;
  if(p.size() <= 0 || p[0] != '/')
    path = "/";

  return path;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
AbstractFilesystemNode* FilesystemNode::getRoot()
{
  return new POSIXFilesystemNode();
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
AbstractFilesystemNode* FilesystemNode::getNodeForPath(const string& path)
{
  return new POSIXFilesystemNode(validatePath(path));
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
POSIXFilesystemNode::POSIXFilesystemNode()
{
  char buf[MAXPATHLEN];
  getcwd(buf, MAXPATHLEN);

  _path = buf;
  _displayName = lastPathComponent(_path);
  _path += '/';
  _isValid = true;
  _isDirectory = true;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
POSIXFilesystemNode::POSIXFilesystemNode(const string& p)
{
  string path = validatePath(p);

  Int32 len = 0, offset = path.size();
  struct stat st;

  _path = path;

  // Extract last component from path
  const char *str = path.c_str();
  while (offset > 0 && str[offset-1] == '/')
    offset--;
  while (offset > 0 && str[offset-1] != '/')
  {
    len++;
    offset--;
  }
  _displayName = string(str + offset, len);

  // Check whether it is a directory, and whether the file actually exists
  _isValid = (0 == stat(_path.c_str(), &st));
  _isDirectory = S_ISDIR(st.st_mode);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
POSIXFilesystemNode::POSIXFilesystemNode(const POSIXFilesystemNode* node)
{
  _displayName = node->_displayName;
  _isValid = node->_isValid;
  _isDirectory = node->_isDirectory;
  _path = node->_path;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
FSList POSIXFilesystemNode::listDir(ListMode mode) const
{
  DIR *dirp = opendir(_path.c_str());
  struct stat st;

  struct dirent *dp;
  FSList myList;

  if (dirp == NULL)
    return myList;

  // ... loop over dir entries using readdir
  while ((dp = readdir(dirp)) != NULL)
  {
    // Skip 'invisible' files
    if (dp->d_name[0] == '.')
      continue;

    POSIXFilesystemNode entry;
    entry._displayName = dp->d_name;
    entry._path = _path;
    if (entry._path.length() > 0 && entry._path[entry._path.length()-1] != '/')
      entry._path += '/';
    entry._path += dp->d_name;

    if (stat(entry._path.c_str(), &st))
      continue;
    entry._isDirectory = S_ISDIR(st.st_mode);

    // Honor the chosen mode
    if ((mode == kListFilesOnly && entry._isDirectory) ||
        (mode == kListDirectoriesOnly && !entry._isDirectory))
      continue;

    if (entry._isDirectory)
      entry._path += "/";

    myList.push_back(wrap(new POSIXFilesystemNode(&entry)));
  }
  closedir(dirp);

  return myList;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
AbstractFilesystemNode *POSIXFilesystemNode::parent() const
{
  if (_path == "/")
    return 0;

  POSIXFilesystemNode* p = new POSIXFilesystemNode();
  const char *start = _path.c_str();
  const char *end = lastPathComponent(_path);

  p->_path = string(start, end - start);
  p->_displayName = lastPathComponent(p->_path);

  p->_isValid = true;
  p->_isDirectory = true;

  return p;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
bool AbstractFilesystemNode::fileExists(const string& path)
{
  struct stat st;
  if(stat(path.c_str(), &st) != 0)
    return false;

  return S_ISREG(st.st_mode);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
bool AbstractFilesystemNode::dirExists(const string& path)
{
  struct stat st;
  if(stat(path.c_str(), &st) != 0)
    return false;

  return S_ISDIR(st.st_mode);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
bool AbstractFilesystemNode::makeDir(const string& path)
{
  return mkdir(path.c_str(), 0777) == 0;
}
