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
// $Id: FSNodeWin32.cxx,v 1.11 2007/08/07 14:38:52 stephena Exp $
//
//   Based on code from ScummVM - Scumm Interpreter
//   Copyright (C) 2002-2004 The ScummVM project
//============================================================================

#include <sstream>

#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include <tchar.h>
#include <direct.h>

#include "FSNode.hxx"

using namespace std;

/*
 * Implementation of the Stella file system API based on Windows API.
 */

class WindowsFilesystemNode : public AbstractFilesystemNode
{
  public:
    WindowsFilesystemNode();
    WindowsFilesystemNode(const string &path);
    WindowsFilesystemNode(const WindowsFilesystemNode* node);

    virtual string displayName() const { return _displayName; }
    virtual bool isValid() const { return _isValid; }
    virtual bool isDirectory() const { return _isDirectory; }
    virtual string path() const { return _path; }

    virtual FSList listDir(ListMode) const;
    virtual AbstractFilesystemNode* parent() const;

  protected:
    string _displayName;
    bool   _isDirectory;
    bool   _isValid;
    bool   _isPseudoRoot;
    string _path;

  private:
    static char* toAscii(TCHAR* x);
    static TCHAR* toUnicode(char* x);
    static void addFile (FSList& list, ListMode mode,
                         const char* base, WIN32_FIND_DATA* find_data);
};

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
static const char* lastPathComponent(const string& str)
{
  const char* start = str.c_str();
  const char* cur = start + str.size() - 2;

  while (cur > start && *cur != '\\')
    --cur;

  return cur + 1;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
static string validatePath(const string& p)
{
  string path = p;
  if(p.size() < 2 || p[1] != ':')
    path = "c:";

  return path;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
char* WindowsFilesystemNode::toAscii(TCHAR* x)
{
#ifndef UNICODE
  return (char*)x;
#else	
  static char asciiString[MAX_PATH];
  WideCharToMultiByte(CP_ACP, 0, x, _tcslen(x) + 1, asciiString, sizeof(asciiString), NULL, NULL);
  return asciiString;
#endif
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
TCHAR* WindowsFilesystemNode::toUnicode(char* x)
{
#ifndef UNICODE
  return (TCHAR*)x;
#else
  static TCHAR unicodeString[MAX_PATH];
  MultiByteToWideChar(CP_ACP, 0, x, strlen(x) + 1, unicodeString, sizeof(unicodeString));
  return unicodeString;
#endif
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void WindowsFilesystemNode::addFile(FSList& list, ListMode mode,
                                    const char* base, WIN32_FIND_DATA* find_data)
{
  WindowsFilesystemNode entry;
  char* asciiName = toAscii(find_data->cFileName);
  bool isDirectory;

  // Skip local directory (.) and parent (..)
  if (!strcmp(asciiName, ".") || !strcmp(asciiName, ".."))
    return;

  isDirectory = (find_data->dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY ? true : false);

  if ((!isDirectory && mode == kListDirectoriesOnly) ||
     (isDirectory && mode == kListFilesOnly))
    return;

  entry._isDirectory = isDirectory;
  entry._displayName = asciiName;
  entry._path = base;
  entry._path += asciiName;
  if (entry._isDirectory)
    entry._path += "\\";
  entry._isValid = true;
  entry._isPseudoRoot = false;

  list.push_back(wrap(new WindowsFilesystemNode(&entry)));
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
AbstractFilesystemNode* FilesystemNode::getRoot()
{
  return new WindowsFilesystemNode();
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
AbstractFilesystemNode* FilesystemNode::getNodeForPath(const string& path)
{
  return new WindowsFilesystemNode(path);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
WindowsFilesystemNode::WindowsFilesystemNode()
{
  _isDirectory = true;

  // Create a virtual root directory for standard Windows system
  _isValid = false;
  _path = "";
  _isPseudoRoot = true;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
WindowsFilesystemNode::WindowsFilesystemNode(const string& path)
{
  _path = validatePath(path);
  _displayName = lastPathComponent(_path);
  _isValid = true;
  _isDirectory = true;
  _isPseudoRoot = false;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
WindowsFilesystemNode::WindowsFilesystemNode(const WindowsFilesystemNode* node)
{
  _displayName = node->_displayName;
  _isDirectory = node->_isDirectory;
  _isValid = node->_isValid;
  _isPseudoRoot = node->_isPseudoRoot;
  _path = node->_path;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
FSList WindowsFilesystemNode::listDir(ListMode mode) const
{
  assert(_isDirectory);

  FSList myList;

  if (_isPseudoRoot)
  {
    // Drives enumeration
    TCHAR drive_buffer[100];
    GetLogicalDriveStrings(sizeof(drive_buffer) / sizeof(TCHAR), drive_buffer);

    for (TCHAR *current_drive = drive_buffer; *current_drive; 
         current_drive += _tcslen(current_drive) + 1)
    {
      WindowsFilesystemNode entry;		
      char drive_name[2];

      drive_name[0] = toAscii(current_drive)[0];
      drive_name[1] = '\0';
      entry._displayName = drive_name;
      entry._isDirectory = true;
      entry._isValid = true;
      entry._isPseudoRoot = false;
      entry._path = toAscii(current_drive);
      myList.push_back(wrap(new WindowsFilesystemNode(&entry)));
    }
  }
  else
  {
    // Files enumeration
    WIN32_FIND_DATA desc;
    HANDLE handle;
    char searchPath[MAX_PATH + 10];

    sprintf(searchPath, "%s*", _path.c_str());

    handle = FindFirstFile(toUnicode(searchPath), &desc);
    if (handle == INVALID_HANDLE_VALUE)
      return myList;

    addFile(myList, mode, _path.c_str(), &desc);
    while (FindNextFile(handle, &desc))
      addFile(myList, mode, _path.c_str(), &desc);

    FindClose(handle);
  }

  return myList;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
AbstractFilesystemNode* WindowsFilesystemNode::parent() const
{
  assert(_isValid || _isPseudoRoot);
  if (_isPseudoRoot)
    return 0;

  WindowsFilesystemNode* p = new WindowsFilesystemNode();
  if (_path.size() > 3)
  {
    const char *start = _path.c_str();
    const char *end = lastPathComponent(_path);

    p = new WindowsFilesystemNode();
    p->_path = string(start, end - start);
    p->_isValid = true;
    p->_isDirectory = true;
    p->_displayName = lastPathComponent(p->_path);
    p->_isPseudoRoot = false;
  }

  return p;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
bool AbstractFilesystemNode::fileExists(const string& path)
{
  WIN32_FILE_ATTRIBUTE_DATA attr;
  BOOL b = GetFileAttributesEx(path.c_str(), GetFileExInfoStandard, &attr);

  return ((b != 0) && !(attr.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY));
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
bool AbstractFilesystemNode::dirExists(const string& path)
{
  WIN32_FILE_ATTRIBUTE_DATA attr;
  BOOL b = GetFileAttributesEx(path.c_str(), GetFileExInfoStandard, &attr);

  return ((b != 0) && (attr.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY));
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
bool AbstractFilesystemNode::makeDir(const string& path)
{
  return CreateDirectory(path.c_str(), NULL) != 0;
}
