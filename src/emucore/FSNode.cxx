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
// $Id: FSNode.cxx,v 1.10 2007/07/31 15:46:20 stephena Exp $
//
//   Based on code from ScummVM - Scumm Interpreter
//   Copyright (C) 2002-2004 The ScummVM project
//============================================================================

#include "FSNode.hxx"
#include "bspf.hxx"
using namespace std;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void FSList::sort()
{
  // Simple selection sort
  for (Int32 i = 0; i < _size-1; i++)
  {
    Int32 min = i;
    for (Int32 j = i+1; j < _size; j++)
    {
      if (_data[j] < _data[min])
        min = j;
    }
    if (min != i)
      BSPF_swap(_data[min], _data[i]);
  }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
FilesystemNode AbstractFilesystemNode::wrap(AbstractFilesystemNode *node)
{
  FilesystemNode wrapper;
  wrapper._realNode = node;

  return wrapper;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
FilesystemNode::FilesystemNode()
{
  _realNode = getRoot();
  _refCount = new int(1);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
FilesystemNode::FilesystemNode(const FilesystemNode &node)
    : AbstractFilesystemNode()
{
  _realNode = node._realNode;
  _refCount = node._refCount;
  ++(*_refCount);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
FilesystemNode::FilesystemNode(const string& p)
{
  _realNode = getNodeForPath(p);
  _refCount = new int(1);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
FilesystemNode::~FilesystemNode()
{
  decRefCount();
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
void FilesystemNode::decRefCount()
{
  --(*_refCount);
  if (*_refCount <= 0)
  {
    delete _refCount;
    delete _realNode;
  }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
FilesystemNode &FilesystemNode::operator  =(const FilesystemNode &node)
{
  ++(*node._refCount);

  decRefCount();

  _realNode = node._realNode;
  _refCount = node._refCount;

  return *this;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
FilesystemNode FilesystemNode::getParent() const
{
  AbstractFilesystemNode *node = _realNode->parent();

  if(node == 0)
    return *this;
  else
    return AbstractFilesystemNode::wrap(node);
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
bool FilesystemNode::hasParent() const
{
  return _realNode->parent() != 0;
}
